// QuantCUDA Risk Engine - Kernel 5: parallel reduction generations.
//
// Four device-side strategies for summing N values, benchmarked head-to-head
// (this is the standalone reduction study cited by the optimization report):
//   0: one global atomicAdd per thread
//   1: shared-memory tree reduction
//   2: pure warp-shuffle reduction
//   3: warp shuffle + one shared-memory value per warp (production default)
#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {
namespace {

__global__ void fill_random_kernel(std::size_t n, std::uint64_t seed, double* out) {
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < n; i += stride) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, i, 0, &state);
        out[i] = curand_normal_double(&state);
    }
}

__global__ void reduce_atomic_kernel(const double* in, std::size_t n, double* total) {
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < n; i += stride)
        atomicAdd(total, in[i]);
}

__global__ void reduce_smem_kernel(const double* in, std::size_t n,
                                   double* block_out) {
    double v = 0.0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < n; i += stride) v += in[i];
    extern __shared__ double smem[];
    double total = dev::block_reduce_smem(v, smem);
    if (threadIdx.x == 0) block_out[blockIdx.x] = total;
}

__global__ void reduce_warp_kernel(const double* in, std::size_t n,
                                   double* block_out) {
    double v = 0.0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < n; i += stride) v += in[i];
    v = dev::warp_reduce(v);
    if ((threadIdx.x & 31) == 0) atomicAdd(block_out, v);  // one atomic per warp
}

__global__ void reduce_warp_smem_kernel(const double* in, std::size_t n,
                                        double* block_out) {
    double v = 0.0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < n; i += stride) v += in[i];
    extern __shared__ double smem[];
    double total = dev::block_reduce_warp(v, smem);
    if (threadIdx.x == 0) block_out[blockIdx.x] = total;
}

template <typename LaunchT>
ReductionRow time_one(int flavor, const char* name, std::size_t n, int block_size,
                      const double* d_in, LaunchT launch) {
    int grid = dev::grid_for(n, block_size);
    DeviceBuffer<double> out(static_cast<std::size_t>(grid));
    QC_CUDA_CHECK(cudaMemset(out.ptr, 0, out.n * sizeof(double)));
    // warm-up
    launch(grid, block_size, d_in, out.ptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    // Atomic flavors accumulate into out[0], so the measured run must start
    // from the same zero state as the warm-up run.
    QC_CUDA_CHECK(cudaMemset(out.ptr, 0, out.n * sizeof(double)));
    EventTimer ev;
    ev.begin();
    launch(grid, block_size, d_in, out.ptr);
    QC_CUDA_CHECK(cudaGetLastError());
    float ms = ev.end();
    std::vector<double> h(out.n);
    out.copy_to(h.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    double total = 0.0;
    for (double x : h) total += x;
    ReductionRow row;
    row.flavor = flavor;
    row.name = name;
    row.kernel_ms = ms;
    row.result = total;
    return row;
}

}  // namespace

std::vector<ReductionRow> reduction_study(std::size_t n, std::uint64_t seed,
                                          int block_size) {
    if (n == 0 || n > 100'000'000)
        throw std::invalid_argument(
            "reduction study: values must be in [1, 100000000]");
    if (block_size < 32 || block_size > 1024 ||
        (block_size & (block_size - 1)) != 0)
        throw std::invalid_argument(
            "reduction study: block size must be a power of two in [32, 1024]");
    int device = 0;
    QC_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    QC_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (block_size > prop.maxThreadsPerBlock)
        throw std::invalid_argument(
            "reduction study: block size exceeds this device's limit");
    DeviceBuffer<double> d_in(n);
    int grid = dev::grid_for(n, block_size);
    fill_random_kernel<<<grid, block_size>>>(n, seed, d_in.ptr);
    QC_CUDA_CHECK(cudaGetLastError());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(double);
    if (smem < 32 * sizeof(double)) smem = 32 * sizeof(double);

    std::vector<ReductionRow> rows;
    rows.push_back(time_one(0, "atomic-per-thread", n, block_size, d_in.ptr,
                            [&](int g, int b, const double* in, double* out) {
                                reduce_atomic_kernel<<<g, b>>>(in, n, out);
                            }));
    rows.push_back(time_one(1, "shared-memory-tree", n, block_size, d_in.ptr,
                            [&](int g, int b, const double* in, double* out) {
                                reduce_smem_kernel<<<g, b, smem>>>(in, n, out);
                            }));
    rows.push_back(time_one(2, "warp-shuffle+atomic", n, block_size, d_in.ptr,
                            [&](int g, int b, const double* in, double* out) {
                                reduce_warp_kernel<<<g, b>>>(in, n, out);
                            }));
    rows.push_back(time_one(3, "warp-shuffle+smem", n, block_size, d_in.ptr,
                            [&](int g, int b, const double* in, double* out) {
                                reduce_warp_smem_kernel<<<g, b, smem>>>(in, n, out);
                            }));
    return rows;
}

}  // namespace quantcuda::gpu
