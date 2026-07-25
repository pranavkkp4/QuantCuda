// QuantCUDA Risk Engine - Kernel: correlated multi-asset shocks.
//
// The Cholesky factor L of the correlation matrix is computed ONCE on the
// host (reused across millions of scenarios) and uploaded to the device.
// Two device strategies (the V5 memory-hierarchy comparison):
//   * warp kernel : one warp per scenario, L in __constant__ memory (d <= 64),
//                   lanes exchange independent normals with __shfl_sync,
//   * block kernel: one block per scenario for larger d, the scenario's
//                   independent normals staged in shared memory, L streamed
//                   from global memory with __restrict__ const hints.
#include <algorithm>

#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {
namespace {

constexpr int kMaxConstDim = 64;  // 64x64 floats = 16KB of the 64KB constant bank
__constant__ float c_chol[kMaxConstDim * kMaxConstDim];

// ---------------- warp-per-scenario kernel (d <= 64) --------------------------
// Lane L holds the independent normals for components L and L+32 (z_local[0],
// z_local[1]). Every lane participates in every output-row reduction, which is
// required by the full-warp shuffle contract.
__global__ void correlated_shocks_warp_kernel(std::size_t scenarios, int d,
                                              std::uint64_t seed,
                                              double* out /* scenarios x d */) {
    int lane = threadIdx.x & 31;
    std::size_t warp = dev::global_thread_id() >> 5;
    std::size_t nwarps = dev::grid_stride() >> 5;
    for (std::size_t s = warp; s < scenarios; s += nwarps) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, s * kMaxConstDim + static_cast<std::size_t>(lane), 0,
                    &state);
        float z_local[2];
        z_local[0] = curand_normal(&state);
        z_local[1] = curand_normal(&state);

        for (int i = 0; i < d; ++i) {
            float contribution = 0.0f;
            const int k0 = lane;
            const int k1 = lane + 32;
            if (k0 <= i && k0 < d)
                contribution += c_chol[i * d + k0] * z_local[0];
            if (k1 <= i && k1 < d)
                contribution += c_chol[i * d + k1] * z_local[1];
            float acc = dev::warp_reduce(contribution);
            if (lane == 0)
                out[s * static_cast<std::size_t>(d) + static_cast<std::size_t>(i)] =
                    static_cast<double>(acc);
        }
    }
}

// ---------------- block-per-scenario kernel (any d) ---------------------------
__global__ void correlated_shocks_block_kernel(std::size_t scenarios, int d,
                                               const float* __restrict__ chol,
                                               std::uint64_t seed,
                                               double* __restrict__ out) {
    extern __shared__ float s_z[];  // d floats: this scenario's independent normals
    for (std::size_t s = blockIdx.x; s < scenarios; s += gridDim.x) {
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            curandStatePhilox4_32_10_t state;
            curand_init(seed, s * 4096 + static_cast<std::size_t>(i), 0, &state);
            s_z[i] = curand_normal(&state);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k <= i; ++k) acc += chol[i * d + k] * s_z[k];
            out[s * static_cast<std::size_t>(d) + i] = static_cast<double>(acc);
        }
        __syncthreads();
    }
}

}  // namespace

std::vector<double> correlated_shocks_gpu(const std::vector<double>& chol,
                                          std::size_t d, std::size_t scenarios,
                                          std::uint64_t seed,
                                          bool use_shared_tiles,
                                          double* kernel_ms_out) {
    std::vector<double> out_host(scenarios * d);
    if (scenarios == 0 || d == 0) {
        if (kernel_ms_out) *kernel_ms_out = 0.0;
        return out_host;
    }
    DeviceBuffer<double> out(scenarios * d);
    EventTimer ev;

    if (d <= kMaxConstDim && !use_shared_tiles) {
        std::vector<float> cf(d * d);
        for (std::size_t i = 0; i < d * d; ++i) cf[i] = static_cast<float>(chol[i]);
        QC_CUDA_CHECK(cudaMemcpyToSymbol(c_chol, cf.data(), cf.size() * sizeof(float)));
        int block = 256;  // 8 warps per block
        std::size_t blocks_needed = (scenarios + 7) / 8;
        int blocks = static_cast<int>(
            std::min<std::size_t>(std::max<std::size_t>(blocks_needed, 1), 65535));
        ev.begin();
        correlated_shocks_warp_kernel<<<blocks, block>>>(
            scenarios, static_cast<int>(d), seed, out.ptr);
        QC_CUDA_CHECK(cudaGetLastError());
        if (kernel_ms_out) *kernel_ms_out = ev.end();
        else (void)ev.end();
    } else {
        DeviceBuffer<float> chol_dev(d * d);
        std::vector<float> cf(d * d);
        for (std::size_t i = 0; i < d * d; ++i) cf[i] = static_cast<float>(chol[i]);
        chol_dev.copy_from(cf.data());
        QC_CUDA_CHECK(cudaDeviceSynchronize());
        int block = 256;
        int grid = static_cast<int>(std::min<std::size_t>(scenarios, 65535));
        std::size_t smem = d * sizeof(float);
        ev.begin();
        correlated_shocks_block_kernel<<<grid, block, smem>>>(
            scenarios, static_cast<int>(d), chol_dev.ptr, seed, out.ptr);
        QC_CUDA_CHECK(cudaGetLastError());
        if (kernel_ms_out) *kernel_ms_out = ev.end();
        else (void)ev.end();
    }
    out.copy_to(out_host.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    return out_host;
}

}  // namespace quantcuda::gpu
