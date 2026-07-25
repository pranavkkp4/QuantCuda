// QuantCUDA Risk Engine - Kernel 4: barrier options (warp-divergence study).
//
// Three implementations, all preserved for profiling:
//   mode 0  early-exit branch:  if (price >= barrier) { alive = false; break; }
//   mode 1  boolean mask:       alive &= price < barrier;   (no break)
//   mode 2  compacted active paths (stretch goal): paths knocked out during
//           the first half of the schedule are stream-compacted away with
//           Thrust before the second half is simulated.
//
// Which is faster DEPENDS on the barrier position - the benchmark sweeps
// near/medium/far barriers instead of assuming (see docs/performance_report).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {
namespace {

template <typename Real>
__device__ Real draw_normal(curandStatePhilox4_32_10_t* s);
template <>
__device__ float draw_normal<float>(curandStatePhilox4_32_10_t* s) {
    return curand_normal(s);
}
template <>
__device__ double draw_normal<double>(curandStatePhilox4_32_10_t* s) {
    return curand_normal_double(s);
}

__device__ __forceinline__ bool breached(int dir, double price, double barrier) {
    return dir > 0 ? (price >= barrier) : (price <= barrier);
}

// ---------------- mode 0: early-exit branch ---------------------------------
template <typename Real, typename AccT>
__global__ void barrier_early_exit_kernel(dev::OptionParams opt,
                                          curandStatePhilox4_32_10_t* states,
                                          double* block_sums, double* block_sq_sums,
                                          unsigned long long* knockout_count,
                                          std::size_t path_count) {
    int steps = opt.time_steps;
    Real dt = static_cast<Real>(opt.maturity) / static_cast<Real>(steps);
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility), dt);
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility), dt);
    Real k = static_cast<Real>(opt.strike);
    Real b = static_cast<Real>(opt.barrier);

    AccT sum = AccT(0), sq = AccT(0);
    unsigned long long local_ko = 0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real s = static_cast<Real>(opt.spot);
        bool alive = !breached(opt.barrier_dir, s, b);
        for (int j = 0; j < steps && alive; ++j) {
            s *= exp(drift + diff * draw_normal<Real>(&state));
            if (breached(opt.barrier_dir, s, b)) {  // divergent branch
                alive = false;
                break;                              // early exit
            }
        }
        if (!alive) {
            ++local_ko;
            continue;
        }
        AccT p = static_cast<AccT>(opt.is_call ? fmax(s - k, Real(0))
                                               : fmax(k - s, Real(0)));
        sum += p;
        sq += p * p;
    }
    if (knockout_count && local_ko) atomicAdd(knockout_count, local_ko);
    extern __shared__ char smem_raw[];
    AccT* red = reinterpret_cast<AccT*>(smem_raw);
    AccT total_sum = dev::block_reduce_warp(sum, red);
    __syncthreads();
    AccT total_sq = dev::block_reduce_warp(sq, red);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = static_cast<double>(total_sum);
        block_sq_sums[blockIdx.x] = static_cast<double>(total_sq);
    }
}

// ---------------- mode 1: boolean alive mask ---------------------------------
template <typename Real, typename AccT>
__global__ void barrier_mask_kernel(dev::OptionParams opt,
                                    curandStatePhilox4_32_10_t* states,
                                    double* block_sums, double* block_sq_sums,
                                    unsigned long long* knockout_count,
                                    std::size_t path_count) {
    int steps = opt.time_steps;
    Real dt = static_cast<Real>(opt.maturity) / static_cast<Real>(steps);
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility), dt);
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility), dt);
    Real k = static_cast<Real>(opt.strike);
    Real b = static_cast<Real>(opt.barrier);

    AccT sum = AccT(0), sq = AccT(0);
    unsigned long long local_ko = 0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real s = static_cast<Real>(opt.spot);
        bool alive = !breached(opt.barrier_dir, s, b);
        for (int j = 0; j < steps; ++j) {
            s *= exp(drift + diff * draw_normal<Real>(&state));
            alive = alive && !breached(opt.barrier_dir, s, b);  // no break
        }
        if (!alive) ++local_ko;
        AccT p = alive ? static_cast<AccT>(opt.is_call ? fmax(s - k, Real(0))
                                                       : fmax(k - s, Real(0)))
                       : AccT(0);
        sum += p;
        sq += p * p;
    }
    if (knockout_count && local_ko) atomicAdd(knockout_count, local_ko);
    extern __shared__ char smem_raw[];
    AccT* red = reinterpret_cast<AccT*>(smem_raw);
    AccT total_sum = dev::block_reduce_warp(sum, red);
    __syncthreads();
    AccT total_sq = dev::block_reduce_warp(sq, red);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = static_cast<double>(total_sum);
        block_sq_sums[blockIdx.x] = static_cast<double>(total_sq);
    }
}

// ---------------- mode 2: two-phase stream compaction -------------------------
// Phase 1: simulate the first half of the schedule, record survivors.
// (Double precision: this stretch-goal path trades the precision study for a
// clean compaction implementation.)
__global__ void barrier_phase1_kernel(dev::OptionParams opt,
                                      curandStatePhilox4_32_10_t* states,
                                      double* path_prices, int* alive_flags,
                                      unsigned long long* knockout_count,
                                      int half_steps, std::size_t path_count) {
    double dt = opt.maturity / static_cast<double>(opt.time_steps);
    double drift = dev::gbm_drift<double>(opt.rate, opt.dividend, opt.volatility, dt);
    double diff = dev::gbm_diffusion<double>(opt.volatility, dt);
    double b = opt.barrier;
    unsigned long long local_ko = 0;
    const std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        double s = opt.spot;
        bool alive = !breached(opt.barrier_dir, s, b);
        for (int j = 0; j < half_steps && alive; ++j) {
            s *= exp(drift + diff * curand_normal_double(&state));
            alive = !breached(opt.barrier_dir, s, b);
        }
        path_prices[i] = s;
        alive_flags[i] = alive ? 1 : 0;
        states[i] = state;  // phase 2 resumes the exact seeded RNG stream
        if (!alive) ++local_ko;
    }
    if (local_ko) atomicAdd(knockout_count, local_ko);
}

// Phase 2: continue ONLY the compacted survivor set to maturity.
__global__ void barrier_phase2_kernel(dev::OptionParams opt,
                                      const int* survivor_ids,
                                      std::size_t survivor_count,
                                      const double* path_prices,
                                      curandStatePhilox4_32_10_t* states,
                                      int half_steps, double* block_sums,
                                      double* block_sq_sums,
                                      unsigned long long* knockout_count) {
    int steps = opt.time_steps;
    double dt = opt.maturity / static_cast<double>(steps);
    double drift = dev::gbm_drift<double>(opt.rate, opt.dividend, opt.volatility, dt);
    double diff = dev::gbm_diffusion<double>(opt.volatility, dt);
    double k = opt.strike;
    double b = opt.barrier;
    double sum = 0.0, sq = 0.0;
    unsigned long long local_ko = 0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t t = dev::global_thread_id(); t < survivor_count; t += stride) {
        int i = survivor_ids[t];
        curandStatePhilox4_32_10_t state = states[i];
        double s = path_prices[i];
        bool alive = true;
        for (int j = half_steps; j < steps && alive; ++j) {
            s *= exp(drift + diff * curand_normal_double(&state));
            alive = !breached(opt.barrier_dir, s, b);
        }
        if (!alive) ++local_ko;
        double p = alive ? (opt.is_call ? fmax(s - k, 0.0) : fmax(k - s, 0.0)) : 0.0;
        sum += p;
        sq += p * p;
    }
    if (local_ko) atomicAdd(knockout_count, local_ko);
    extern __shared__ double red[];
    double total_sum = dev::block_reduce_warp(sum, red);
    __syncthreads();
    double total_sq = dev::block_reduce_warp(sq, red);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = total_sum;
        block_sq_sums[blockIdx.x] = total_sq;
    }
}

struct IsAlive {
    __device__ bool operator()(int flag) const { return flag != 0; }
};

template <typename Real, typename AccT, typename KernelT>
GpuMcRun run_branchy(const dev::OptionParams& opt, std::size_t paths,
                     std::uint64_t seed, int block_size, KernelT kernel) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<double> sums(static_cast<std::size_t>(grid));
    DeviceBuffer<double> sqs(static_cast<std::size_t>(grid));
    DeviceBuffer<unsigned long long> ko(1);
    QC_CUDA_CHECK(cudaMemset(ko.ptr, 0, sizeof(unsigned long long)));
    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer ev;
    std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(double);
    if (smem < 32 * sizeof(double)) smem = 32 * sizeof(double);
    ev.begin();
    kernel<<<grid, block_size, smem>>>(opt, states.ptr, sums.ptr, sqs.ptr, ko.ptr,
                                       paths);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<double> h_sums(static_cast<std::size_t>(grid));
    std::vector<double> h_sqs(static_cast<std::size_t>(grid));
    sums.copy_to(h_sums.data());
    sqs.copy_to(h_sqs.data());
    unsigned long long h_ko = 0;
    QC_CUDA_CHECK(cudaMemcpy(&h_ko, ko.ptr, sizeof(h_ko), cudaMemcpyDeviceToHost));
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    GpuMcRun out;
    out.stats = finalize_partials(h_sums, h_sqs, paths, exp(-opt.rate * opt.maturity));
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    out.extra = paths ? static_cast<double>(h_ko) / static_cast<double>(paths) : 0.0;
    return out;
}

GpuMcRun run_compacted(const dev::OptionParams& opt, std::size_t paths,
                       std::uint64_t seed, int block_size) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    int half = opt.time_steps / 2;
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<double> path_prices(paths);
    DeviceBuffer<int> flags(paths);
    DeviceBuffer<int> survivors(paths);
    DeviceBuffer<int> ids(paths);
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    DeviceBuffer<double> sums(static_cast<std::size_t>(grid));
    DeviceBuffer<double> sqs(static_cast<std::size_t>(grid));
    DeviceBuffer<unsigned long long> ko(1);
    QC_CUDA_CHECK(cudaMemset(sums.ptr, 0, sums.n * sizeof(double)));
    QC_CUDA_CHECK(cudaMemset(sqs.ptr, 0, sqs.n * sizeof(double)));
    QC_CUDA_CHECK(cudaMemset(ko.ptr, 0, sizeof(unsigned long long)));
    init_rng(states.ptr, seed, paths, block_size, nullptr);

    EventTimer ev;
    ev.begin();
    barrier_phase1_kernel<<<grid, block_size>>>(opt, states.ptr, path_prices.ptr,
                                                 flags.ptr, ko.ptr, half, paths);
    QC_CUDA_CHECK(cudaGetLastError());
    // Stream compaction: survivor ids packed to the front.
    thrust::sequence(thrust::device, ids.ptr, ids.ptr + paths, 0);
    int* end = thrust::copy_if(thrust::device, ids.ptr, ids.ptr + paths, flags.ptr,
                               survivors.ptr, IsAlive{});
    std::size_t survivor_count = static_cast<std::size_t>(end - survivors.ptr);

    if (survivor_count > 0) {
        std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(double);
        if (smem < 32 * sizeof(double)) smem = 32 * sizeof(double);
        int survivor_grid = dev::grid_for(survivor_count, block_size);
        barrier_phase2_kernel<<<survivor_grid, block_size, smem>>>(
            opt, survivors.ptr, survivor_count, path_prices.ptr, states.ptr, half,
            sums.ptr, sqs.ptr, ko.ptr);
    }
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<double> h_sums(static_cast<std::size_t>(grid));
    std::vector<double> h_sqs(static_cast<std::size_t>(grid));
    sums.copy_to(h_sums.data());
    sqs.copy_to(h_sqs.data());
    unsigned long long h_ko = 0;
    QC_CUDA_CHECK(cudaMemcpy(&h_ko, ko.ptr, sizeof(h_ko), cudaMemcpyDeviceToHost));
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    GpuMcRun out;
    out.stats = finalize_partials(h_sums, h_sqs, paths, exp(-opt.rate * opt.maturity));
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    out.extra = static_cast<double>(h_ko) / static_cast<double>(paths);
    return out;
}

}  // namespace

GpuMcRun barrier_gpu(const dev::OptionParams& opt, std::size_t paths,
                     std::uint64_t seed, Precision precision, int block_size,
                     int mode) {
    if (mode == 2) return run_compacted(opt, paths, seed, block_size);
    if (mode == 1) {
        switch (precision) {
            case Precision::Float:
                return run_branchy<float, float>(opt, paths, seed, block_size,
                                                 barrier_mask_kernel<float, float>);
            case Precision::Mixed:
                return run_branchy<float, double>(opt, paths, seed, block_size,
                                                  barrier_mask_kernel<float, double>);
            case Precision::Double:
            default:
                return run_branchy<double, double>(opt, paths, seed, block_size,
                                                   barrier_mask_kernel<double, double>);
        }
    }
    switch (precision) {
        case Precision::Float:
            return run_branchy<float, float>(opt, paths, seed, block_size,
                                             barrier_early_exit_kernel<float, float>);
        case Precision::Mixed:
            return run_branchy<float, double>(opt, paths, seed, block_size,
                                              barrier_early_exit_kernel<float, double>);
        case Precision::Double:
        default:
            return run_branchy<double, double>(opt, paths, seed, block_size,
                                               barrier_early_exit_kernel<double, double>);
    }
}

}  // namespace quantcuda::gpu
