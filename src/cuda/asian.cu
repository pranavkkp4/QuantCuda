// QuantCUDA Risk Engine - Kernel 3: arithmetic-average Asian options.
//
// Each thread simulates one entire path and keeps ONLY the running price and
// running sum - the full path is never stored, which keeps memory traffic at
// O(1) per path regardless of the step count. Benchmarked at 12/52/252 steps
// (see benchmark suite B).
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

template <typename Real, typename AccT>
__global__ void asian_payoff_kernel(dev::OptionParams opt,
                                    curandStatePhilox4_32_10_t* states,
                                    dev::MomentPartial* block_moments,
                                    std::size_t path_count) {
    int steps = opt.time_steps;
    Real dt = static_cast<Real>(opt.maturity) / static_cast<Real>(steps);
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility), dt);
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility), dt);
    Real k = static_cast<Real>(opt.strike);

    dev::RunningMoments<AccT> moments;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real s = static_cast<Real>(opt.spot);
        Real running_sum = Real(0);
        for (int j = 0; j < steps; ++j) {
            s *= exp(drift + diff * draw_normal<Real>(&state));
            running_sum += s;
        }
        Real avg = running_sum / static_cast<Real>(steps);
        AccT p = static_cast<AccT>(opt.is_call ? fmax(avg - k, Real(0))
                                               : fmax(k - avg, Real(0)));
        dev::moments_add(moments, p);
    }
    extern __shared__ char smem_raw[];
    auto* red = reinterpret_cast<dev::RunningMoments<AccT>*>(smem_raw);
    const auto total = dev::block_reduce_moments_warp(moments, red);
    if (threadIdx.x == 0) block_moments[blockIdx.x] = dev::moments_partial(total);
}

template <typename Real, typename AccT>
GpuMcRun run(const dev::OptionParams& opt, std::size_t paths, std::uint64_t seed,
             int block_size) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<dev::MomentPartial> moments(static_cast<std::size_t>(grid));
    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer ev;
    std::size_t smem = 32 * sizeof(dev::RunningMoments<AccT>);
    ev.begin();
    asian_payoff_kernel<Real, AccT><<<grid, block_size, smem>>>(
        opt, states.ptr, moments.ptr, paths);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<dev::MomentPartial> h_moments(static_cast<std::size_t>(grid));
    moments.copy_to(h_moments.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    GpuMcRun out;
    out.stats = finalize_partials(h_moments, paths, exp(-opt.rate * opt.maturity));
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    return out;
}

}  // namespace

GpuMcRun asian_gpu(const dev::OptionParams& opt, std::size_t paths,
                   std::uint64_t seed, Precision precision, int block_size) {
    switch (precision) {
        case Precision::Float: return run<float, float>(opt, paths, seed, block_size);
        case Precision::Mixed: return run<float, double>(opt, paths, seed, block_size);
        case Precision::Double:
        default: return run<double, double>(opt, paths, seed, block_size);
    }
}

}  // namespace quantcuda::gpu
