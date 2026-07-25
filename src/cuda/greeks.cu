// QuantCUDA Risk Engine - Kernel: fused pricing + Greeks.
//
// One launch, one draw per path, FIVE payoffs: V(S-h), V(S), V(S+h) for
// central delta/gamma and V(sigma-h), V(sigma+h) for vega - common random
// numbers cancel most of the sampling noise in the differences. A separate
// pathwise-delta kernel implements the GBM estimator
//     e^{-rT} * 1{S_T > K} * S_T / S0
// for comparison against Black-Scholes and the FD estimator.
#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {
namespace {

// Five mergeable moment sets per block. Besides making the price SE stable,
// their means avoid needlessly summing large bumped payoffs on the host.
struct GreekPartials {
    dev::MomentPartial dn, base, up, vol_dn, vol_up;
};

template <typename Real>
__device__ Real draw_normal_g(curandStatePhilox4_32_10_t* s);
template <>
__device__ double draw_normal_g<double>(curandStatePhilox4_32_10_t* s) {
    return curand_normal_double(s);
}

template <typename Real>
__device__ Real payoff_at(int is_call, Real s0, Real k, Real drift, Real diff, Real z) {
    Real st = s0 * exp(drift + diff * z);
    return dev::european_payoff_d<Real>(is_call, st, k);
}

template <typename Real>
__global__ void greeks_crn_kernel(dev::OptionParams opt, Real h, Real h_sigma,
                                  curandStatePhilox4_32_10_t* states,
                                  GreekPartials* block_out, std::size_t path_count) {
    Real k = static_cast<Real>(opt.strike);
    Real s0 = static_cast<Real>(opt.spot);
    Real r = static_cast<Real>(opt.rate), q = static_cast<Real>(opt.dividend);
    Real sig = static_cast<Real>(opt.volatility);
    Real T = static_cast<Real>(opt.maturity);

    Real drift = dev::gbm_drift<Real>(r, q, sig, T);
    Real diff = dev::gbm_diffusion<Real>(sig, T);
    // Match the CPU estimator at the volatility boundary: use a forward
    // difference from sigma when a symmetric down bump would become negative.
    const bool central_vega = sig >= h_sigma;
    Real sig_dn = central_vega ? sig - h_sigma : sig;
    Real drift_vu = dev::gbm_drift<Real>(r, q, sig + h_sigma, T);
    Real drift_vd = dev::gbm_drift<Real>(r, q, sig_dn, T);
    Real diff_vu = dev::gbm_diffusion<Real>(sig + h_sigma, T);
    Real diff_vd = dev::gbm_diffusion<Real>(sig_dn, T);

    dev::RunningMoments<double> dn, base, up, vdn, vup;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real z = draw_normal_g<Real>(&state);  // ONE draw prices all five points
        double pb = static_cast<double>(payoff_at(opt.is_call, s0, k, drift, diff, z));
        dev::moments_add(
            dn, static_cast<double>(payoff_at(opt.is_call, s0 - h, k, drift, diff, z)));
        dev::moments_add(base, pb);
        dev::moments_add(
            up, static_cast<double>(payoff_at(opt.is_call, s0 + h, k, drift, diff, z)));
        dev::moments_add(
            vdn, static_cast<double>(payoff_at(opt.is_call, s0, k, drift_vd, diff_vd, z)));
        dev::moments_add(
            vup, static_cast<double>(payoff_at(opt.is_call, s0, k, drift_vu, diff_vu, z)));
    }
    extern __shared__ char smem_raw[];
    auto* red = reinterpret_cast<dev::RunningMoments<double>*>(smem_raw);
    const auto t_dn = dev::block_reduce_moments_warp(dn, red);
    __syncthreads();
    const auto t_base = dev::block_reduce_moments_warp(base, red);
    __syncthreads();
    const auto t_up = dev::block_reduce_moments_warp(up, red);
    __syncthreads();
    const auto t_vdn = dev::block_reduce_moments_warp(vdn, red);
    __syncthreads();
    const auto t_vup = dev::block_reduce_moments_warp(vup, red);
    if (threadIdx.x == 0) {
        block_out[blockIdx.x] = {
            dev::moments_partial(t_dn), dev::moments_partial(t_base),
            dev::moments_partial(t_up), dev::moments_partial(t_vdn),
            dev::moments_partial(t_vup)};
    }
}

template <typename Real>
__global__ void pathwise_delta_kernel(dev::OptionParams opt,
                                      curandStatePhilox4_32_10_t* states,
                                      dev::MomentPartial* block_moments,
                                      std::size_t path_count) {
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility),
                                      static_cast<Real>(opt.maturity));
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility),
                                         static_cast<Real>(opt.maturity));
    Real k = static_cast<Real>(opt.strike);
    Real s0 = static_cast<Real>(opt.spot);
    dev::RunningMoments<double> moments;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real st = s0 * exp(drift + diff * draw_normal_g<Real>(&state));
        bool itm = opt.is_call ? (st > k) : (st < k);
        double c = itm ? static_cast<double>(st / s0) * (opt.is_call ? 1.0 : -1.0)
                       : 0.0;
        dev::moments_add(moments, c);
    }
    extern __shared__ char smem_raw[];
    auto* red = reinterpret_cast<dev::RunningMoments<double>*>(smem_raw);
    const auto total = dev::block_reduce_moments_warp(moments, red);
    if (threadIdx.x == 0) block_moments[blockIdx.x] = dev::moments_partial(total);
}

}  // namespace

GpuGreeksRun greeks_crn_gpu(const dev::OptionParams& opt, std::size_t paths,
                            std::uint64_t seed, double h, double h_sigma,
                            int block_size) {
    if (paths == 0) return GpuGreeksRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<GreekPartials> partials(static_cast<std::size_t>(grid));
    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer ev;
    std::size_t smem = 32 * sizeof(dev::RunningMoments<double>);
    ev.begin();
    greeks_crn_kernel<double><<<grid, block_size, smem>>>(
        opt, h, h_sigma, states.ptr, partials.ptr, paths);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<GreekPartials> host_parts(static_cast<std::size_t>(grid));
    QC_CUDA_CHECK(cudaMemcpy(host_parts.data(), partials.ptr,
                             host_parts.size() * sizeof(GreekPartials),
                             cudaMemcpyDeviceToHost));
    dev::RunningMoments<double> dn, base, up, vdn, vup;
    auto merge_partial = [](dev::RunningMoments<double>& total,
                            const dev::MomentPartial& partial) {
        total = dev::moments_merge(
            total, dev::RunningMoments<double>{partial.count, partial.mean,
                                                partial.m2});
    };
    for (const auto& p : host_parts) {
        merge_partial(dn, p.dn);
        merge_partial(base, p.base);
        merge_partial(up, p.up);
        merge_partial(vdn, p.vol_dn);
        merge_partial(vup, p.vol_up);
    }
    double df = exp(-opt.rate * opt.maturity);
    GpuGreeksRun out;
    const McResult dn_stats = finalize_moments(dn, paths, 1.0);
    const McResult base_stats = finalize_moments(base, paths, df);
    const McResult up_stats = finalize_moments(up, paths, 1.0);
    const McResult vdn_stats = finalize_moments(vdn, paths, 1.0);
    const McResult vup_stats = finalize_moments(vup, paths, 1.0);
    out.price = base_stats.price;
    out.price_se = base_stats.std_error;
    out.delta = df * (up_stats.mean_payoff - dn_stats.mean_payoff) / (2.0 * h);
    out.gamma = df * (up_stats.mean_payoff -
                      2.0 * base_stats.mean_payoff + dn_stats.mean_payoff) /
                (h * h);
    const double vega_denominator = opt.volatility >= h_sigma
                                        ? 2.0 * h_sigma
                                        : h_sigma;
    out.vega = df * (vup_stats.mean_payoff - vdn_stats.mean_payoff) /
               vega_denominator;
    out.kernel_ms = kernel_ms;
    return out;
}

GpuMcRun delta_pathwise_gpu(const dev::OptionParams& opt, std::size_t paths,
                            std::uint64_t seed, int block_size) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<dev::MomentPartial> moments(static_cast<std::size_t>(grid));
    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    EventTimer ev;
    std::size_t smem = 32 * sizeof(dev::RunningMoments<double>);
    ev.begin();
    pathwise_delta_kernel<double><<<grid, block_size, smem>>>(
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

}  // namespace quantcuda::gpu
