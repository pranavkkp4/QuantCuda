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

// Five accumulators per block.
struct GreekPartials {
    double dn, base, up, vol_dn, vol_up;
    double sq_base;  // for the price standard error
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

    double dn = 0, base = 0, up = 0, vdn = 0, vup = 0, sq = 0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real z = draw_normal_g<Real>(&state);  // ONE draw prices all five points
        double pb = static_cast<double>(payoff_at(opt.is_call, s0, k, drift, diff, z));
        dn += static_cast<double>(payoff_at(opt.is_call, s0 - h, k, drift, diff, z));
        base += pb;
        up += static_cast<double>(payoff_at(opt.is_call, s0 + h, k, drift, diff, z));
        vdn += static_cast<double>(payoff_at(opt.is_call, s0, k, drift_vd, diff_vd, z));
        vup += static_cast<double>(payoff_at(opt.is_call, s0, k, drift_vu, diff_vu, z));
        sq += pb * pb;
    }
    extern __shared__ double red[];
    double t_dn = dev::block_reduce_warp(dn, red);
    __syncthreads();
    double t_base = dev::block_reduce_warp(base, red);
    __syncthreads();
    double t_up = dev::block_reduce_warp(up, red);
    __syncthreads();
    double t_vdn = dev::block_reduce_warp(vdn, red);
    __syncthreads();
    double t_vup = dev::block_reduce_warp(vup, red);
    __syncthreads();
    double t_sq = dev::block_reduce_warp(sq, red);
    if (threadIdx.x == 0) {
        block_out[blockIdx.x] = {t_dn, t_base, t_up, t_vdn, t_vup, t_sq};
    }
}

template <typename Real>
__global__ void pathwise_delta_kernel(dev::OptionParams opt,
                                      curandStatePhilox4_32_10_t* states,
                                      double* block_sums, double* block_sq_sums,
                                      std::size_t path_count) {
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility),
                                      static_cast<Real>(opt.maturity));
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility),
                                         static_cast<Real>(opt.maturity));
    Real k = static_cast<Real>(opt.strike);
    Real s0 = static_cast<Real>(opt.spot);
    double sum = 0, sq = 0;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real st = s0 * exp(drift + diff * draw_normal_g<Real>(&state));
        bool itm = opt.is_call ? (st > k) : (st < k);
        double c = itm ? static_cast<double>(st / s0) * (opt.is_call ? 1.0 : -1.0)
                       : 0.0;
        sum += c;
        sq += c * c;
    }
    extern __shared__ double red2[];
    double ts = dev::block_reduce_warp(sum, red2);
    __syncthreads();
    double tq = dev::block_reduce_warp(sq, red2);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = ts;
        block_sq_sums[blockIdx.x] = tq;
    }
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
    std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(double);
    if (smem < 32 * sizeof(double)) smem = 32 * sizeof(double);
    ev.begin();
    greeks_crn_kernel<double><<<grid, block_size, smem>>>(
        opt, h, h_sigma, states.ptr, partials.ptr, paths);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<GreekPartials> host_parts(static_cast<std::size_t>(grid));
    QC_CUDA_CHECK(cudaMemcpy(host_parts.data(), partials.ptr,
                             host_parts.size() * sizeof(GreekPartials),
                             cudaMemcpyDeviceToHost));
    double dn = 0, base = 0, up = 0, vdn = 0, vup = 0, sq = 0;
    for (const auto& p : host_parts) {
        dn += p.dn; base += p.base; up += p.up;
        vdn += p.vol_dn; vup += p.vol_up; sq += p.sq_base;
    }
    double n = static_cast<double>(paths);
    double df = exp(-opt.rate * opt.maturity);
    GpuGreeksRun out;
    double mean = base / n;
    const double centered_sq = stable_centered_sum_squares(base, sq, n);
    const double var = paths > 1 ? centered_sq / (n - 1.0) : 0.0;
    out.price = df * mean;
    out.price_se = df * sqrt(var / n);
    out.delta = df * (up / n - dn / n) / (2.0 * h);
    out.gamma = df * (up / n - 2.0 * base / n + dn / n) / (h * h);
    const double vega_denominator = opt.volatility >= h_sigma
                                        ? 2.0 * h_sigma
                                        : h_sigma;
    out.vega = df * (vup / n - vdn / n) / vega_denominator;
    out.kernel_ms = kernel_ms;
    return out;
}

GpuMcRun delta_pathwise_gpu(const dev::OptionParams& opt, std::size_t paths,
                            std::uint64_t seed, int block_size) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<double> sums(static_cast<std::size_t>(grid));
    DeviceBuffer<double> sqs(static_cast<std::size_t>(grid));
    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    EventTimer ev;
    std::size_t smem = static_cast<std::size_t>(block_size) * sizeof(double);
    if (smem < 32 * sizeof(double)) smem = 32 * sizeof(double);
    ev.begin();
    pathwise_delta_kernel<double><<<grid, block_size, smem>>>(
        opt, states.ptr, sums.ptr, sqs.ptr, paths);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();
    std::vector<double> h_sums(static_cast<std::size_t>(grid));
    std::vector<double> h_sqs(static_cast<std::size_t>(grid));
    sums.copy_to(h_sums.data());
    sqs.copy_to(h_sqs.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    GpuMcRun out;
    out.stats = finalize_partials(h_sums, h_sqs, paths, exp(-opt.rate * opt.maturity));
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    return out;
}

}  // namespace quantcuda::gpu
