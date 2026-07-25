// QuantCUDA Risk Engine - Kernel: portfolio scenario P&L + VaR/CVaR.
//
// One thread per market scenario: correlated shocks propagate each underlying
// over the risk horizon, every instrument is repriced (Black-Scholes full
// revaluation for Europeans; delta-gamma for path-dependent books), and the
// scenario loss is written once - coalesced. Losses are then sorted on device
// with Thrust and the tail beyond each VaR level is reduced for CVaR.
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {
namespace {

struct DeviceInstruments {
    int* asset;
    int* is_call;
    int* analytic;      // 1: Black-Scholes reprice; 0: delta-gamma approx
    double* strike;
    double* maturity;
    double* base_price;
    double* delta;
    double* gamma;
    double* quantity;
    int count;
};

__device__ double bs_price_device(int is_call, double S, double K, double r,
                                  double q, double sigma, double T) {
    if (T <= 0.0) return is_call ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
    if (S <= 0.0) return is_call ? 0.0 : K * exp(-r * T);
    if (K <= 0.0) return is_call ? S * exp(-q * T) : 0.0;
    if (sigma <= 0.0) {
        double fwd = S * exp((r - q) * T);
        return exp(-r * T) * (is_call ? fmax(fwd - K, 0.0) : fmax(K - fwd, 0.0));
    }
    double sqT = sigma * sqrt(T);
    double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sqT;
    double d2 = d1 - sqT;
    if (is_call)
        return S * exp(-q * T) * dev::norm_cdf_d(d1) -
               K * exp(-r * T) * dev::norm_cdf_d(d2);
    return K * exp(-r * T) * dev::norm_cdf_d(-d2) -
           S * exp(-q * T) * dev::norm_cdf_d(-d1);
}

__global__ void portfolio_pnl_kernel(const double* __restrict__ shocks,  // scenarios x d
                                     std::size_t scenarios, int d,
                                     const double* __restrict__ spots,
                                     const double* __restrict__ vols,
                                     const double* __restrict__ dividends,
                                     double rate, double horizon,
                                     DeviceInstruments instr,
                                     double* __restrict__ losses) {
    std::size_t stride = dev::grid_stride();
    double sqrt_h = sqrt(horizon);
    for (std::size_t s = dev::global_thread_id(); s < scenarios; s += stride) {
        double v = 0.0;
        for (int n = 0; n < instr.count; ++n) {
            int a = instr.asset[n];
            double spot0 = spots[a];
            double z = shocks[s * static_cast<std::size_t>(d) +
                              static_cast<std::size_t>(a)];
            double sigma = vols[a];
            double q = dividends[a];
            double drift = (rate - q - 0.5 * sigma * sigma) * horizon;
            double s_new = spot0 * exp(drift + sigma * sqrt_h * z);
            double price;
            if (instr.analytic[n]) {
                double remaining = instr.maturity[n] - horizon;
                price = bs_price_device(instr.is_call[n], s_new, instr.strike[n],
                                        rate, q, sigma, remaining);
            } else {
                double dS = s_new - spot0;
                price = instr.base_price[n] + instr.delta[n] * dS +
                        0.5 * instr.gamma[n] * dS * dS;
                if (price < 0.0) price = 0.0;
            }
            v += instr.quantity[n] * price;
        }
        losses[s] = v;  // scenario VALUE here; host converts to loss = V0 - V
    }
}

}  // namespace

std::vector<double> portfolio_losses_gpu(
    const std::vector<double>& shocks, std::size_t scenarios, std::size_t d,
    const std::vector<double>& spots, const std::vector<double>& vols,
    const std::vector<double>& dividends, double rate, double horizon,
    const std::vector<int>& instr_asset, const std::vector<int>& instr_is_call,
    const std::vector<int>& instr_analytic, const std::vector<double>& instr_strike,
    const std::vector<double>& instr_maturity, const std::vector<double>& instr_base,
    const std::vector<double>& instr_delta, const std::vector<double>& instr_gamma,
    const std::vector<double>& instr_quantity, double* kernel_ms_out) {
    if (scenarios == 0) {
        if (kernel_ms_out) *kernel_ms_out = 0.0;
        return {};
    }
    DeviceBuffer<double> d_shocks(shocks.size());
    d_shocks.copy_from(shocks.data());
    DeviceBuffer<double> d_spots(spots.size()), d_vols(vols.size()),
        d_divs(dividends.size());
    d_spots.copy_from(spots.data());
    d_vols.copy_from(vols.data());
    d_divs.copy_from(dividends.data());

    const int n = static_cast<int>(instr_asset.size());
    DeviceBuffer<int> i_asset(instr_asset.size()), i_call(instr_is_call.size()),
        i_an(instr_analytic.size());
    DeviceBuffer<double> i_strike(instr_strike.size()), i_mat(instr_maturity.size()),
        i_base(instr_base.size()), i_delta(instr_delta.size()),
        i_gamma(instr_gamma.size()), i_qty(instr_quantity.size());
    i_asset.copy_from(instr_asset.data());
    i_call.copy_from(instr_is_call.data());
    i_an.copy_from(instr_analytic.data());
    i_strike.copy_from(instr_strike.data());
    i_mat.copy_from(instr_maturity.data());
    i_base.copy_from(instr_base.data());
    i_delta.copy_from(instr_delta.data());
    i_gamma.copy_from(instr_gamma.data());
    i_qty.copy_from(instr_quantity.data());

    DeviceInstruments h_instr{i_asset.ptr, i_call.ptr,  i_an.ptr,  i_strike.ptr,
                              i_mat.ptr,   i_base.ptr,  i_delta.ptr, i_gamma.ptr,
                              i_qty.ptr,   n};
    DeviceBuffer<double> d_values(scenarios);

    int block = 256;
    int grid = dev::grid_for(scenarios, block);
    EventTimer ev;
    ev.begin();
    portfolio_pnl_kernel<<<grid, block>>>(
        d_shocks.ptr, scenarios, static_cast<int>(d), d_spots.ptr, d_vols.ptr,
        d_divs.ptr, rate, horizon, h_instr, d_values.ptr);
    QC_CUDA_CHECK(cudaGetLastError());
    const double kernel_ms = ev.end();
    if (kernel_ms_out) *kernel_ms_out = kernel_ms;

    std::vector<double> values(scenarios);
    d_values.copy_to(values.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    return values;
}

std::vector<VarEstimate> var_cvar_gpu(std::vector<double> losses,
                                      const std::vector<double>& levels) {
    std::size_t n = losses.size();
    std::vector<VarEstimate> out;
    if (n == 0) return out;
    DeviceBuffer<double> d(n);
    d.copy_from(losses.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    thrust::device_ptr<double> begin(d.ptr);
    thrust::sort(begin, begin + n);  // ascending losses
    for (double alpha : levels) {
        std::size_t idx = static_cast<std::size_t>(
            std::ceil(alpha * static_cast<double>(n)));
        if (idx == 0) idx = 1;
        if (idx > n) idx = n;
        double var_value = 0.0;
        QC_CUDA_CHECK(cudaMemcpy(&var_value, d.ptr + (idx - 1), sizeof(double),
                                 cudaMemcpyDeviceToHost));
        // Quantile-integral expected shortfall. Alpha may cut through one
        // empirical observation's probability interval, so include only the
        // fractional boundary mass and every strictly higher observation.
        const long double scaled_alpha =
            static_cast<long double>(alpha) * static_cast<long double>(n);
        const std::size_t first_tail = static_cast<std::size_t>(
            std::floor(scaled_alpha));
        const long double boundary_mass =
            static_cast<long double>(first_tail + 1) - scaled_alpha;
        double boundary_value = 0.0;
        QC_CUDA_CHECK(cudaMemcpy(&boundary_value, d.ptr + first_tail,
                                 sizeof(double), cudaMemcpyDeviceToHost));
        const double strict_tail_sum = thrust::reduce(
            begin + static_cast<std::ptrdiff_t>(first_tail + 1), begin + n, 0.0);
        const long double tail_sum =
            boundary_mass * static_cast<long double>(boundary_value) +
            static_cast<long double>(strict_tail_sum);
        const long double tail_mass =
            static_cast<long double>(n) - scaled_alpha;
        VarEstimate e;
        e.confidence = alpha;
        e.var = var_value;
        e.cvar = static_cast<double>(tail_sum / tail_mass);
        out.push_back(e);
    }
    return out;
}

}  // namespace quantcuda::gpu
