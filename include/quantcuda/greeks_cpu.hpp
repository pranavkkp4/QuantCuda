// QuantCUDA Risk Engine - Monte Carlo Greeks on CPU.
//
//  * Central finite differences with COMMON RANDOM NUMBERS: one set of draws
//    prices V(S-h), V(S), V(S+h) (and the vega pair) so differencing cancels
//    most sampling noise. A weak implementation runs three unrelated
//    simulations; the CRN version has materially lower estimator variance
//    (verified by test_greeks).
//  * Pathwise delta for European options under GBM:
//        dV/dS0 = E[ e^{-rT} * 1{S_T > K} * S_T / S0 ]   (call)
//    compared against Black-Scholes delta and FD delta in tests and reports.
#pragma once

#include <cmath>

#include "black_scholes.hpp"
#include "monte_carlo_cpu.hpp"

namespace quantcuda {

struct McGreeksResult {
    Greeks greeks;
    double price = 0.0;
    double price_std_error = 0.0;
    // diagnostic: standard errors of the FD estimators themselves
    double delta_std_error = 0.0;
    double gamma_std_error = 0.0;
    std::size_t paths = 0;
};

class CpuGreeksEngine {
public:
    // Central FD delta/gamma + FD vega for a European option, common random
    // numbers. h defaults to 1% of spot, h_sigma to 1 vol point.
    static McGreeksResult european_fd(OptionType t, double S0, double K, double r,
                                      double q, double sigma, double T,
                                      std::size_t paths, std::uint64_t seed,
                                      double h = 0.0, double h_sigma = 0.0) {
        if (!is_european(t)) throw std::invalid_argument("european_fd: European options only");
        detail::validate_mc_inputs(S0, K, r, q, sigma, T, paths, "european_fd");
        if (S0 <= 0.0) throw std::invalid_argument("european_fd: S0 must be > 0");
        if (T <= 0.0) throw std::invalid_argument("european_fd: T must be > 0");
        if (h <= 0.0) h = 0.01 * S0;
        if (h_sigma <= 0.0) h_sigma = 0.01;
        if (!std::isfinite(h) || h >= S0)
            throw std::invalid_argument("european_fd: spot bump must be finite and < S0");
        if (!std::isfinite(h_sigma))
            throw std::invalid_argument("european_fd: volatility bump must be finite");

        double drift = detail::gbm_drift(r, q, sigma, T);
        double diff  = detail::gbm_diffusion(sigma, T);
        double df    = std::exp(-r * T);

        const bool central_vega = sigma >= h_sigma;
        double sig_dn = central_vega ? sigma - h_sigma : sigma;
        double drift_vu = drift, drift_vd = drift;  // drift depends on sigma^2
        if (central_vega) {
            drift_vu = detail::gbm_drift(r, q, sigma + h_sigma, T);
            drift_vd = detail::gbm_drift(r, q, sig_dn, T);
        } else {
            drift_vu = detail::gbm_drift(r, q, sigma + h_sigma, T);
            drift_vd = drift;
        }
        double diff_vu = detail::gbm_diffusion(sigma + h_sigma, T);
        double diff_vd = detail::gbm_diffusion(sig_dn, T);

        detail::Accumulator acc_dn, acc_base, acc_up, acc_vdn, acc_vup;
        detail::Accumulator acc_delta, acc_gamma;  // per-path FD contributions
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            double z = rng.next_normal();  // ONE draw prices all five points
            double p_dn  = detail::european_path(t, S0 - h, K, drift, diff, z);
            double p_mid = detail::european_path(t, S0,     K, drift, diff, z);
            double p_up  = detail::european_path(t, S0 + h, K, drift, diff, z);
            acc_dn.add(p_dn); acc_base.add(p_mid); acc_up.add(p_up);
            acc_delta.add((p_up - p_dn) / (2.0 * h));
            acc_gamma.add((p_up - 2.0 * p_mid + p_dn) / (h * h));
            double pv_dn = detail::european_path(t, S0, K, drift_vd, diff_vd, z);
            double pv_up = detail::european_path(t, S0, K, drift_vu, diff_vu, z);
            acc_vdn.add(pv_dn); acc_vup.add(pv_up);
        }

        McResult r_dn  = detail::finalize(acc_dn, paths, df);
        McResult r_mid = detail::finalize(acc_base, paths, df);
        McResult r_up  = detail::finalize(acc_up, paths, df);
        McResult r_vdn = detail::finalize(acc_vdn, paths, df);
        McResult r_vup = detail::finalize(acc_vup, paths, df);
        McResult r_d   = detail::finalize(acc_delta, paths, df);
        McResult r_g   = detail::finalize(acc_gamma, paths, df);

        McGreeksResult out;
        out.price = r_mid.price;
        out.price_std_error = r_mid.std_error;
        out.greeks.delta = (r_up.price - r_dn.price) / (2.0 * h);
        out.greeks.gamma = (r_up.price - 2.0 * r_mid.price + r_dn.price) / (h * h);
        out.greeks.vega  = (r_vup.price - r_vdn.price) /
                           (central_vega ? 2.0 * h_sigma : h_sigma);
        out.delta_std_error = r_d.std_error;
        out.gamma_std_error = r_g.std_error;
        out.paths = paths;
        return out;
    }

    // Pathwise delta (European, GBM). Discounted mean of 1{ITM} * S_T / S0.
    static McResult european_delta_pathwise(OptionType t, double S0, double K,
                                            double r, double q, double sigma,
                                            double T, std::size_t paths,
                                            std::uint64_t seed) {
        if (!is_european(t)) throw std::invalid_argument("pathwise: European only");
        detail::validate_mc_inputs(S0, K, r, q, sigma, T, paths, "pathwise");
        if (S0 <= 0.0) throw std::invalid_argument("pathwise: S0 must be > 0");
        if (T <= 0.0) throw std::invalid_argument("pathwise: T must be > 0");
        double drift = detail::gbm_drift(r, q, sigma, T);
        double diff  = detail::gbm_diffusion(sigma, T);
        double df    = std::exp(-r * T);
        detail::Accumulator acc;
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            double st = S0 * std::exp(drift + diff * rng.next_normal());
            bool itm = is_call(t) ? (st > K) : (st < K);
            // For puts the pathwise derivative carries a minus sign.
            double contrib = itm ? (st / S0) * (is_call(t) ? 1.0 : -1.0) : 0.0;
            acc.add(contrib);
        }
        return detail::finalize(acc, paths, df);
    }

    // Likelihood-ratio-free convenience: analytic for European, FD otherwise.
    static Greeks analytic_or_fd(const OptionSpec& opt, double S0, double r, double q,
                                 double sigma, const SimulationConfig& cfg) {
        if (is_european(opt.type)) {
            return BlackScholesPricer::greeks(opt.type, S0, opt.strike, r, q, sigma,
                                              opt.maturity);
        }
        // FD on the MC price with common random numbers (bump-revalue).
        if (!std::isfinite(S0) || S0 <= 0.0)
            throw std::invalid_argument("analytic_or_fd: S0 must be finite and > 0");
        double h = 0.01 * S0;
        double S_dn = S0 - h, S_up = S0 + h;
        auto price_at = [&](double spot, double vol) {
            return cfg.backend == Backend::OpenMP
                       ? OpenMpMonteCarloPricer::price(opt, spot, r, q, vol, cfg)
                       : CpuMonteCarloPricer::price(opt, spot, r, q, vol, cfg);
        };
        McResult p_dn = price_at(S_dn, sigma);
        McResult p_md = price_at(S0, sigma);
        McResult p_up = price_at(S_up, sigma);
        Greeks g;
        g.delta = (p_up.price - p_dn.price) / (2.0 * h);
        g.gamma = (p_up.price - 2.0 * p_md.price + p_dn.price) / (h * h);
        double hs = 0.01;
        const bool central_vega = sigma >= hs;
        McResult v_dn = price_at(S0, central_vega ? sigma - hs : sigma);
        McResult v_up = price_at(S0, sigma + hs);
        g.vega = (v_up.price - v_dn.price) / (central_vega ? 2.0 * hs : hs);
        return g;
    }
};

}  // namespace quantcuda
