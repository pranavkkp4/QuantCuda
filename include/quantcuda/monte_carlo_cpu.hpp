// QuantCUDA Risk Engine - CPU Monte Carlo engines (scalar + OpenMP).
//
// Both engines use counter-based Philox streams keyed by (seed, path_id), so
// per-path draws do not depend on thread scheduling. OpenMP accumulates into
// per-thread partial sums combined in a fixed order: results are bitwise
// reproducible for a fixed thread count and agree across thread counts to
// within floating-point reduction noise (~1e-16 relative).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef QUANTCUDA_HAS_OPENMP
#include <omp.h>
#endif

#include "rng_cpu.hpp"
#include "types.hpp"

namespace quantcuda {

struct McResult {
    double price       = 0.0;   // discounted mean payoff
    double std_error   = 0.0;   // MC standard error of the price
    double mean_payoff = 0.0;   // undiscounted mean terminal payoff
    double variance    = 0.0;   // variance of discounted payoff
    std::size_t paths  = 0;

    double confidence_half_width(double z = 1.959963985) const { return z * std_error; }
};

// Barrier path-evaluation strategy, mirroring the GPU divergence experiment.
enum class BarrierCheckMode { EarlyExit, BooleanMask };

namespace detail {

struct Accumulator {
    double sum = 0.0;
    double sq_sum = 0.0;
    double mean = 0.0;
    double m2 = 0.0;
    std::size_t count = 0;

    void add(double x) {
        sum += x;
        sq_sum += x * x;
        ++count;
        const double delta = x - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (x - mean);
    }

    void merge(const Accumulator& other) {
        if (other.count == 0) return;
        if (count == 0) {
            *this = other;
            return;
        }
        const std::size_t combined = count + other.count;
        const double delta = other.mean - mean;
        m2 += other.m2 + delta * delta * static_cast<double>(count) *
                             static_cast<double>(other.count) /
                             static_cast<double>(combined);
        mean += delta * static_cast<double>(other.count) /
                static_cast<double>(combined);
        count = combined;
        sum += other.sum;
        sq_sum += other.sq_sum;
    }
};

inline McResult finalize(const Accumulator& acc, std::size_t n, double discount) {
    if (n == 0) return McResult{};
    if (acc.count != n)
        throw std::logic_error("Monte Carlo accumulator count mismatch");
    const double mean = acc.mean;
    // Unbiased sample variance. Welford moments avoid catastrophic cancellation
    // for nearly deterministic, deep-ITM portfolios.
    double var = n > 1 ? acc.m2 / static_cast<double>(n - 1) : 0.0;
    if (var < 0.0) var = 0.0;
    McResult r;
    r.mean_payoff = mean;
    r.price       = discount * mean;
    r.variance    = var * discount * discount;
    r.std_error   = std::sqrt(r.variance / static_cast<double>(n));
    r.paths       = n;
    return r;
}

inline void validate_mc_inputs(double S0, double K, double r, double q,
                               double sigma, double T, std::size_t paths,
                               const char* context) {
    if (!std::isfinite(S0) || !std::isfinite(K) || !std::isfinite(r) ||
        !std::isfinite(q) || !std::isfinite(sigma) || !std::isfinite(T))
        throw std::invalid_argument(std::string(context) + ": inputs must be finite");
    if (S0 < 0.0) throw std::invalid_argument(std::string(context) + ": spot must be >= 0");
    if (K < 0.0) throw std::invalid_argument(std::string(context) + ": strike must be >= 0");
    if (sigma < 0.0)
        throw std::invalid_argument(std::string(context) + ": volatility must be >= 0");
    if (T < 0.0)
        throw std::invalid_argument(std::string(context) + ": maturity must be >= 0");
    if (paths == 0)
        throw std::invalid_argument(std::string(context) + ": paths must be >= 1");
}

inline McResult deterministic_result(double payoff, std::size_t paths,
                                     double discount = 1.0) {
    McResult out;
    out.price = discount * payoff;
    out.mean_payoff = payoff;
    out.paths = paths;
    return out;
}

// GBM drift/diffusion coefficients for one step of length dt.
inline double gbm_drift(double r, double q, double sigma, double dt) {
    return (r - q - 0.5 * sigma * sigma) * dt;
}
inline double gbm_diffusion(double sigma, double dt) { return sigma * std::sqrt(dt); }

// Discounted payoff of one European path with terminal draw z.
inline double european_path(OptionType t, double S0, double K, double drift,
                            double diffusion, double z) {
    double st = S0 * std::exp(drift + diffusion * z);
    return european_payoff(t, st, K);
}

// Undiscounted payoff of one Asian path (arithmetic average over `steps` fixings).
inline double asian_path(OptionType t, double S0, double K, double drift,
                         double diffusion, int steps, Philox4x32_10& rng) {
    double s = S0;
    double running_sum = 0.0;  // running average only - full path never stored
    for (int j = 0; j < steps; ++j) {
        s *= std::exp(drift + diffusion * rng.next_normal());
        running_sum += s;
    }
    double avg = running_sum / static_cast<double>(steps);
    return is_call(t) ? (avg > K ? avg - K : 0.0) : (K > avg ? K - avg : 0.0);
}

// Undiscounted payoff of one barrier path, early-exit strategy.
inline double barrier_path_early_exit(OptionType t, double S0, double K, double barrier,
                                      double drift, double diffusion, int steps,
                                      Philox4x32_10& rng) {
    bool up = (t == OptionType::BarrierUpOutCall || t == OptionType::BarrierUpOutPut);
    double s = S0;
    for (int j = 0; j < steps; ++j) {
        s *= std::exp(drift + diffusion * rng.next_normal());
        if (up ? (s >= barrier) : (s <= barrier)) return 0.0;  // knocked out
    }
    return is_call(t) ? (s > K ? s - K : 0.0) : (K > s ? K - s : 0.0);
}

// Undiscounted payoff of one barrier path, boolean-mask strategy (no break).
inline double barrier_path_mask(OptionType t, double S0, double K, double barrier,
                                double drift, double diffusion, int steps,
                                Philox4x32_10& rng) {
    bool up = (t == OptionType::BarrierUpOutCall || t == OptionType::BarrierUpOutPut);
    double s = S0;
    bool alive = true;
    for (int j = 0; j < steps; ++j) {
        s *= std::exp(drift + diffusion * rng.next_normal());
        alive = alive && (up ? (s < barrier) : (s > barrier));
    }
    if (!alive) return 0.0;
    return is_call(t) ? (s > K ? s - K : 0.0) : (K > s ? K - s : 0.0);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Scalar engine (correctness reference). Double precision throughout.
// ---------------------------------------------------------------------------

class CpuMonteCarloPricer {
public:
    static McResult price_european(OptionType t, double S0, double K, double r,
                                   double q, double sigma, double T,
                                   std::size_t paths, std::uint64_t seed) {
        if (!is_european(t))
            throw std::invalid_argument("european: option type must be European");
        detail::validate_mc_inputs(S0, K, r, q, sigma, T, paths, "european");
        if (T == 0.0)
            return detail::deterministic_result(european_payoff(t, S0, K), paths);
        double drift = detail::gbm_drift(r, q, sigma, T);
        double diff  = detail::gbm_diffusion(sigma, T);
        double df    = std::exp(-r * T);
        detail::Accumulator acc;
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            acc.add(detail::european_path(t, S0, K, drift, diff, rng.next_normal()));
        }
        return detail::finalize(acc, paths, df);
    }

    static McResult price_asian(OptionType t, double S0, double K, double r,
                                double q, double sigma, double T, int steps,
                                std::size_t paths, std::uint64_t seed) {
        if (!is_asian(t))
            throw std::invalid_argument("asian: option type must be Asian");
        detail::validate_mc_inputs(S0, K, r, q, sigma, T, paths, "asian");
        if (steps <= 0) throw std::invalid_argument("asian: steps must be > 0");
        double dt    = T / steps;
        double drift = detail::gbm_drift(r, q, sigma, dt);
        double diff  = detail::gbm_diffusion(sigma, dt);
        double df    = std::exp(-r * T);
        detail::Accumulator acc;
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            acc.add(detail::asian_path(t, S0, K, drift, diff, steps, rng));
        }
        return detail::finalize(acc, paths, df);
    }

    static McResult price_barrier(OptionType t, double S0, double K, double barrier,
                                  double r, double q, double sigma, double T, int steps,
                                  std::size_t paths, std::uint64_t seed,
                                  BarrierCheckMode mode = BarrierCheckMode::EarlyExit) {
        if (!is_barrier(t))
            throw std::invalid_argument("barrier: option type must be a barrier option");
        detail::validate_mc_inputs(S0, K, r, q, sigma, T, paths, "barrier");
        if (steps <= 0) throw std::invalid_argument("barrier: steps must be > 0");
        if (!std::isfinite(barrier) || barrier <= 0.0)
            throw std::invalid_argument("barrier: barrier must be finite and > 0");
        // Barrier already breached at inception.
        bool up = (t == OptionType::BarrierUpOutCall || t == OptionType::BarrierUpOutPut);
        if ((up && S0 >= barrier) || (!up && S0 <= barrier)) {
            return detail::deterministic_result(0.0, paths, std::exp(-r * T));
        }
        double dt    = T / steps;
        double drift = detail::gbm_drift(r, q, sigma, dt);
        double diff  = detail::gbm_diffusion(sigma, dt);
        double df    = std::exp(-r * T);
        detail::Accumulator acc;
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            double p = (mode == BarrierCheckMode::EarlyExit)
                           ? detail::barrier_path_early_exit(t, S0, K, barrier, drift,
                                                             diff, steps, rng)
                           : detail::barrier_path_mask(t, S0, K, barrier, drift, diff,
                                                       steps, rng);
            acc.add(p);
        }
        return detail::finalize(acc, paths, df);
    }

    // Dispatch on option type.
    static McResult price(const OptionSpec& opt, double S0, double r, double q,
                          double sigma, const SimulationConfig& cfg) {
        int steps = opt.time_steps > 0 ? opt.time_steps : cfg.time_steps;
        if (is_european(opt.type))
            return price_european(opt.type, S0, opt.strike, r, q, sigma, opt.maturity,
                                  cfg.paths, cfg.seed);
        if (is_asian(opt.type))
            return price_asian(opt.type, S0, opt.strike, r, q, sigma, opt.maturity,
                               steps, cfg.paths, cfg.seed);
        return price_barrier(opt.type, S0, opt.strike, opt.barrier, r, q, sigma,
                             opt.maturity, steps, cfg.paths, cfg.seed);
    }
};

// ---------------------------------------------------------------------------
// OpenMP engine (multicore baseline). Falls back to scalar without OpenMP.
// ---------------------------------------------------------------------------

class OpenMpMonteCarloPricer {
public:
    static int max_threads() {
#ifdef QUANTCUDA_HAS_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    template <typename PathFn>
    static McResult run(std::size_t paths, double discount, std::uint64_t seed,
                        int requested_threads, PathFn&& path_payoff) {
        if (paths == 0) throw std::invalid_argument("openmp: paths must be >= 1");
#ifdef QUANTCUDA_HAS_OPENMP
        int nthreads = requested_threads > 0 ? requested_threads : omp_get_max_threads();
        if (nthreads <= 0) nthreads = 1;
        if (paths < static_cast<std::size_t>(nthreads))
            nthreads = static_cast<int>(paths);
        std::vector<detail::Accumulator> partials(static_cast<std::size_t>(nthreads));
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            auto& local = partials[static_cast<std::size_t>(tid)];
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(paths); ++i) {
                auto rng = make_path_rng(seed, static_cast<std::uint64_t>(i));
                local.add(path_payoff(i, rng));
            }
        }
        detail::Accumulator total;
        for (const auto& p : partials) total.merge(p);
        return detail::finalize(total, paths, discount);
#else
        (void)requested_threads;
        detail::Accumulator acc;
        for (std::size_t i = 0; i < paths; ++i) {
            auto rng = make_path_rng(seed, i);
            acc.add(path_payoff(i, rng));
        }
        return detail::finalize(acc, paths, discount);
#endif
    }

    static McResult price(const OptionSpec& opt, double S0, double r, double q,
                          double sigma, const SimulationConfig& cfg,
                          BarrierCheckMode barrier_mode = BarrierCheckMode::EarlyExit) {
        double T  = opt.maturity;
        detail::validate_mc_inputs(S0, opt.strike, r, q, sigma, T, cfg.paths,
                                   "openmp price");
        double df = std::exp(-r * T);
        if (is_european(opt.type)) {
            if (T == 0.0)
                return detail::deterministic_result(
                    european_payoff(opt.type, S0, opt.strike), cfg.paths);
            double drift = detail::gbm_drift(r, q, sigma, T);
            double diff  = detail::gbm_diffusion(sigma, T);
            return run(cfg.paths, df, cfg.seed, cfg.openmp_threads,
                       [&](std::size_t, Philox4x32_10& rng) {
                           return detail::european_path(opt.type, S0, opt.strike, drift,
                                                        diff, rng.next_normal());
                       });
        }
        int steps = opt.time_steps > 0 ? opt.time_steps : cfg.time_steps;
        if (steps <= 0) throw std::invalid_argument("path option: steps must be > 0");
        double dt    = T / steps;
        double drift = detail::gbm_drift(r, q, sigma, dt);
        double diff  = detail::gbm_diffusion(sigma, dt);
        if (is_asian(opt.type)) {
            return run(cfg.paths, df, cfg.seed, cfg.openmp_threads,
                       [&](std::size_t, Philox4x32_10& rng) {
                           return detail::asian_path(opt.type, S0, opt.strike, drift,
                                                     diff, steps, rng);
                       });
        }
        if (!is_barrier(opt.type))
            throw std::invalid_argument("openmp price: unsupported option type");
        if (!std::isfinite(opt.barrier) || opt.barrier <= 0.0)
            throw std::invalid_argument("openmp price: barrier must be finite and > 0");
        // Barrier: knock-out at inception is trivial.
        bool up = (opt.type == OptionType::BarrierUpOutCall ||
                   opt.type == OptionType::BarrierUpOutPut);
        if ((up && S0 >= opt.barrier) || (!up && S0 <= opt.barrier)) {
            return detail::deterministic_result(0.0, cfg.paths, df);
        }
        return run(cfg.paths, df, cfg.seed, cfg.openmp_threads,
                   [&](std::size_t, Philox4x32_10& rng) {
                       return barrier_mode == BarrierCheckMode::EarlyExit
                                  ? detail::barrier_path_early_exit(
                                        opt.type, S0, opt.strike, opt.barrier, drift,
                                        diff, steps, rng)
                                  : detail::barrier_path_mask(
                                        opt.type, S0, opt.strike, opt.barrier, drift,
                                        diff, steps, rng);
                   });
    }
};

}  // namespace quantcuda
