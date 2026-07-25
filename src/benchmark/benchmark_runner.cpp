// QuantCUDA Risk Engine - benchmark runner.
//
// Methodology (per the specification):
//   * warm-up runs before measurement,
//   * median / min / p95 of repeated measured runs,
//   * identical seeds and parameters across backends,
//   * environment (compiler, flags, CPU, GPU, OS) recorded in every report,
//   * kernel-only vs end-to-end timing kept separate on the GPU rows.
#include "quantcuda/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <stdexcept>

#include "quantcuda/black_scholes.hpp"
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/json_config.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "quantcuda/risk_cpu.hpp"
#include "quantcuda/timer.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace quantcuda {
namespace {

// ---------------------------------------------------------------------------
// Result rows
// ---------------------------------------------------------------------------
struct Row {
    std::string suite;
    std::string backend;
    std::string variant;
    std::uint64_t paths = 0;
    int steps = 1;
    int instruments = 1;
    std::size_t scenarios = 0;
    int repeats = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    // Secondary timing: version-study stores E2E here; layout-study stores
    // kernel time. CSV serialization expands this into explicit columns.
    double secondary_ms = 0.0;
    double price = 0.0;
    double std_error = 0.0;
    double throughput = 0.0;   // paths/sec (or scenarios/sec)
    std::string note;
    double abs_error = 0.0;    // explicit correctness metric (for convergence)
};

struct Stats {
    double min, median, p95;
};

constexpr std::uint64_t kMaxBenchmarkPaths = 100'000'000;
constexpr std::uint64_t kMaxBenchmarkScenarios = 10'000'000;
constexpr std::uint64_t kMaxBenchmarkWork = 5'000'000'000;

void require_work_within_limit(std::uint64_t paths, std::uint64_t steps,
                               std::uint64_t instruments) {
    if (paths == 0 || steps == 0 || instruments == 0)
        throw std::runtime_error("benchmark workload dimensions must be positive");
    if (paths > kMaxBenchmarkWork / steps ||
        paths * steps > kMaxBenchmarkWork / instruments)
        throw std::runtime_error(
            "benchmark paths x steps x instruments exceeds safety limit 5000000000");
}

Stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.min = v.front();
    s.median = v[v.size() / 2];
    s.p95 = v[static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(v.size()))) - 1];
    return s;
}

// Time `fn` (returns a value for verification) with warmup + repeats.
template <typename Fn>
Stats time_repeated(int warmup, int repeats, Fn&& fn) {
    for (int i = 0; i < warmup; ++i) (void)fn();
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        WallTimer t;
        (void)fn();
        times.push_back(t.elapsed_ms());
    }
    return summarize(std::move(times));
}

// ---------------------------------------------------------------------------
// Canonical instruments for the suites
// ---------------------------------------------------------------------------
OptionSpec euro_call(double S = 100.0, double K = 105.0) {
    OptionSpec o;
    o.id = "BENCH_EU_CALL";
    o.type = OptionType::EuropeanCall;
    o.underlying = "BENCH";
    o.strike = K;
    o.maturity = 1.0;
    o.quantity = 1.0;
    (void)S;
    return o;
}

MarketData bench_market() {
    MarketData m;
    m.risk_free_rate = 0.0425;
    m.assets.push_back({"BENCH", 100.0, 0.20, 0.01});
    return m;
}

SimulationConfig sim_cfg(std::uint64_t seed, std::size_t paths, int steps,
                         Backend b) {
    SimulationConfig c;
    c.seed = seed;
    c.paths = paths;
    c.time_steps = steps;
    c.backend = b;
    c.precision = Precision::Double;
    return c;
}

McResult price_dispatch(const OptionSpec& o, const MarketData& m,
                        const SimulationConfig& c) {
    const AssetData& ad = m.assets[0];
    switch (c.backend) {
        case Backend::Scalar:
            return CpuMonteCarloPricer::price(o, ad.spot, m.risk_free_rate,
                                              ad.dividend_yield, ad.volatility, c);
        case Backend::OpenMP:
            return OpenMpMonteCarloPricer::price(o, ad.spot, m.risk_free_rate,
                                                 ad.dividend_yield, ad.volatility, c);
        case Backend::Cuda:
            return gpu::price_option(o, ad.spot, m.risk_free_rate,
                                     ad.dividend_yield, ad.volatility, c);
    }
    throw std::runtime_error("backend");
}

bool backend_runnable(Backend b) {
    if (b == Backend::OpenMP) {
#ifdef QUANTCUDA_HAS_OPENMP
        return true;
#else
        return false;
#endif
    }
    if (b != Backend::Cuda) return true;
    return gpu::available();
}

// ---------------------------------------------------------------------------
// Suite A: European throughput
// ---------------------------------------------------------------------------
std::vector<Row> suite_european(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 10'000'000;
    require_work_within_limit(paths, 1, 1);
    OptionSpec opt = euro_call();
    MarketData m = bench_market();
    for (Backend b : o.backends) {
        if (!backend_runnable(b)) {
            rows.push_back({"european-throughput", to_string(b), "-", paths, 1, 1, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, "skipped: backend unavailable"});
            continue;
        }
        SimulationConfig c = sim_cfg(o.seed, paths, 1, b);
        McResult last;
        Stats s = time_repeated(o.warmup, o.repeats,
                                [&] { last = price_dispatch(opt, m, c); return 0; });
        Row r;
        r.suite = "european-throughput";
        r.backend = to_string(b);
        r.variant = b == Backend::Cuda ? to_string(c.gpu_version) : "-";
        r.paths = paths;
        r.repeats = o.repeats;
        r.min_ms = s.min; r.median_ms = s.median; r.p95_ms = s.p95;
        r.price = last.price;
        r.std_error = last.std_error;
        r.throughput = static_cast<double>(paths) / (s.median / 1000.0);
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Suite B: path-dependent workload (Asian, configurable steps)
// ---------------------------------------------------------------------------
std::vector<Row> suite_path_dependent(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    int instruments = o.overrides.instruments ? o.overrides.instruments : 64;
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 100'000;
    int steps = o.overrides.steps ? o.overrides.steps : 252;
    require_work_within_limit(paths, static_cast<std::uint64_t>(steps),
                              static_cast<std::uint64_t>(instruments));
    MarketData m = bench_market();
    OptionSpec asian;
    asian.id = "BENCH_ASIAN";
    asian.type = OptionType::AsianCall;
    asian.underlying = "BENCH";
    asian.strike = 100.0;
    asian.maturity = 1.0;

    for (Backend b : o.backends) {
        if (!backend_runnable(b)) {
            rows.push_back({"path-dependent", to_string(b), "-", paths, steps,
                            instruments, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            "skipped: backend unavailable"});
            continue;
        }
        SimulationConfig c = sim_cfg(o.seed, paths, steps, b);
        McResult last;
        Stats s = time_repeated(o.warmup, o.repeats, [&] {
            double total = 0.0;
            for (int i = 0; i < instruments; ++i) {
                asian.strike = 90.0 + static_cast<double>(i % 21);  // vary strikes
                last = price_dispatch(asian, m, c);
                total += last.price;
            }
            return total;
        });
        Row r;
        r.suite = "path-dependent";
        r.backend = to_string(b);
        r.paths = paths;
        r.steps = steps;
        r.instruments = instruments;
        r.repeats = o.repeats;
        r.min_ms = s.min; r.median_ms = s.median; r.p95_ms = s.p95;
        r.price = last.price;
        r.std_error = last.std_error;
        r.throughput = static_cast<double>(instruments) *
                       static_cast<double>(paths) / (s.median / 1000.0);
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Suite C: barrier divergence (near/medium/far x early-exit/mask)
// ---------------------------------------------------------------------------
std::vector<Row> suite_divergence(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 200'000;
    int steps = o.overrides.steps ? o.overrides.steps : 252;
    require_work_within_limit(paths, static_cast<std::uint64_t>(steps), 1);
    MarketData m = bench_market();
    const struct { const char* name; double barrier; } cases[] = {
        {"near", 110.0}, {"medium", 130.0}, {"far", 180.0}};
    for (Backend b : o.backends) {
        for (const auto& cse : cases) {
            for (int mode = 0; mode <= (b == Backend::Cuda ? 2 : 1); ++mode) {
                if (!backend_runnable(b)) {
                    rows.push_back({"divergence", to_string(b), cse.name, paths,
                                    steps, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    "skipped: backend unavailable"});
                    break;
                }
                OptionSpec bar;
                bar.id = "BENCH_BARRIER";
                bar.type = OptionType::BarrierUpOutCall;
                bar.underlying = "BENCH";
                bar.strike = 105.0;
                bar.barrier = cse.barrier;
                bar.maturity = 1.0;
                SimulationConfig c = sim_cfg(o.seed, paths, steps, b);
                McResult last;
                const char* mode_name =
                    mode == 0 ? "early-exit" : (mode == 1 ? "bool-mask" : "compacted");
                Stats s = time_repeated(o.warmup, o.repeats, [&] {
                    if (b == Backend::Cuda)
                        last = gpu::price_barrier(bar, 100.0, m.risk_free_rate, 0.01,
                                                  0.20, c, mode);
                    else {
                        c.backend = b;
                        // CPU engines expose the same two strategies for the study.
                        last = (b == Backend::Scalar)
                                   ? CpuMonteCarloPricer::price_barrier(
                                         bar.type, 100.0, bar.strike, bar.barrier,
                                         m.risk_free_rate, 0.01, 0.20, 1.0, steps,
                                         paths, o.seed,
                                         mode == 0 ? BarrierCheckMode::EarlyExit
                                                   : BarrierCheckMode::BooleanMask)
                                   : OpenMpMonteCarloPricer::price(
                                         bar, 100.0, m.risk_free_rate, 0.01, 0.20, c,
                                         mode == 0 ? BarrierCheckMode::EarlyExit
                                                   : BarrierCheckMode::BooleanMask);
                    }
                    return 0;
                });
                Row r;
                r.suite = "divergence";
                r.backend = to_string(b);
                r.variant = std::string(cse.name) + "/" + mode_name;
                r.paths = paths;
                r.steps = steps;
                r.repeats = o.repeats;
                r.min_ms = s.min; r.median_ms = s.median; r.p95_ms = s.p95;
                r.price = last.price;
                r.std_error = last.std_error;
                r.throughput = static_cast<double>(paths) / (s.median / 1000.0);
                rows.push_back(r);
            }
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Suite D: portfolio risk
// ---------------------------------------------------------------------------
std::vector<Row> suite_portfolio_risk(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    EngineInput in;
    if (!o.config_file.empty()) {
        in = load_engine_input_file(o.config_file);
    } else {
        // Default: 16 assets, 64 European instruments, block correlation.
        MarketData& m = in.market;
        m.risk_free_rate = 0.0425;
        const int d = 16;
        m.correlation.assign(d, std::vector<double>(d, 0.0));
        for (int i = 0; i < d; ++i) {
            m.assets.push_back({"A" + std::to_string(i), 80.0 + 5.0 * i,
                                0.15 + 0.01 * (i % 5), 0.005 * (i % 3)});
            for (int j = 0; j < d; ++j)
                m.correlation[i][j] = (i == j) ? 1.0 : (0.15 + 0.05 * ((i + j) % 4));
        }
        const int instrument_count =
            o.overrides.instruments ? o.overrides.instruments : 64;
        for (int n = 0; n < instrument_count; ++n) {
            OptionSpec opt;
            opt.id = "P" + std::to_string(n);
            opt.type = n % 3 == 0 ? OptionType::EuropeanPut : OptionType::EuropeanCall;
            opt.underlying = "A" + std::to_string(n % d);
            opt.strike = 80.0 + 5.0 * (n % d) + 5.0;
            opt.maturity = 0.25 + 0.25 * (n % 4);
            opt.quantity = (n % 7 == 0) ? -20.0 : 50.0;
            in.portfolio.options.push_back(opt);
        }
    }
    std::size_t scenarios = o.overrides.scenarios
                                ? static_cast<std::size_t>(o.overrides.scenarios)
                                : in.risk.scenarios;
    RiskConfig rcfg = in.risk;
    rcfg.scenarios = scenarios;
    SimulationConfig price_cfg = in.simulation;
    price_cfg.paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                        : 20'000;
    price_cfg.seed = o.seed;

    for (Backend b : o.backends) {
        if (!backend_runnable(b)) {
            rows.push_back({"portfolio-risk", to_string(b), "-", price_cfg.paths, 1,
                            static_cast<int>(in.portfolio.options.size()), scenarios,
                            0, 0, 0, 0, 0, 0, 0, 0, "skipped: backend unavailable"});
            continue;
        }
        RiskResult last;
        price_cfg.backend = b;
        if (b != Backend::Cuda) price_cfg.precision = Precision::Double;
        Stats s = time_repeated(o.warmup, o.repeats, [&] {
            if (b == Backend::Cuda)
                last = gpu::portfolio_risk(in.portfolio, in.market, price_cfg, rcfg);
            else
                last = PortfolioRiskEngineCpu::run(in.portfolio, in.market, price_cfg,
                                                   rcfg, b);
            return 0;
        });
        Row r;
        r.suite = "portfolio-risk";
        r.backend = to_string(b);
        r.paths = price_cfg.paths;
        r.instruments = static_cast<int>(in.portfolio.options.size());
        r.scenarios = scenarios;
        r.repeats = o.repeats;
        r.min_ms = s.min; r.median_ms = s.median; r.p95_ms = s.p95;
        r.price = last.base_value;
        r.throughput = static_cast<double>(scenarios) / (s.median / 1000.0);
        r.note = "VaR95=" + std::to_string(last.estimates.empty()
                                               ? 0.0 : last.estimates[0].var);
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Suite E: instrument scaling
// ---------------------------------------------------------------------------
std::vector<Row> suite_scaling(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 100'000;
    require_work_within_limit(paths, 1, 1024);
    MarketData m = bench_market();
    OptionSpec opt = euro_call();
    for (Backend b : o.backends) {
        for (int instruments : {1, 8, 64, 256, 1024}) {
            if (!backend_runnable(b)) {
                rows.push_back({"scaling", to_string(b), "-", paths, 1, instruments,
                                0, 0, 0, 0, 0, 0, 0, 0, 0,
                                "skipped: backend unavailable"});
                continue;
            }
            SimulationConfig c = sim_cfg(o.seed, paths, 1, b);
            Stats s = time_repeated(o.warmup, o.repeats, [&] {
                double total = 0.0;
                for (int i = 0; i < instruments; ++i) {
                    opt.strike = 95.0 + static_cast<double>(i % 21);
                    total += price_dispatch(opt, m, c).price;
                }
                return total;
            });
            Row r;
            r.suite = "scaling";
            r.backend = to_string(b);
            r.paths = paths;
            r.instruments = instruments;
            r.repeats = o.repeats;
            r.min_ms = s.min; r.median_ms = s.median; r.p95_ms = s.p95;
            r.throughput = static_cast<double>(instruments) *
                           static_cast<double>(paths) / (s.median / 1000.0);
            rows.push_back(r);
        }
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Suite: Monte Carlo convergence (correctness-flavored benchmark)
// ---------------------------------------------------------------------------
std::vector<Row> suite_convergence(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    MarketData m = bench_market();
    OptionSpec opt = euro_call();
    double bs = BlackScholesPricer::call(100.0, 105.0, 0.0425, 0.01, 0.20, 1.0);
    Backend b = o.backends.empty() ? Backend::Scalar : o.backends[0];
    if (!backend_runnable(b)) b = Backend::Scalar;
    for (std::size_t paths : {10'000UL, 100'000UL, 1'000'000UL, 10'000'000UL}) {
        SimulationConfig c = sim_cfg(o.seed, paths, 1, b);
        McResult r = price_dispatch(opt, m, c);
        Row row;
        row.suite = "convergence";
        row.backend = to_string(b);
        row.paths = paths;
        row.price = r.price;
        row.std_error = r.std_error;
        row.abs_error = std::fabs(r.price - bs);
        row.note = "abs_error_vs_BS";
        rows.push_back(row);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// GPU-only studies
// ---------------------------------------------------------------------------
std::vector<Row> suite_version_study(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    if (!gpu::available()) {
        rows.push_back({"version-study", "cuda", "-", 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, "skipped: backend unavailable"});
        return rows;
    }
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 10'000'000;
    require_work_within_limit(paths, 1, 1);
    OptionSpec opt = euro_call();
    SimulationConfig c = sim_cfg(o.seed, paths, 1, Backend::Cuda);
    const int repeats = std::max(o.repeats, 1);
    for (int rep = 0; rep < o.warmup; ++rep)
        (void)gpu::european_version_study(opt, 100.0, 0.0425, 0.01, 0.20, c);

    std::vector<gpu::VersionRow> table;
    std::vector<std::vector<double>> kernel_samples, e2e_samples;
    for (int rep = 0; rep < repeats; ++rep) {
        auto current =
            gpu::european_version_study(opt, 100.0, 0.0425, 0.01, 0.20, c);
        if (rep == 0) {
            kernel_samples.resize(current.size());
            e2e_samples.resize(current.size());
        } else if (current.size() != kernel_samples.size()) {
            throw std::runtime_error("version-study row count changed between repeats");
        }
        for (std::size_t i = 0; i < current.size(); ++i) {
            kernel_samples[i].push_back(current[i].kernel_ms);
            e2e_samples[i].push_back(current[i].e2e_ms);
        }
        table = std::move(current);
    }
    for (std::size_t i = 0; i < table.size(); ++i) {
        const auto& v = table[i];
        Stats kernel = summarize(kernel_samples[i]);
        Stats e2e = summarize(e2e_samples[i]);
        Row r;
        r.suite = "version-study";
        r.backend = "cuda";
        r.variant = to_string(v.version);
        r.paths = paths;
        r.repeats = repeats;
        r.min_ms = kernel.min;
        r.median_ms = kernel.median;
        r.p95_ms = kernel.p95;
        r.secondary_ms = e2e.median;
        r.price = v.price;
        r.std_error = v.std_error;
        r.throughput = static_cast<double>(paths) / (kernel.median / 1000.0);
        r.note = v.note;
        rows.push_back(r);
    }

    // V3 comprises four preserved reduction implementations. Append their
    // rows to this suite instead of creating a tenth advertised suite.
    for (int rep = 0; rep < o.warmup; ++rep)
        (void)gpu::reduction_study(paths, o.seed, c.block_size);
    std::vector<gpu::ReductionRow> reductions;
    std::vector<std::vector<double>> reduction_samples;
    for (int rep = 0; rep < repeats; ++rep) {
        auto current = gpu::reduction_study(paths, o.seed, c.block_size);
        if (rep == 0)
            reduction_samples.resize(current.size());
        else if (current.size() != reduction_samples.size())
            throw std::runtime_error("reduction-study row count changed between repeats");
        for (std::size_t i = 0; i < current.size(); ++i)
            reduction_samples[i].push_back(current[i].kernel_ms);
        reductions = std::move(current);
    }
    const double reduction_ref = reductions.empty() ? 0.0 : reductions.back().result;
    for (std::size_t i = 0; i < reductions.size(); ++i) {
        Stats timing = summarize(reduction_samples[i]);
        Row r;
        r.suite = "version-study";
        r.backend = "cuda";
        r.variant = "reduction/" + reductions[i].name;
        r.paths = paths;
        r.repeats = repeats;
        r.min_ms = timing.min;
        r.median_ms = timing.median;
        r.p95_ms = timing.p95;
        r.throughput = static_cast<double>(paths) / (timing.median / 1000.0);
        r.abs_error = std::fabs(reductions[i].result - reduction_ref);
        r.note = "V3 reduction flavor; sum=" + std::to_string(reductions[i].result);
        rows.push_back(r);
    }
    return rows;
}

std::vector<Row> suite_gpu_divergence(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    if (!gpu::available()) {
        rows.push_back({"gpu-divergence", "cuda", "-", 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, "skipped: backend unavailable"});
        return rows;
    }
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 1'000'000;
    int steps = o.overrides.steps ? o.overrides.steps : 252;
    require_work_within_limit(paths, static_cast<std::uint64_t>(steps), 1);
    OptionSpec opt;
    opt.id = "BENCH_BARRIER";
    opt.type = OptionType::BarrierUpOutCall;
    opt.underlying = "BENCH";
    opt.strike = 105.0;
    opt.maturity = 1.0;
    SimulationConfig c = sim_cfg(o.seed, paths, steps, Backend::Cuda);
    const int repeats = std::max(o.repeats, 1);
    for (int rep = 0; rep < o.warmup; ++rep)
        (void)gpu::barrier_divergence_study(opt, 100.0, 0.0425, 0.01, 0.20, c);
    std::vector<gpu::DivergenceRow> table;
    std::vector<std::vector<double>> kernel_samples;
    for (int rep = 0; rep < repeats; ++rep) {
        auto current =
            gpu::barrier_divergence_study(opt, 100.0, 0.0425, 0.01, 0.20, c);
        if (rep == 0)
            kernel_samples.resize(current.size());
        else if (current.size() != kernel_samples.size())
            throw std::runtime_error(
                "gpu-divergence row count changed between repeats");
        for (std::size_t i = 0; i < current.size(); ++i)
            kernel_samples[i].push_back(current[i].kernel_ms);
        table = std::move(current);
    }
    const char* modes[] = {"early-exit", "bool-mask", "compacted"};
    for (std::size_t i = 0; i < table.size(); ++i) {
        const auto& v = table[i];
        Stats timing = summarize(kernel_samples[i]);
        Row r;
        r.suite = "gpu-divergence";
        r.backend = "cuda";
        r.variant = v.barrier_position + "/" + modes[v.mode];
        r.paths = paths;
        r.steps = steps;
        r.repeats = repeats;
        r.min_ms = timing.min;
        r.median_ms = timing.median;
        r.p95_ms = timing.p95;
        r.price = v.price;
        r.throughput = static_cast<double>(paths) / (timing.median / 1000.0);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "barrier=%.1f knockout=%.1f%%", v.barrier,
                      v.knockout_fraction * 100.0);
        r.note = buf;
        rows.push_back(r);
    }
    return rows;
}

std::vector<Row> suite_layout_study(const BenchmarkOptions& o) {
    std::vector<Row> rows;
    if (!gpu::available()) {
        rows.push_back({"layout-study", "cuda", "-", 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, "skipped: backend unavailable"});
        return rows;
    }
    int instruments = o.overrides.instruments ? o.overrides.instruments : 64;
    std::size_t paths = o.overrides.paths ? static_cast<std::size_t>(o.overrides.paths)
                                          : 100'000;
    require_work_within_limit(paths, 1,
                              static_cast<std::uint64_t>(instruments));
    auto table =
        gpu::layout_study(instruments, paths, o.seed, o.warmup, o.repeats);
    for (const auto& lr : table) {
        Row r;
        r.suite = "layout-study";
        r.backend = "cuda";
        r.variant = lr.aos ? "AoS" : "SoA";
        r.paths = paths;
        r.instruments = instruments;
        r.repeats = std::max(o.repeats, 1);
        r.median_ms = lr.e2e_ms;
        r.secondary_ms = lr.kernel_ms;
        r.throughput = static_cast<double>(instruments) *
                       static_cast<double>(paths) / (lr.e2e_ms / 1000.0);
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
struct TimingColumns {
    const char* scope = "none";
    double gpu_interval_ms = 0.0;
    double end_to_end_ms = 0.0;
};

TimingColumns explicit_timings(const Row& r) {
    if (r.suite == "version-study")
        return {"gpu_interval", r.median_ms, r.secondary_ms};
    if (r.suite == "gpu-divergence")
        return {"gpu_interval", r.median_ms, 0.0};
    if (r.suite == "layout-study")
        return {"end_to_end", r.secondary_ms, r.median_ms};
    if (r.suite == "convergence") return {};
    return {"end_to_end", 0.0, r.median_ms};
}

std::string csv_escape(std::string text) {
    std::size_t pos = 0;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        text.insert(pos, 1, '"');
        pos += 2;
    }
    return text;
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output file: " + path);
    f << std::setprecision(17);
    f << "suite,backend,variant,paths,steps,instruments,scenarios,repeats,"
         "min_ms,median_ms,p95_ms,timing_scope,gpu_interval_ms,end_to_end_ms,price,std_error,"
         "throughput_per_sec,abs_error,note,environment\n";
    std::string environment = environment_report();
    std::replace(environment.begin(), environment.end(), '\n', ';');
    for (const auto& r : rows) {
        const TimingColumns timing = explicit_timings(r);
        f << r.suite << ',' << r.backend << ',' << r.variant << ',' << r.paths << ','
          << r.steps << ',' << r.instruments << ',' << r.scenarios << ',' << r.repeats
          << ',' << r.min_ms << ',' << r.median_ms << ',' << r.p95_ms << ','
          << timing.scope << ',' << timing.gpu_interval_ms << ',' << timing.end_to_end_ms
          << ',' << r.price << ',' << r.std_error << ',' << r.throughput
          << ',' << r.abs_error << ",\"" << csv_escape(r.note) << "\",\""
          << csv_escape(environment) << "\"\n";
    }
}

void print_table(const std::vector<Row>& rows) {
    std::printf("%-20s %-8s %-22s %10s %6s %6s %12s %-10s %12s %12s %s\n", "suite",
                "backend", "variant", "paths", "steps", "instr", "median_ms",
                "scope", "gpu_ms", "e2e_ms", "note");
    std::printf("%.130s\n", std::string(130, '-').c_str());
    for (const auto& r : rows) {
        const TimingColumns timing = explicit_timings(r);
        std::printf("%-20s %-8s %-22s %10llu %6d %6d %12.2f %-10s %12.2f %12.2f %s\n",
                    r.suite.c_str(), r.backend.c_str(), r.variant.c_str(),
                    static_cast<unsigned long long>(r.paths), r.steps,
                    r.instruments, r.median_ms, timing.scope,
                    timing.gpu_interval_ms,
                    timing.end_to_end_ms,
                    r.note.c_str());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------
std::string environment_report() {
    std::string out = "QuantCUDA build/environment\n";
    out += "------------------------------------------------\n";
    char buf[256];
#if defined(__clang__)
    std::snprintf(buf, sizeof(buf), "Compiler:                 clang %d.%d.%d\n",
                  __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::snprintf(buf, sizeof(buf), "Compiler:                 GCC %d.%d.%d\n",
                  __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    std::snprintf(buf, sizeof(buf), "Compiler:                 MSVC %d.%02d\n",
                  _MSC_VER / 100, _MSC_VER % 100);
#else
    std::snprintf(buf, sizeof(buf), "Compiler:                 unknown\n");
#endif
    out += buf;
#if defined(_WIN32)
    out += "Operating system:          Windows\n";
#elif defined(__linux__)
    out += "Operating system:          Linux\n";
#elif defined(__APPLE__)
    out += "Operating system:          macOS\n";
#else
    out += "Operating system:          unknown\n";
#endif
#if defined(NDEBUG)
    out += "Optimization:             release (NDEBUG)\n";
#else
    out += "Optimization:             debug\n";
#endif
#ifdef QUANTCUDA_HAS_OPENMP
    std::snprintf(buf, sizeof(buf), "OpenMP:                   available (%d max threads)\n",
                  OpenMpMonteCarloPricer::max_threads());
#else
    std::snprintf(buf, sizeof(buf), "OpenMP:                   not available\n");
#endif
    out += buf;
    // CPU model from /proc/cpuinfo (Linux) or the standard Windows process
    // environment populated by the OS.
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line, model;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) model = line.substr(pos + 2);
            break;
        }
    }
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    if (model.empty()) {
        int registers[4]{};
        __cpuid(registers, static_cast<int>(0x80000000u));
        if (static_cast<unsigned>(registers[0]) >= 0x80000004u) {
            char brand[49]{};
            for (unsigned leaf = 0; leaf < 3; ++leaf) {
                __cpuid(registers, static_cast<int>(0x80000002u + leaf));
                std::memcpy(brand + leaf * 16, registers, 16);
            }
            model = brand;
            const std::size_t first = model.find_first_not_of(' ');
            if (first != std::string::npos) model.erase(0, first);
        }
    }
#elif defined(_WIN32)
    if (model.empty()) {
        if (const char* processor = std::getenv("PROCESSOR_IDENTIFIER"))
            model = processor;
    }
#endif
    out += "CPU:                      " + (model.empty() ? "unknown" : model) + "\n";
#ifdef QUANTCUDA_HAS_CUDA
    out += "CUDA toolkit in build:    yes\n";
#else
    out += "CUDA toolkit in build:    no (CPU-only binary)\n";
#endif
    out += "CUDA device:              " + gpu::device_name() + "\n";
    out += "------------------------------------------------\n";
    return out;
}

int run_benchmark(const BenchmarkOptions& opts) {
    if (opts.repeats < 1 || opts.repeats > 101)
        throw std::runtime_error("benchmark repeats must be in [1, 101]");
    if (opts.warmup < 0 || opts.warmup > 101)
        throw std::runtime_error("benchmark warmup must be in [0, 101]");
    if (opts.backends.empty())
        throw std::runtime_error("benchmark requires at least one backend");
    if (opts.overrides.paths > kMaxBenchmarkPaths)
        throw std::runtime_error("benchmark paths exceeds safety limit 100000000");
    if (opts.overrides.scenarios > kMaxBenchmarkScenarios)
        throw std::runtime_error("benchmark scenarios exceeds safety limit 10000000");
    if (opts.overrides.steps < 0 || opts.overrides.steps > 100'000)
        throw std::runtime_error("benchmark steps must be in [0, 100000]");
    if (opts.overrides.instruments < 0 || opts.overrides.instruments > 100'000)
        throw std::runtime_error("benchmark instruments must be in [0, 100000]");
    std::vector<Row> rows;
    const std::string& s = opts.suite;
    if (s == "european-throughput" || s == "all") {
        auto r = suite_european(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "path-dependent" || s == "all") {
        auto r = suite_path_dependent(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "divergence" || s == "all") {
        auto r = suite_divergence(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "portfolio-risk" || s == "all") {
        auto r = suite_portfolio_risk(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "scaling" || s == "all") {
        auto r = suite_scaling(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "convergence" || s == "all") {
        auto r = suite_convergence(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "version-study" || s == "all") {
        auto r = suite_version_study(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "gpu-divergence" || s == "all") {
        auto r = suite_gpu_divergence(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (s == "layout-study" || s == "all") {
        auto r = suite_layout_study(opts);
        rows.insert(rows.end(), r.begin(), r.end());
    }
    if (rows.empty())
        throw std::runtime_error(
            "unknown suite '" + s + "'. Suites: european-throughput, "
            "path-dependent, divergence, portfolio-risk, scaling, convergence, "
            "version-study, gpu-divergence, layout-study, all");

    std::printf("%s", environment_report().c_str());
    print_table(rows);
    if (!opts.output_csv.empty()) {
        write_csv(opts.output_csv, rows);
        std::printf("CSV written: %s\n", opts.output_csv.c_str());
    }
    return 0;
}

}  // namespace quantcuda
