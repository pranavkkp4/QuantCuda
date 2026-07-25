// QuantCUDA Risk Engine - CUDA engine public interface.
//
// Declarations only: no CUDA headers leak into host translation units. When
// the build has no CUDA toolkit, every function in this namespace is provided
// by cuda_stub.cpp and throws std::runtime_error("CUDA not available").
#pragma once

#include <string>
#include <vector>

#include "greeks_cpu.hpp"       // McGreeksResult
#include "monte_carlo_cpu.hpp"  // McResult
#include "types.hpp"

namespace quantcuda::gpu {

// ---------------------------------------------------------------------------
// Device information
// ---------------------------------------------------------------------------
bool available();
std::string device_name();       // "none (CPU-only build)" when unavailable
int device_count();

// ---------------------------------------------------------------------------
// Single-option pricing (SoA batch pricing lives in benchmark runner)
// ---------------------------------------------------------------------------
McResult price_european(const OptionSpec& opt, double spot, double r, double q,
                        double sigma, const SimulationConfig& cfg);
McResult price_asian(const OptionSpec& opt, double spot, double r, double q,
                     double sigma, const SimulationConfig& cfg);
// mode: 0 = early-exit branch, 1 = boolean mask, 2 = compacted active paths
McResult price_barrier(const OptionSpec& opt, double spot, double r, double q,
                       double sigma, const SimulationConfig& cfg, int mode);

McResult price_option(const OptionSpec& opt, double spot, double r, double q,
                      double sigma, const SimulationConfig& cfg);

// ---------------------------------------------------------------------------
// Greeks (fused common-random-number kernel + pathwise delta)
// ---------------------------------------------------------------------------
McGreeksResult price_greeks_european(const OptionSpec& opt, double spot, double r,
                                     double q, double sigma,
                                     const SimulationConfig& cfg);
McResult delta_pathwise_european(const OptionSpec& opt, double spot, double r,
                                 double q, double sigma,
                                 const SimulationConfig& cfg);

// ---------------------------------------------------------------------------
// Correlated multi-asset simulation
// ---------------------------------------------------------------------------
// Generates scenarios x d correlated standard normals with Cholesky factor L
// (computed on host from market.correlation). Returns row-major matrix.
std::vector<double> correlated_shocks(const MarketData& market,
                                      std::size_t scenarios, std::uint64_t seed,
                                      bool use_shared_tiles);

// ---------------------------------------------------------------------------
// Portfolio risk on GPU (scenario P&L kernel + device sort + tail reduction)
// ---------------------------------------------------------------------------
RiskResult portfolio_risk(const Portfolio& pf, const MarketData& market,
                          const SimulationConfig& price_cfg, const RiskConfig& rcfg);

// ---------------------------------------------------------------------------
// Version study: time every GPU generation V0..V7 on one European workload.
// Returns rows (version, kernel_ms, e2e_ms, price, std_error).
// ---------------------------------------------------------------------------
struct VersionRow {
    GpuVersion version;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
    double price = 0.0;
    double std_error = 0.0;
    std::string note;
};
std::vector<VersionRow> european_version_study(const OptionSpec& opt, double spot,
                                               double r, double q, double sigma,
                                               const SimulationConfig& cfg);

// Standalone V3 reduction-flavor comparison. The version-study benchmark
// appends these four rows so every preserved reduction is externally runnable.
struct ReductionRow {
    int flavor = 0;  // 0 atomic, 1 smem tree, 2 warp+atomic, 3 warp+smem
    std::string name;
    double kernel_ms = 0.0;
    double result = 0.0;
};
std::vector<ReductionRow> reduction_study(std::size_t n, std::uint64_t seed,
                                          int block_size);

// Barrier divergence study: near/medium/far barrier x branch/mask kernels.
struct DivergenceRow {
    std::string barrier_position;
    double barrier = 0.0;
    int mode = 0;
    double kernel_ms = 0.0;
    double price = 0.0;
    double knockout_fraction = 0.0;
};
std::vector<DivergenceRow> barrier_divergence_study(const OptionSpec& opt,
                                                    double spot, double r, double q,
                                                    double sigma,
                                                    const SimulationConfig& cfg);

// AoS vs SoA batch layout study (V1). Returns two rows (AoS, SoA).
struct LayoutRow {
    bool aos = false;
    double kernel_ms = 0.0;
    double e2e_ms = 0.0;
};
std::vector<LayoutRow> layout_study(int instruments, std::size_t paths_each,
                                    std::uint64_t seed, int warmup, int repeats);

}  // namespace quantcuda::gpu
