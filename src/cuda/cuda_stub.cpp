// QuantCUDA Risk Engine - stub GPU implementation for CPU-only builds.
// Compiled when no CUDA toolkit is present; the CLI reports the situation and
// all scalar/OpenMP functionality remains available.
#include <stdexcept>

#include "quantcuda/cuda_pricer.hpp"

namespace quantcuda::gpu {

namespace {
[[noreturn]] void no_cuda() {
    throw std::runtime_error(
        "CUDA backend not available: this binary was built without a CUDA "
        "toolkit. Rebuild on a machine with the CUDA toolkit installed "
        "(cmake auto-detects it; see README.md section 'Building with CUDA').");
}
}  // namespace

bool available() { return false; }
std::string device_name() { return "none (CPU-only build)"; }
int device_count() { return 0; }

McResult price_european(const OptionSpec&, double, double, double, double,
                        const SimulationConfig&) { no_cuda(); }
McResult price_asian(const OptionSpec&, double, double, double, double,
                     const SimulationConfig&) { no_cuda(); }
McResult price_barrier(const OptionSpec&, double, double, double, double,
                       const SimulationConfig&, int) { no_cuda(); }
McResult price_option(const OptionSpec&, double, double, double, double,
                      const SimulationConfig&) { no_cuda(); }
McGreeksResult price_greeks_european(const OptionSpec&, double, double, double,
                                     double, const SimulationConfig&) { no_cuda(); }
McResult delta_pathwise_european(const OptionSpec&, double, double, double, double,
                                 const SimulationConfig&) { no_cuda(); }
std::vector<double> correlated_shocks(const MarketData&, std::size_t, std::uint64_t,
                                      bool) { no_cuda(); }
RiskResult portfolio_risk(const Portfolio&, const MarketData&,
                          const SimulationConfig&, const RiskConfig&) { no_cuda(); }
std::vector<VersionRow> european_version_study(const OptionSpec&, double, double,
                                               double, double,
                                               const SimulationConfig&) { no_cuda(); }
std::vector<ReductionRow> reduction_study(std::size_t, std::uint64_t, int) {
    no_cuda();
}
std::vector<DivergenceRow> barrier_divergence_study(const OptionSpec&, double,
                                                    double, double, double,
                                                    const SimulationConfig&) {
    no_cuda();
}
std::vector<LayoutRow> layout_study(int, std::size_t, std::uint64_t, int, int) {
    no_cuda();
}

}  // namespace quantcuda::gpu
