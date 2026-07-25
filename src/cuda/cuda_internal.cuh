// QuantCUDA Risk Engine - internal declarations shared between .cu files.
// Host-side helpers live in namespace quantcuda::gpu and are not part of the
// public API (see include/quantcuda/cuda_pricer.hpp).
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernels_common.cuh"
#include "quantcuda/cuda_pricer.hpp"
#include "quantcuda/monte_carlo_cpu.hpp"
#include "quantcuda/types.hpp"

namespace quantcuda::gpu {

// Timed result of a GPU pricing run.
struct GpuMcRun {
    McResult stats;
    double kernel_ms = 0.0;   // CUDA-event device time (kernel sequence only)
    double e2e_ms = 0.0;      // host wall clock incl. alloc/copy/launch
    double extra = 0.0;       // e.g. knockout fraction for barrier runs
};

// ---- rng.cu ---------------------------------------------------------------
void init_rng(curandStatePhilox4_32_10_t* states, std::uint64_t seed,
              std::size_t path_count, int block_size, cudaStream_t stream);

// ---- european.cu ----------------------------------------------------------
GpuMcRun european_gpu(const dev::OptionParams&, std::size_t paths, std::uint64_t seed,
                      Precision precision, GpuVersion version, int block_size);
GpuMcRun european_soa_batch_gpu(const std::vector<dev::OptionParams>& batch,
                                std::size_t paths_each, std::uint64_t seed,
                                int block_size, bool use_aos,
                                Precision precision = Precision::Double);
GpuMcRun european_async_gpu(const dev::OptionParams&, std::size_t paths,
                            std::uint64_t seed, Precision precision,
                            int block_size, int n_streams);

// ---- asian.cu -------------------------------------------------------------
GpuMcRun asian_gpu(const dev::OptionParams&, std::size_t paths, std::uint64_t seed,
                   Precision precision, int block_size);

// ---- barrier.cu -----------------------------------------------------------
// mode: 0 early-exit, 1 boolean mask, 2 compacted active paths
GpuMcRun barrier_gpu(const dev::OptionParams&, std::size_t paths, std::uint64_t seed,
                     Precision precision, int block_size, int mode);

// ---- greeks.cu ------------------------------------------------------------
struct GpuGreeksRun {
    double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0;
    double price_se = 0.0;
    double kernel_ms = 0.0;
};
GpuGreeksRun greeks_crn_gpu(const dev::OptionParams&, std::size_t paths,
                            std::uint64_t seed, double h, double h_sigma,
                            int block_size);
GpuMcRun delta_pathwise_gpu(const dev::OptionParams&, std::size_t paths,
                            std::uint64_t seed, int block_size);

// ---- correlation.cu -------------------------------------------------------
// scenarios x d row-major correlated standard normals.
std::vector<double> correlated_shocks_gpu(const std::vector<double>& chol,
                                          std::size_t d, std::size_t scenarios,
                                          std::uint64_t seed,
                                          bool use_shared_tiles,
                                          double* kernel_ms_out);

// ---- portfolio_risk.cu ----------------------------------------------------
// Scenario portfolio values; the public host wrapper converts V_i to V0 - V_i.
std::vector<double> portfolio_losses_gpu(
    const std::vector<double>& shocks, std::size_t scenarios, std::size_t d,
    const std::vector<double>& spots,           // d
    const std::vector<double>& vols,            // d
    const std::vector<double>& dividends,       // d
    double rate, double horizon,
    // SoA instrument arrays (n instruments)
    const std::vector<int>& instr_asset, const std::vector<int>& instr_is_call,
    const std::vector<int>& instr_analytic, const std::vector<double>& instr_strike,
    const std::vector<double>& instr_maturity, const std::vector<double>& instr_base,
    const std::vector<double>& instr_delta, const std::vector<double>& instr_gamma,
    const std::vector<double>& instr_quantity,
    double* kernel_ms_out);

// Sort losses on device and compute VaR/CVaR (levels in (0,1)).
std::vector<VarEstimate> var_cvar_gpu(std::vector<double> losses_device_copy,
                                      const std::vector<double>& levels);

}  // namespace quantcuda::gpu
