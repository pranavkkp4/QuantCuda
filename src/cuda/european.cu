// QuantCUDA Risk Engine - European option CUDA kernels, generations V0-V7.
//
// Every generation is preserved as a benchmarkable version:
//   V0 naive     : separate RNG/sim/payoff kernels, global arrays, atomicAdd,
//                  synchronous host-device copies
//   V1 SoA       : coalesced structure-of-arrays batch layout (vs AoS)
//   V2 fused     : one kernel = draw + terminal price + payoff + block reduce
//   V3 reduction : warp-shuffle block reduction replaces the shared tree
//   V5 memory    : parameters via __constant__ memory vs by-value passing
//   V6 async     : pinned memory + chunked multi-stream execution
//   V7 precision : float / double / mixed (float sim, double accumulation)
#include "cuda_internal.cuh"
#include "cuda_utils.cuh"
#include "quantcuda/black_scholes.hpp"

#include <type_traits>

namespace quantcuda::gpu {
namespace {

// ---------------------------------------------------------------------------
// Precision helpers
// ---------------------------------------------------------------------------
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

// Exact expected undiscounted payoff for the European model. V0/V1 subtract
// this control center before their defining global atomic reductions, keeping
// shifted moments well conditioned even when deterministic carry moves a very
// large payoff far from its spot intrinsic value.
double payoff_center(const dev::OptionParams& opt) {
    const OptionType type = opt.is_call ? OptionType::EuropeanCall
                                        : OptionType::EuropeanPut;
    const double discounted = BlackScholesPricer::price(
        type, opt.spot, opt.strike, opt.rate, opt.dividend, opt.volatility,
        opt.maturity);
    const double center = discounted * std::exp(opt.rate * opt.maturity);
    if (std::isfinite(center)) return center;
    return opt.is_call ? std::fmax(opt.spot - opt.strike, 0.0)
                       : std::fmax(opt.strike - opt.spot, 0.0);
}

// ===========================================================================
// V0 - naive: three kernels + global arrays + atomic reduction
// ===========================================================================
template <typename Real>
__global__ void v0_simulate_terminal_kernel(const dev::OptionAoS* options,
                                            curandStatePhilox4_32_10_t* states,
                                            Real* terminal_prices,
                                            std::size_t path_count) {
    const dev::OptionAoS opt = options[0];
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility),
                                      static_cast<Real>(opt.maturity));
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility),
                                         static_cast<Real>(opt.maturity));
    const std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real z = draw_normal<Real>(&state);
        terminal_prices[i] = static_cast<Real>(opt.spot) * exp(drift + diff * z);
    }
}

template <typename Real>
__global__ void v0_payoff_kernel(const dev::OptionAoS* options,
                                 const Real* terminal_prices,
                                 Real* payoffs, std::size_t path_count) {
    const dev::OptionAoS opt = options[0];
    const std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride)
        payoffs[i] = dev::european_payoff_d<Real>(
            opt.is_call, terminal_prices[i], static_cast<Real>(opt.strike));
}

// Every thread atomically adds into one global accumulator (slow by design -
// this is the baseline the optimized reductions are measured against).
template <typename Real>
__global__ void v0_atomic_reduce_kernel(const Real* payoffs, std::size_t path_count,
                                        double center, double* shifted_sum,
                                        double* shifted_sq_sum) {
    const std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        const double shifted = static_cast<double>(payoffs[i]) - center;
        atomicAdd(shifted_sum, shifted);
        atomicAdd(shifted_sq_sum, shifted * shifted);
    }
}

// ===========================================================================
// V1 - SoA batch (one block row per instrument, coalesced parameter reads)
// ===========================================================================
template <typename Real, bool UseAoS>
__global__ void v1_batch_kernel(const void* params, /*SoA ptr or AoS ptr*/
                                int option_count, std::size_t paths_each,
                                const double* payoff_centers,
                                curandStatePhilox4_32_10_t* states,
                                double* option_sums, double* option_sq_sums) {
    const std::size_t total = static_cast<std::size_t>(option_count) * paths_each;
    const std::size_t stride = dev::grid_stride();
    for (std::size_t idx = dev::global_thread_id(); idx < total; idx += stride) {
        // Path-major ordering makes adjacent threads read adjacent instruments.
        // AoS therefore performs strided field reads while SoA performs one
        // coalesced read per field.
        const int instrument = static_cast<int>(idx % static_cast<std::size_t>(option_count));
        dev::OptionParams opt;
        if constexpr (UseAoS) {
            const dev::OptionAoS* aos = static_cast<const dev::OptionAoS*>(params);
            dev::OptionAoS o = aos[instrument];
            opt = {o.spot, o.strike, o.rate, o.dividend, o.volatility, o.maturity,
                   o.barrier, o.is_call, o.barrier_dir, o.time_steps};
        } else {
            const dev::DeviceOptionBatch* soa =
                static_cast<const dev::DeviceOptionBatch*>(params);
            opt.spot = soa->spots[instrument];
            opt.strike = soa->strikes[instrument];
            opt.rate = soa->rates[instrument];
            opt.dividend = soa->dividends[instrument];
            opt.volatility = soa->volatilities[instrument];
            opt.maturity = soa->maturities[instrument];
            opt.barrier = soa->barriers[instrument];
            opt.is_call = soa->is_call[instrument];
            opt.barrier_dir = soa->barrier_dir[instrument];
            opt.time_steps = soa->time_steps[instrument];
        }
        Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                          static_cast<Real>(opt.dividend),
                                          static_cast<Real>(opt.volatility),
                                          static_cast<Real>(opt.maturity));
        Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility),
                                             static_cast<Real>(opt.maturity));
        curandStatePhilox4_32_10_t state = states[idx];
        Real z = draw_normal<Real>(&state);
        Real st = static_cast<Real>(opt.spot) * exp(drift + diff * z);
        const double p = static_cast<double>(
            dev::european_payoff_d<Real>(opt.is_call, st, static_cast<Real>(opt.strike)));
        const double shifted = p - payoff_centers[instrument];
        atomicAdd(option_sums + instrument, shifted);
        atomicAdd(option_sq_sums + instrument, shifted * shifted);
    }
}

// ===========================================================================
// V2/V3/V7 - fused kernel (simulation + payoff + reduction in one launch)
//   ReduceFlavor: 1 = shared tree (V2), 3 = warp shuffle + smem per warp (V3)
// ===========================================================================
template <typename Real, typename AccT, int ReduceFlavor>
__global__ void european_fused_kernel(dev::OptionParams opt,
                                      curandStatePhilox4_32_10_t* states,
                                      dev::MomentPartial* block_moments,
                                      std::size_t path_count) {
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(opt.rate),
                                      static_cast<Real>(opt.dividend),
                                      static_cast<Real>(opt.volatility),
                                      static_cast<Real>(opt.maturity));
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(opt.volatility),
                                         static_cast<Real>(opt.maturity));
    dev::RunningMoments<AccT> moments;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real z = draw_normal<Real>(&state);
        Real st = static_cast<Real>(opt.spot) * exp(drift + diff * z);
        AccT p = static_cast<AccT>(
            dev::european_payoff_d<Real>(opt.is_call, st, static_cast<Real>(opt.strike)));
        dev::moments_add(moments, p);
    }

    extern __shared__ char smem_raw[];
    auto* red = reinterpret_cast<dev::RunningMoments<AccT>*>(smem_raw);
    dev::RunningMoments<AccT> total;
    if constexpr (ReduceFlavor == 1) {
        total = dev::block_reduce_moments_smem(moments, red);
    } else {
        total = dev::block_reduce_moments_warp(moments, red);
    }
    if (threadIdx.x == 0) block_moments[blockIdx.x] = dev::moments_partial(total);
}

// ===========================================================================
// V5 - parameters in __constant__ memory (vs by-value kernel argument)
// ===========================================================================
__constant__ dev::OptionParams c_opt_params;

template <typename Real, typename AccT>
__global__ void european_fused_constmem_kernel(curandStatePhilox4_32_10_t* states,
                                               dev::MomentPartial* block_moments,
                                               std::size_t path_count) {
    Real drift = dev::gbm_drift<Real>(static_cast<Real>(c_opt_params.rate),
                                      static_cast<Real>(c_opt_params.dividend),
                                      static_cast<Real>(c_opt_params.volatility),
                                      static_cast<Real>(c_opt_params.maturity));
    Real diff = dev::gbm_diffusion<Real>(static_cast<Real>(c_opt_params.volatility),
                                         static_cast<Real>(c_opt_params.maturity));
    dev::RunningMoments<AccT> moments;
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = dev::global_thread_id(); i < path_count; i += stride) {
        curandStatePhilox4_32_10_t state = states[i];
        Real z = draw_normal<Real>(&state);
        Real st = static_cast<Real>(c_opt_params.spot) * exp(drift + diff * z);
        AccT p = static_cast<AccT>(dev::european_payoff_d<Real>(
            c_opt_params.is_call, st, static_cast<Real>(c_opt_params.strike)));
        dev::moments_add(moments, p);
    }
    extern __shared__ char smem_raw[];
    auto* red = reinterpret_cast<dev::RunningMoments<AccT>*>(smem_raw);
    const auto total = dev::block_reduce_moments_warp(moments, red);
    if (threadIdx.x == 0) block_moments[blockIdx.x] = dev::moments_partial(total);
}

// ---------------------------------------------------------------------------
// Host-side launch helpers
// ---------------------------------------------------------------------------
template <typename Real, typename AccT>
GpuMcRun run_fused(const dev::OptionParams& opt, std::size_t paths,
                   std::uint64_t seed, int flavor, bool use_constant_memory,
                   int block_size) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    int grid = dev::grid_for(paths, block_size);
    DeviceBuffer<dev::MomentPartial> moments(static_cast<std::size_t>(grid));

    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());

    // V5 measures the kernel, not the host-to-constant-memory setup copy.
    if (use_constant_memory)
        QC_CUDA_CHECK(cudaMemcpyToSymbol(c_opt_params, &opt, sizeof(opt)));

    EventTimer ev;
    const std::size_t moment_slots =
        (!use_constant_memory && flavor == 1)
            ? static_cast<std::size_t>(block_size)
            : std::size_t{32};
    std::size_t smem = moment_slots * sizeof(dev::RunningMoments<AccT>);
    ev.begin();
    if (use_constant_memory) {
        european_fused_constmem_kernel<Real, AccT>
            <<<grid, block_size, smem>>>(states.ptr, moments.ptr, paths);
    } else if (flavor == 1) {
        european_fused_kernel<Real, AccT, 1>
            <<<grid, block_size, smem>>>(opt, states.ptr, moments.ptr, paths);
    } else {
        european_fused_kernel<Real, AccT, 3>
            <<<grid, block_size, smem>>>(opt, states.ptr, moments.ptr, paths);
    }
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

template <typename Real, typename AccT>
GpuMcRun run_v0(const dev::OptionParams& opt, std::size_t paths, std::uint64_t seed,
                int block_size) {
    WallTimer wall;  // e2e includes all allocations and synchronous copies
    if (paths == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    DeviceBuffer<Real> terminal(paths);
    DeviceBuffer<Real> payoffs(paths);
    DeviceBuffer<double> acc(2);
    DeviceBuffer<dev::OptionAoS> options(1);
    const dev::OptionAoS h_option{opt.spot,       opt.strike,    opt.rate,
                                  opt.dividend,   opt.volatility, opt.maturity,
                                  opt.barrier,    opt.is_call,   opt.barrier_dir,
                                  opt.time_steps};
    QC_CUDA_CHECK(cudaMemcpy(options.ptr, &h_option, sizeof(h_option),
                             cudaMemcpyHostToDevice));
    QC_CUDA_CHECK(cudaMemset(acc.ptr, 0, 2 * sizeof(double)));

    init_rng(states.ptr, seed, paths, block_size, nullptr);
    QC_CUDA_CHECK(cudaDeviceSynchronize());  // V0: fully synchronous pipeline

    EventTimer ev;
    int grid = dev::grid_for(paths, block_size);
    ev.begin();
    v0_simulate_terminal_kernel<Real><<<grid, block_size>>>(options.ptr, states.ptr,
                                                            terminal.ptr, paths);
    v0_payoff_kernel<Real><<<grid, block_size>>>(options.ptr, terminal.ptr,
                                                 payoffs.ptr, paths);
    const double center = payoff_center(opt);
    v0_atomic_reduce_kernel<Real><<<grid, block_size>>>(
        payoffs.ptr, paths, center, acc.ptr, acc.ptr + 1);
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    double h_acc[2];
    QC_CUDA_CHECK(cudaMemcpy(h_acc, acc.ptr, sizeof(h_acc), cudaMemcpyDeviceToHost));
    GpuMcRun out;
    double df = exp(-opt.rate * opt.maturity);
    out.stats = finalize_shifted_sums(h_acc[0], h_acc[1], center, paths, df);
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    return out;
}

template <typename Real, typename AccT>
GpuMcRun dispatch_sim(const dev::OptionParams& opt, std::size_t paths,
                      std::uint64_t seed, GpuVersion version, int block_size) {
    switch (version) {
        case GpuVersion::V0_Naive:
            return run_v0<Real, AccT>(opt, paths, seed, block_size);
        case GpuVersion::V1_SoA: {
            constexpr Precision precision =
                std::is_same_v<Real, double>
                    ? Precision::Double
                    : (std::is_same_v<AccT, double> ? Precision::Mixed
                                                    : Precision::Float);
            return european_soa_batch_gpu({opt}, paths, seed, block_size,
                                          /*use_aos=*/false, precision);
        }
        case GpuVersion::V2_Fused:
            return run_fused<Real, AccT>(opt, paths, seed, /*flavor=*/1, false,
                                         block_size);
        case GpuVersion::V5_Memory:
            return run_fused<Real, AccT>(opt, paths, seed, /*flavor=*/3, true,
                                         block_size);
        case GpuVersion::V3_Reduction:
        case GpuVersion::V4_Specialized:
        case GpuVersion::V7_Precision:
        default:
            return run_fused<Real, AccT>(opt, paths, seed, /*flavor=*/3, false,
                                         block_size);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points (internal to the CUDA library)
// ---------------------------------------------------------------------------
GpuMcRun european_gpu(const dev::OptionParams& opt, std::size_t paths,
                      std::uint64_t seed, Precision precision, GpuVersion version,
                      int block_size) {
    switch (precision) {
        case Precision::Float:
            return dispatch_sim<float, float>(opt, paths, seed, version, block_size);
        case Precision::Mixed:
            return dispatch_sim<float, double>(opt, paths, seed, version, block_size);
        case Precision::Double:
        default:
            return dispatch_sim<double, double>(opt, paths, seed, version, block_size);
    }
}

// ===========================================================================
// V6 - asynchronous chunked execution with pinned host memory + streams
// ===========================================================================
GpuMcRun european_async_gpu(const dev::OptionParams& opt, std::size_t paths,
                            std::uint64_t seed, Precision precision, int block_size,
                            int n_streams) {
    WallTimer wall;
    if (paths == 0) return GpuMcRun{};
    if (n_streams < 1) n_streams = 1;
    std::size_t chunk = (paths + static_cast<std::size_t>(n_streams) - 1) /
                        static_cast<std::size_t>(n_streams);

    // Pinned host buffers for async transfers.
    dev::MomentPartial* h_moments = nullptr;
    QC_CUDA_CHECK(cudaHostAlloc(&h_moments, static_cast<std::size_t>(n_streams) *
                                            65536 * sizeof(dev::MomentPartial),
                                cudaHostAllocDefault));

    std::vector<cudaStream_t> streams(static_cast<std::size_t>(n_streams));
    for (auto& s : streams) QC_CUDA_CHECK(cudaStreamCreate(&s));

    DeviceBuffer<curandStatePhilox4_32_10_t> states(paths);
    init_rng(states.ptr, seed, paths, block_size, nullptr);

    EventTimer ev;
    ev.begin();
    DeviceBuffer<dev::MomentPartial> d_moments(
        static_cast<std::size_t>(n_streams) * 65536);
    for (int c = 0; c < n_streams; ++c) {
        std::size_t begin = static_cast<std::size_t>(c) * chunk;
        if (begin >= paths) continue;
        std::size_t count = std::min(chunk, paths - begin);
        cudaStream_t s = streams[static_cast<std::size_t>(c)];
        int grid = dev::grid_for(count, block_size);
        // Per-chunk kernel launched into its own stream; results copied back
        // asynchronously into pinned host memory.
        auto launch = [&](auto real_tag, auto acc_tag) {
            using Real = decltype(real_tag);
            using AccT = decltype(acc_tag);
            const std::size_t smem =
                32 * sizeof(dev::RunningMoments<AccT>);
            // Chunk c uses states[begin..]; the kernel indexes 0..count, so a
            // shifted state pointer keeps per-path streams reproducible.
            european_fused_kernel<Real, AccT, 3><<<grid, block_size, smem, s>>>(
                opt, states.ptr + begin,
                d_moments.ptr + static_cast<std::size_t>(c) * 65536, count);
        };
        if (precision == Precision::Float) launch(float{}, float{});
        else if (precision == Precision::Mixed) launch(float{}, double{});
        else launch(double{}, double{});

        QC_CUDA_CHECK(cudaMemcpyAsync(
                                      h_moments + static_cast<std::size_t>(c) * 65536,
                                      d_moments.ptr + static_cast<std::size_t>(c) * 65536,
                                      static_cast<std::size_t>(grid) *
                                          sizeof(dev::MomentPartial),
                                      cudaMemcpyDeviceToHost, s));
    }
    QC_CUDA_CHECK(cudaGetLastError());
    for (auto& s : streams) QC_CUDA_CHECK(cudaStreamSynchronize(s));
    float kernel_ms = ev.end();

    std::vector<dev::MomentPartial> all_moments;
    for (int c = 0; c < n_streams; ++c) {
        std::size_t begin = static_cast<std::size_t>(c) * chunk;
        if (begin >= paths) continue;
        std::size_t count = std::min(chunk, paths - begin);
        int grid = dev::grid_for(count, block_size);
        for (int g = 0; g < grid; ++g) {
            all_moments.push_back(
                h_moments[static_cast<std::size_t>(c) * 65536 +
                          static_cast<std::size_t>(g)]);
        }
    }
    GpuMcRun out;
    out.stats = finalize_partials(all_moments, paths,
                                  exp(-opt.rate * opt.maturity));
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();

    for (auto& s : streams) cudaStreamDestroy(s);
    cudaFreeHost(h_moments);
    return out;
}

// ===========================================================================
// V1 layout study: AoS vs SoA batch pricing (benchmark-only entry point)
// ===========================================================================
GpuMcRun european_soa_batch_gpu(const std::vector<dev::OptionParams>& batch,
                                std::size_t paths_each, std::uint64_t seed,
                                int block_size, bool use_aos,
                                Precision precision) {
    WallTimer wall;
    const int n = static_cast<int>(batch.size());
    std::size_t total_paths = static_cast<std::size_t>(n) * paths_each;
    if (n == 0 || paths_each == 0) return GpuMcRun{};
    DeviceBuffer<curandStatePhilox4_32_10_t> states(total_paths);
    init_rng(states.ptr, seed, total_paths, block_size, nullptr);
    DeviceBuffer<double> sums(static_cast<std::size_t>(n));
    DeviceBuffer<double> sqs(static_cast<std::size_t>(n));
    DeviceBuffer<double> centers(static_cast<std::size_t>(n));
    QC_CUDA_CHECK(cudaMemset(sums.ptr, 0, sums.n * sizeof(double)));
    QC_CUDA_CHECK(cudaMemset(sqs.ptr, 0, sqs.n * sizeof(double)));
    std::vector<double> h_centers(batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i)
        h_centers[i] = payoff_center(batch[i]);
    centers.copy_from(h_centers.data());

    DeviceBuffer<char> params_buf;
    DeviceBuffer<double> f_spots, f_strikes, f_rates, f_divs, f_vols, f_mats, f_bars;
    DeviceBuffer<int> f_call, f_dir, f_steps;
    if (use_aos) {
        std::vector<dev::OptionAoS> aos;
        aos.reserve(batch.size());
        for (const auto& o : batch)
            aos.push_back({o.spot, o.strike, o.rate, o.dividend, o.volatility,
                           o.maturity, o.barrier, o.is_call, o.barrier_dir,
                           o.time_steps});
        params_buf.allocate(sizeof(dev::OptionAoS) * aos.size());
        QC_CUDA_CHECK(cudaMemcpy(params_buf.ptr, aos.data(), params_buf.n,
                                 cudaMemcpyHostToDevice));
    } else {
        auto fill = [&](DeviceBuffer<double>& buf, auto member) {
            buf.allocate(batch.size());
            std::vector<double> h(batch.size());
            for (std::size_t i = 0; i < batch.size(); ++i) h[i] = batch[i].*member;
            QC_CUDA_CHECK(cudaMemcpy(buf.ptr, h.data(), buf.n * sizeof(double),
                                     cudaMemcpyHostToDevice));
        };
        auto fill_i = [&](DeviceBuffer<int>& buf, auto member) {
            buf.allocate(batch.size());
            std::vector<int> h(batch.size());
            for (std::size_t i = 0; i < batch.size(); ++i) h[i] = batch[i].*member;
            QC_CUDA_CHECK(cudaMemcpy(buf.ptr, h.data(), buf.n * sizeof(int),
                                     cudaMemcpyHostToDevice));
        };
        fill(f_spots, &dev::OptionParams::spot);
        fill(f_strikes, &dev::OptionParams::strike);
        fill(f_rates, &dev::OptionParams::rate);
        fill(f_divs, &dev::OptionParams::dividend);
        fill(f_vols, &dev::OptionParams::volatility);
        fill(f_mats, &dev::OptionParams::maturity);
        fill(f_bars, &dev::OptionParams::barrier);
        fill_i(f_call, &dev::OptionParams::is_call);
        fill_i(f_dir, &dev::OptionParams::barrier_dir);
        fill_i(f_steps, &dev::OptionParams::time_steps);
        dev::DeviceOptionBatch h_soa{f_spots.ptr, f_strikes.ptr, f_rates.ptr,
                                     f_divs.ptr,  f_vols.ptr,    f_mats.ptr,
                                     f_bars.ptr,   f_call.ptr,    f_dir.ptr,
                                     f_steps.ptr,  n};
        params_buf.allocate(sizeof(dev::DeviceOptionBatch));
        QC_CUDA_CHECK(cudaMemcpy(params_buf.ptr, &h_soa, sizeof(h_soa),
                                 cudaMemcpyHostToDevice));
    }
    QC_CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer ev;
    int grid = dev::grid_for(total_paths, block_size);
    ev.begin();
    auto launch = [&](auto real_tag) {
        using Real = decltype(real_tag);
        if (use_aos)
            v1_batch_kernel<Real, true><<<grid, block_size>>>(
                params_buf.ptr, n, paths_each, centers.ptr, states.ptr, sums.ptr,
                sqs.ptr);
        else
            v1_batch_kernel<Real, false><<<grid, block_size>>>(
                params_buf.ptr, n, paths_each, centers.ptr, states.ptr, sums.ptr,
                sqs.ptr);
    };
    if (precision == Precision::Double) launch(double{});
    else launch(float{});
    QC_CUDA_CHECK(cudaGetLastError());
    float kernel_ms = ev.end();

    std::vector<double> h_sums(static_cast<std::size_t>(n));
    std::vector<double> h_sqs(static_cast<std::size_t>(n));
    sums.copy_to(h_sums.data());
    sqs.copy_to(h_sqs.data());
    QC_CUDA_CHECK(cudaDeviceSynchronize());
    GpuMcRun out;
    out.kernel_ms = kernel_ms;
    out.e2e_ms = wall.elapsed_ms();
    if (n == 1)
        out.stats = finalize_shifted_sums(
            h_sums[0], h_sqs[0], payoff_center(batch[0]), paths_each,
            exp(-batch[0].rate * batch[0].maturity));
    else
        out.stats.paths = total_paths;
    return out;
}

}  // namespace quantcuda::gpu
