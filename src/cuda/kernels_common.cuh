// QuantCUDA Risk Engine - shared device helpers for all CUDA kernels.
//
// Design rules (from the project specification):
//   * no virtual dispatch on the device - one specialized kernel per
//     instrument type,
//   * fused simulation + payoff + block-level reduction in the optimized
//     generations, with every earlier generation preserved for benchmarks,
//   * template on simulation real type / accumulator type to support the
//     float / double / mixed-precision study.
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstddef>
#include <cstdint>

namespace quantcuda::gpu::dev {

// ---------------------------------------------------------------------------
// Instrument parameters passed to kernels BY VALUE (fast constant-cache path)
// ---------------------------------------------------------------------------
struct OptionParams {
    double spot;
    double strike;
    double rate;
    double dividend;
    double volatility;
    double maturity;
    double barrier;
    int is_call;      // 1 call, 0 put
    int barrier_dir;  // 0 none, +1 up-and-out, -1 down-and-out
    int time_steps;
};

// SoA batch layout (V1+): adjacent threads read adjacent fields.
struct DeviceOptionBatch {
    double* spots;
    double* strikes;
    double* rates;
    double* dividends;
    double* volatilities;
    double* maturities;
    double* barriers;
    int* is_call;
    int* barrier_dir;
    int* time_steps;
    int count;
};

// AoS layout retained ONLY for the V0 naïve baseline benchmark.
struct OptionAoS {
    double spot, strike, rate, dividend, volatility, maturity, barrier;
    int is_call, barrier_dir, time_steps;
};

// ---------------------------------------------------------------------------
// Device math
// ---------------------------------------------------------------------------
__device__ __forceinline__ double norm_cdf_d(double x) {
    return 0.5 * erfc(-x * 0.70710678118654752440);
}

template <typename Real>
__device__ __forceinline__ Real gbm_drift(Real r, Real q, Real sigma, Real dt) {
    return (r - q - Real(0.5) * sigma * sigma) * dt;
}

template <typename Real>
__device__ __forceinline__ Real gbm_diffusion(Real sigma, Real dt) {
    return sigma * sqrt(dt);
}

template <typename Real>
__device__ __forceinline__ Real european_payoff_d(int is_call, Real st, Real k) {
    return is_call ? fmax(st - k, Real(0)) : fmax(k - st, Real(0));
}

// ---------------------------------------------------------------------------
// Reductions - all four generations kept for the comparative benchmark:
//   0: global atomicAdd per thread (V0)
//   1: shared-memory tree reduction
//   2: warp-shuffle reduction
//   3: warp-shuffle + one shared value per warp (default in fused kernels)
// ---------------------------------------------------------------------------
template <typename AccT>
__device__ __forceinline__ AccT warp_reduce(AccT v) {
    const unsigned mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int active_lanes = __popc(mask);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        AccT other = __shfl_down_sync(mask, v, offset);
        if (lane + offset < active_lanes) v += other;
    }
    return v;
}

// Shared-memory tree reduction. smem must hold blockDim.x elements. The
// rounded-up active count keeps every element even for internal launches whose
// block size is not a power of two (public pricing APIs require powers of two).
template <typename AccT>
__device__ AccT block_reduce_smem(AccT v, AccT* smem) {
    int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int active = blockDim.x; active > 1; active = (active + 1) / 2) {
        const int next = (active + 1) / 2;
        if (tid + next < active) smem[tid] += smem[tid + next];
        __syncthreads();
    }
    return smem[0];
}

// Warp shuffle + one shared slot per warp (few shared bytes, few syncs).
template <typename AccT>
__device__ AccT block_reduce_warp(AccT v, AccT* smem /* >= 32 slots */) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : AccT(0);
    if (warp == 0) v = warp_reduce(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------
__device__ __forceinline__ std::size_t global_thread_id() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ std::size_t grid_stride() {
    return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

inline int grid_for(std::size_t paths, int block_size) {
    std::size_t blocks = (paths + static_cast<std::size_t>(block_size) - 1) /
                         static_cast<std::size_t>(block_size);
    const std::size_t cap = 65535;  // plenty of headroom; grid-stride covers rest
    return static_cast<int>(blocks > cap ? cap : blocks);
}

}  // namespace quantcuda::gpu::dev
