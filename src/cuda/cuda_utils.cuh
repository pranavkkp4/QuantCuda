// QuantCUDA Risk Engine - CUDA error checking, event timing, RAII device memory.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcuda/timer.hpp"

namespace quantcuda::gpu {

#define QC_CUDA_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err_) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

// CUDA-event timer around a kernel sequence.
struct EventTimer {
    cudaEvent_t start = nullptr, stop = nullptr;
    EventTimer() {
        QC_CUDA_CHECK(cudaEventCreate(&start));
        QC_CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~EventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s = nullptr) { QC_CUDA_CHECK(cudaEventRecord(start, s)); }
    float end(cudaStream_t s = nullptr) {
        QC_CUDA_CHECK(cudaEventRecord(stop, s));
        QC_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        QC_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// RAII device buffer.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t n = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    void allocate(std::size_t count) {
        release();
        n = count;
        if (n > 0) QC_CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    void release() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        n = 0;
    }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void copy_from(const T* host, cudaStream_t s = nullptr) {
        QC_CUDA_CHECK(cudaMemcpyAsync(ptr, host, n * sizeof(T),
                                      cudaMemcpyHostToDevice, s));
    }
    void copy_to(T* host, cudaStream_t s = nullptr) const {
        QC_CUDA_CHECK(cudaMemcpyAsync(host, ptr, n * sizeof(T),
                                      cudaMemcpyDeviceToHost, s));
    }
};

inline dev::RunningMoments<double> combine_moment_partials(
    const std::vector<dev::MomentPartial>& partials) {
    dev::RunningMoments<double> combined;
    for (const auto& partial : partials) {
        const dev::RunningMoments<double> next{partial.count, partial.mean,
                                                partial.m2};
        combined = dev::moments_merge(combined, next);
    }
    return combined;
}

inline McResult finalize_moments(dev::RunningMoments<double> moments,
                                 std::size_t expected_paths,
                                 double discount,
                                 bool allow_implicit_zeros = false) {
    if (expected_paths == 0) return McResult{};
    if (moments.count != expected_paths &&
        !(allow_implicit_zeros && moments.count < expected_paths))
        throw std::runtime_error("CUDA moment count does not match path count");

    // Compacted barrier kernels only emit surviving payoffs. Merge the omitted
    // knocked-out paths as a zero-valued group so both mean and M2 use all paths.
    if (moments.count < expected_paths) {
        const dev::RunningMoments<double> zeros{
            static_cast<unsigned long long>(expected_paths - moments.count),
            0.0, 0.0};
        moments = dev::moments_merge(moments, zeros);
    }

    const double n = static_cast<double>(expected_paths);
    const double m2 = std::max(moments.m2, 0.0);
    const double var = expected_paths > 1 ? m2 / (n - 1.0) : 0.0;
    McResult r;
    r.mean_payoff = moments.mean;
    r.price = discount * moments.mean;
    r.variance = var * discount * discount;
    r.std_error = std::sqrt(r.variance / n);
    r.paths = expected_paths;
    return r;
}

// Finalize partial block moments into an McResult (host side, deterministic).
inline McResult finalize_partials(const std::vector<dev::MomentPartial>& partials,
                                  std::size_t paths, double discount,
                                  bool allow_implicit_zeros = false) {
    return finalize_moments(combine_moment_partials(partials), paths, discount,
                            allow_implicit_zeros);
}

// V0 and V1 intentionally retain per-thread global atomics for their benchmark
// identity. They accumulate x-center and (x-center)^2 around the analytic
// expected European payoff supplied by the host. Unlike raw x/x^2 moments,
// this remains well conditioned for, e.g., payoffs near 1e16 with 1e9 spread.
inline McResult finalize_shifted_sums(double shifted_sum,
                                      double shifted_sq_sum, double center,
                                      std::size_t paths, double discount) {
    if (paths == 0) return McResult{};
    const double n = static_cast<double>(paths);
    const double shifted_mean = shifted_sum / n;
    double m2 = std::fma(-shifted_mean, shifted_sum, shifted_sq_sum);
    const double roundoff =
        32.0 * std::numeric_limits<double>::epsilon() *
        (std::fabs(shifted_sq_sum) + std::fabs(shifted_mean * shifted_sum));
    if (std::fabs(m2) <= roundoff) m2 = 0.0;
    dev::RunningMoments<double> moments{
        static_cast<unsigned long long>(paths), center + shifted_mean,
        std::max(m2, 0.0)};
    return finalize_moments(moments, paths, discount);
}

}  // namespace quantcuda::gpu
