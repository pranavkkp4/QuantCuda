// QuantCUDA Risk Engine - CUDA error checking, event timing, RAII device memory.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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

// Neumaier summation keeps the host-side combination of block partials from
// needlessly discarding low-order bits. The device still reports sum/sum(x^2),
// so use an FMA for the centered sum of squares and the unbiased N-1 sample
// variance expected by the CPU Monte Carlo contract.
inline double compensated_sum(const std::vector<double>& values) {
    double sum = 0.0;
    double correction = 0.0;
    for (double value : values) {
        const double next = sum + value;
        if (std::fabs(sum) >= std::fabs(value))
            correction += (sum - next) + value;
        else
            correction += (value - next) + sum;
        sum = next;
    }
    return sum + correction;
}

inline double stable_centered_sum_squares(double sum, double sq,
                                          double count) {
    const double mean = sum / count;
    double centered = std::fma(-mean, sum, sq);  // sq - sum^2 / n
    // Raw second moments cannot resolve a variance beneath their rounding
    // envelope. Treat that envelope as zero instead of reporting numerical
    // noise as genuine Monte Carlo dispersion.
    const double roundoff =
        32.0 * std::numeric_limits<double>::epsilon() *
        (std::fabs(sq) + std::fabs(mean * sum));
    if (std::fabs(centered) <= roundoff) centered = 0.0;
    return std::max(centered, 0.0);
}

inline McResult finalize_sums(double sum, double sq, std::size_t paths,
                              double discount) {
    if (paths == 0) return McResult{};
    const double n = static_cast<double>(paths);
    const double mean = sum / n;
    const double centered_sq = stable_centered_sum_squares(sum, sq, n);
    const double var = paths > 1 ? centered_sq / (n - 1.0) : 0.0;
    McResult r;
    r.mean_payoff = mean;
    r.price = discount * mean;
    r.variance = var * discount * discount;
    r.std_error = std::sqrt(r.variance / n);
    r.paths = paths;
    return r;
}

// Finalize partial block sums into an McResult (host side, deterministic).
inline McResult finalize_partials(const std::vector<double>& block_sums,
                                  const std::vector<double>& block_sq_sums,
                                  std::size_t paths, double discount) {
    return finalize_sums(compensated_sum(block_sums),
                         compensated_sum(block_sq_sums), paths, discount);
}

}  // namespace quantcuda::gpu
