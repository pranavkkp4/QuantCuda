// QuantCUDA Risk Engine - Kernel 1: random-state initialization.
//
// One Philox4x32-10 state per simulation path. Philox is counter-based, so a
// fixed (seed, sequence) pair reproduces the identical stream on any GPU -
// the seed and generator configuration are recorded in every benchmark report.
#include "cuda_internal.cuh"
#include "cuda_utils.cuh"

namespace quantcuda::gpu {

__global__ void initialize_rng_kernel(curandStatePhilox4_32_10_t* states,
                                      unsigned long long seed,
                                      std::size_t path_count) {
    std::size_t tid = dev::global_thread_id();
    std::size_t stride = dev::grid_stride();
    for (std::size_t i = tid; i < path_count; i += stride) {
        // sequence == path id: independent stream per path.
        curand_init(seed, i, 0, &states[i]);
    }
}

void init_rng(curandStatePhilox4_32_10_t* states, std::uint64_t seed,
              std::size_t path_count, int block_size, cudaStream_t stream) {
    initialize_rng_kernel<<<dev::grid_for(path_count, block_size), block_size, 0,
                            stream>>>(states, seed, path_count);
    QC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace quantcuda::gpu
