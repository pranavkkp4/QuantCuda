// QuantCUDA Risk Engine - deterministic counter-based CPU random numbers.
//
// Implements the Philox4x32-10 counter-based generator (Salmon et al., SC'11)
// - the same family used by cuRAND's curandStatePhilox4_32_10_t on the GPU -
// plus a Box-Muller transform for standard normals.
//
// Counter-based design means the random stream for a path depends only on
// (seed, path_id), never on thread scheduling, so scalar and OpenMP engines
// produce bit-identical per-path draws for any thread count.
#pragma once

#include <cstdint>
#include <cmath>

namespace quantcuda {

struct Philox4x32_10 {
    // Salmon et al. constants.
    static constexpr std::uint32_t M0 = 0xD2511F53u;
    static constexpr std::uint32_t M1 = 0xCD9E8D57u;
    static constexpr std::uint32_t W0 = 0x9E3779B9u;  // key-bump constants
    static constexpr std::uint32_t W1 = 0xBB67AE85u;

    std::uint64_t counter = 0;      // low 64 bits of the 128-bit counter
    std::uint64_t subsequence = 0;  // high 64 bits: independent path/stream id
    std::uint64_t key     = 0;      // 64-bit seed
    std::uint32_t cache[4] = {0, 0, 0, 0};
    int cache_pos = 4;           // 4 => cache exhausted

    explicit Philox4x32_10(std::uint64_t seed = 0, std::uint64_t ctr = 0,
                           std::uint64_t stream = 0)
        : counter(ctr), subsequence(stream), key(seed) {}

    struct U4 { std::uint32_t v[4]; };

    static U4 round4(std::uint32_t c0, std::uint32_t c1, std::uint32_t c2,
                     std::uint32_t c3, std::uint32_t k0, std::uint32_t k1) {
        for (int i = 0; i < 10; ++i) {
            std::uint64_t p0 = static_cast<std::uint64_t>(M0) * c0;
            std::uint64_t p1 = static_cast<std::uint64_t>(M1) * c2;
            std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32);
            std::uint32_t lo0 = static_cast<std::uint32_t>(p0);
            std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32);
            std::uint32_t lo1 = static_cast<std::uint32_t>(p1);
            std::uint32_t n0 = hi1 ^ c1 ^ k0;
            std::uint32_t n1 = lo1;
            std::uint32_t n2 = hi0 ^ c3 ^ k1;
            std::uint32_t n3 = lo0;
            c0 = n0; c1 = n1; c2 = n2; c3 = n3;
            k0 += W0; k1 += W1;
        }
        return U4{c0, c1, c2, c3};
    }

    void refill() {
        std::uint32_t c0 = static_cast<std::uint32_t>(counter & 0xFFFFFFFFu);
        std::uint32_t c1 = static_cast<std::uint32_t>(counter >> 32);
        std::uint32_t c2 = static_cast<std::uint32_t>(subsequence & 0xFFFFFFFFu);
        std::uint32_t c3 = static_cast<std::uint32_t>(subsequence >> 32);
        U4 r = round4(c0, c1, c2, c3,
                      static_cast<std::uint32_t>(key & 0xFFFFFFFFu),
                      static_cast<std::uint32_t>(key >> 32));
        cache[0] = r.v[0]; cache[1] = r.v[1];
        cache[2] = r.v[2]; cache[3] = r.v[3];
        cache_pos = 0;
        ++counter;
        if (counter == 0) ++subsequence;  // carry through the full 128-bit counter
    }

    std::uint32_t next_u32() {
        if (cache_pos >= 4) refill();
        return cache[cache_pos++];
    }

    // Uniform on (0,1): 53-bit mantissa, never exactly 0 (Box-Muller log safety).
    double next_uniform_open01() {
        std::uint64_t hi = next_u32();
        std::uint64_t lo = next_u32();
        std::uint64_t bits = (hi << 21) | (lo >> 11);   // 53 random bits
        return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
    }

    // Standard normal via Box-Muller.
    double next_normal() {
        double u1 = next_uniform_open01();
        double u2 = next_uniform_open01();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.2831853071795864769 * u2);
    }
};

// Convenience: a generator positioned at a given path (and step within path).
// Stream layout uses the high half of Philox's 128-bit counter as the path id
// and the low half as the step/block index. Unlike a fixed numeric stride, this
// cannot overlap when a path consumes more than a few thousand draws.
inline Philox4x32_10 make_path_rng(std::uint64_t seed, std::uint64_t path_id,
                                   std::uint64_t step = 0) {
    return Philox4x32_10(seed, step, path_id);
}

}  // namespace quantcuda
