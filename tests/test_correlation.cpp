// TC-COR-001..005: Cholesky + correlated scenario validation (P0/P1).
#include "quantcuda/cholesky.hpp"
#include "quantcuda/rng_cpu.hpp"
#include "test_util.hpp"

using namespace quantcuda;

namespace {
const std::vector<std::vector<double>> R3 = {
    {1.00, 0.65, -0.20},
    {0.65, 1.00, 0.10},
    {-0.20, 0.10, 1.00},
};
}

QC_TEST(correlation, cholesky_reconstruction) {
    auto L = CorrelationEngine::cholesky(R3);
    QC_CHECK(CorrelationEngine::reconstruction_error(R3, L) < 1e-12);
}

QC_TEST(correlation, identity_and_single_asset) {
    auto L1 = CorrelationEngine::cholesky({{1.0}});
    QC_CHECK_NEAR(L1[0], 1.0, 1e-15);
    auto L2 = CorrelationEngine::cholesky({{1.0, 0.0}, {0.0, 1.0}});
    QC_CHECK_NEAR(L2[0], 1.0, 1e-15);
    QC_CHECK_NEAR(L2[3], 1.0, 1e-15);
    QC_CHECK_NEAR(L2[1], 0.0, 1e-15);
    QC_CHECK_THROWS(CorrelationEngine::cholesky({}));
    QC_CHECK_THROWS(CorrelationEngine::cholesky({{1.0, 0.0}, {0.0}}));
    QC_CHECK_THROWS(CorrelationEngine::cholesky({{std::nan("")}}));
}

QC_TEST(correlation, non_positive_definite_rejected) {
    // Classic non-PD correlation matrix.
    std::vector<std::vector<double>> bad = {
        {1.0, 0.9, 0.9}, {0.9, 1.0, -0.9}, {0.9, -0.9, 1.0}};
    QC_CHECK_THROWS(CorrelationEngine::cholesky(bad));
    // Perfect negative correlation pair is singular.
    QC_CHECK_THROWS(CorrelationEngine::cholesky({{1.0, -1.0}, {-1.0, 1.0}}));
}

QC_TEST(correlation, near_singular_handled) {
    // rho = 1 - 1e-10: PD but ill-conditioned; decomposition must succeed and
    // reconstruction stay accurate.
    double rho = 1.0 - 1e-10;
    std::vector<std::vector<double>> near_sing = {{1.0, rho}, {rho, 1.0}};
    auto L = CorrelationEngine::cholesky(near_sing);
    QC_CHECK(CorrelationEngine::reconstruction_error(near_sing, L) < 1e-8);
    QC_CHECK_THROWS(CorrelationEngine::reconstruction_error(near_sing, {1.0}));
    QC_CHECK_THROWS(CorrelationEngine::max_abs_difference(
        near_sing, std::vector<std::vector<double>>{{1.0}}));
}

QC_TEST(correlation, empirical_matches_requested) {
    auto L = CorrelationEngine::cholesky(R3);
    const std::size_t n = 200'000, d = 3;
    std::vector<double> samples(n * d);
    std::vector<double> z(d), zc(d);
    for (std::size_t i = 0; i < n; ++i) {
        auto rng = make_path_rng(2024, i);
        for (std::size_t a = 0; a < d; ++a) z[a] = rng.next_normal();
        CorrelationEngine::apply(L, d, z, zc);
        for (std::size_t a = 0; a < d; ++a) samples[i * d + a] = zc[a];
    }
    auto emp = CorrelationEngine::empirical_correlation(samples, n, d);
    QC_CHECK(CorrelationEngine::max_abs_difference(R3, emp) < 0.01);

    // Public helpers reject dimensions that would otherwise index past the
    // supplied factor/sample buffers.
    QC_CHECK_THROWS(CorrelationEngine::apply(L, 2, z, zc));
    QC_CHECK_THROWS(CorrelationEngine::apply(
        L, d, std::span<const double>{}, std::span<double>(zc)));
    const std::span<const double> short_input(z.data(), d - 1);
    const std::span<double> short_output(zc.data(), d - 1);
    QC_CHECK_THROWS(CorrelationEngine::apply(L, d, short_input, zc));
    QC_CHECK_THROWS(CorrelationEngine::apply(L, d, z, short_output));
    QC_CHECK_THROWS(CorrelationEngine::apply(L, 0, z, zc));
    QC_CHECK_THROWS(CorrelationEngine::empirical_correlation({}, 0, d));
    QC_CHECK_THROWS(CorrelationEngine::empirical_correlation({}, 1, 0));
    QC_CHECK_THROWS(CorrelationEngine::empirical_correlation({1.0, 2.0}, 2, 2));
}
