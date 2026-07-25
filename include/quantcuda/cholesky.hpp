// QuantCUDA Risk Engine - Cholesky decomposition (CPU).
//
// Decomposition is done on the host because the correlation matrix is reused
// across millions of scenarios; the factor L is transferred to the GPU once.
// Non-positive-definite input is rejected with a descriptive error.
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace quantcuda {

class CorrelationEngine {
public:
    // Cholesky-Banachiewicz. Returns lower-triangular L (row-major d x d)
    // such that R = L L^T. Throws std::invalid_argument if R is not
    // (numerically) positive definite.
    static std::vector<double> cholesky(const std::vector<std::vector<double>>& R) {
        const std::size_t d = R.size();
        validate_square_matrix(R, "cholesky");
        std::vector<double> L(checked_product(d, d, "cholesky"), 0.0);
        for (std::size_t i = 0; i < d; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double sum = R[i][j];
                for (std::size_t k = 0; k < j; ++k) sum -= L[i * d + k] * L[j * d + k];
                if (i == j) {
                    if (!std::isfinite(sum) || sum <= 0.0)
                        throw std::invalid_argument(
                            "cholesky: correlation matrix is not positive definite "
                            "(pivot " + std::to_string(i) + " = " + std::to_string(sum) + ")");
                    L[i * d + i] = std::sqrt(sum);
                } else {
                    L[i * d + j] = sum / L[j * d + j];
                }
            }
        }
        return L;
    }

    // Frobenius norm ||R - L L^T||_F for reconstruction validation.
    static double reconstruction_error(const std::vector<std::vector<double>>& R,
                                       const std::vector<double>& L) {
        const std::size_t d = R.size();
        validate_square_matrix(R, "reconstruction_error");
        const std::size_t expected = checked_product(d, d, "reconstruction_error");
        if (L.size() != expected)
            throw std::invalid_argument(
                "reconstruction_error: factor size does not match matrix dimension");
        for (double value : L)
            if (!std::isfinite(value))
                throw std::invalid_argument("reconstruction_error: factor contains non-finite value");
        double err = 0.0;
        for (std::size_t i = 0; i < d; ++i)
            for (std::size_t j = 0; j < d; ++j) {
                double ll = 0.0;
                for (std::size_t k = 0; k <= std::min(i, j); ++k)
                    ll += L[i * d + k] * L[j * d + k];
                double diff = R[i][j] - ll;
                err += diff * diff;
            }
        return std::sqrt(err);
    }

    // z_corr = L z (L row-major lower-triangular d x d).
    static void apply(const std::vector<double>& L, std::size_t d,
                      std::span<const double> z_indep,
                      std::span<double> z_corr) {
        if (d == 0) throw std::invalid_argument("correlation apply: dimension must be > 0");
        const std::size_t expected = checked_product(d, d, "correlation apply");
        if (L.size() != expected)
            throw std::invalid_argument(
                "correlation apply: factor size does not match dimension");
        if (z_indep.size() != d || z_corr.size() != d)
            throw std::invalid_argument(
                "correlation apply: input and output sizes must equal dimension");

        // Snapshot only when the spans overlap. Otherwise writing an early
        // output component can clobber an independent draw needed by a later
        // row of L. std::less provides a total order even for pointers into
        // different allocations; the usual disjoint-buffer path stays
        // allocation-free.
        const double* input_begin = z_indep.data();
        const double* input_end = input_begin + d;
        const double* output_begin = z_corr.data();
        const double* output_end = output_begin + d;
        const std::less<const double*> before;
        std::vector<double> input_copy;
        if (before(input_begin, output_end) && before(output_begin, input_end)) {
            input_copy.assign(z_indep.begin(), z_indep.end());
            z_indep = std::span<const double>(input_copy);
        }
        for (std::size_t i = 0; i < d; ++i) {
            double sum = 0.0;
            for (std::size_t k = 0; k <= i; ++k) sum += L[i * d + k] * z_indep[k];
            z_corr[i] = sum;
        }
    }

    // Empirical correlation matrix of samples (n x d, row-major).
    static std::vector<std::vector<double>> empirical_correlation(
        const std::vector<double>& samples, std::size_t n, std::size_t d) {
        if (n == 0) throw std::invalid_argument("empirical correlation: sample count must be > 0");
        if (d == 0) throw std::invalid_argument("empirical correlation: dimension must be > 0");
        const std::size_t expected = checked_product(n, d, "empirical correlation");
        if (samples.size() != expected)
            throw std::invalid_argument(
                "empirical correlation: sample vector size does not equal n * d");
        const std::size_t matrix_size = checked_product(d, d, "empirical correlation");
        for (double value : samples)
            if (!std::isfinite(value))
                throw std::invalid_argument("empirical correlation: samples must be finite");
        std::vector<double> mean(d, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < d; ++a) mean[a] += samples[i * d + a];
        for (double& m : mean) m /= static_cast<double>(n);

        std::vector<double> cov(matrix_size, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < d; ++a) {
                double da = samples[i * d + a] - mean[a];
                for (std::size_t b = 0; b <= a; ++b)
                    cov[a * d + b] += da * (samples[i * d + b] - mean[b]);
            }
        std::vector<std::vector<double>> out(d, std::vector<double>(d, 0.0));
        for (std::size_t a = 0; a < d; ++a)
            for (std::size_t b = 0; b <= a; ++b) {
                double c = cov[a * d + b] /
                           std::sqrt(std::max(cov[a * d + a], 1e-300) *
                                     std::max(cov[b * d + b], 1e-300));
                out[a][b] = out[b][a] = (a == b) ? 1.0 : c;
            }
        return out;
    }

    static double max_abs_difference(const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("matrix difference: row counts do not match");
        double m = 0.0;
        for (std::size_t i = 0; i < A.size(); ++i) {
            if (A[i].size() != B[i].size())
                throw std::invalid_argument("matrix difference: row sizes do not match");
            for (std::size_t j = 0; j < A[i].size(); ++j) {
                if (!std::isfinite(A[i][j]) || !std::isfinite(B[i][j]))
                    throw std::invalid_argument("matrix difference: entries must be finite");
                m = std::max(m, std::fabs(A[i][j] - B[i][j]));
            }
        }
        return m;
    }

private:
    static std::size_t checked_product(std::size_t a, std::size_t b,
                                       const char* context) {
        if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::invalid_argument(std::string(context) + ": dimensions overflow");
        return a * b;
    }

    static void validate_square_matrix(const std::vector<std::vector<double>>& matrix,
                                       const char* context) {
        const std::size_t d = matrix.size();
        if (d == 0)
            throw std::invalid_argument(std::string(context) + ": matrix must be non-empty");
        (void)checked_product(d, d, context);
        for (const auto& row : matrix) {
            if (row.size() != d)
                throw std::invalid_argument(std::string(context) + ": matrix not square");
            for (double value : row)
                if (!std::isfinite(value))
                    throw std::invalid_argument(std::string(context) +
                                                ": matrix entries must be finite");
        }
    }
};

}  // namespace quantcuda
