// QuantCUDA Risk Engine - double-precision Black-Scholes analytics.
//
// Analytical reference implementation used to:
//   * validate Monte Carlo prices,
//   * validate analytic and finite-difference Greeks,
//   * measure Monte Carlo convergence,
//   * separate numerical error from implementation bugs.
#pragma once

#include <cmath>
#include <stdexcept>

#include "types.hpp"

namespace quantcuda {

class BlackScholesPricer {
public:
    // Standard normal CDF (Abramowitz-Stegun-quality via erfc).
    static double norm_cdf(double x) { return 0.5 * std::erfc(-x * 0.70710678118654752440); }
    static double norm_pdf(double x) {
        return 0.39894228040143267794 * std::exp(-0.5 * x * x);
    }

    // d1, d2 with edge-case guards. T==0 or sigma==0 handled by price()/greeks().
    static double d1(double S, double K, double r, double q, double sigma, double T) {
        return (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
               (sigma * std::sqrt(T));
    }
    static double d2(double S, double K, double r, double q, double sigma, double T) {
        return d1(S, K, r, q, sigma, T) - sigma * std::sqrt(T);
    }

    // Forward price of the underlying under risk-neutral drift.
    static double forward(double S, double r, double q, double T) {
        return S * std::exp((r - q) * T);
    }

    // Deterministic payoff limit: value when sigma -> 0 or T -> 0.
    static double deterministic_value(OptionType t, double S, double K, double r,
                                      double q, double T) {
        double fwd = forward(S, r, q, T);
        double df  = std::exp(-r * T);
        return df * european_payoff(t, fwd, K);
    }

    static double price(OptionType t, double S, double K, double r, double q,
                        double sigma, double T) {
        validate_inputs(S, K, sigma, T, r, q);
        if (T <= 0.0)    return european_payoff(t, S, K);  // at maturity: intrinsic
        if (S == 0.0)    return s0_zero_value(t, K, r, T);
        if (sigma <= 0.0) return deterministic_value(t, S, K, r, q, T);

        double sqT = sigma * std::sqrt(T);
        double D1  = d1(S, K, r, q, sigma, T);
        double D2  = D1 - sqT;
        double dfR = std::exp(-r * T);
        double dfQ = std::exp(-q * T);

        if (is_call(t))
            return S * dfQ * norm_cdf(D1) - K * dfR * norm_cdf(D2);
        return K * dfR * norm_cdf(-D2) - S * dfQ * norm_cdf(-D1);
    }

    static double call(double S, double K, double r, double q, double sigma, double T) {
        return price(OptionType::EuropeanCall, S, K, r, q, sigma, T);
    }
    static double put(double S, double K, double r, double q, double sigma, double T) {
        return price(OptionType::EuropeanPut, S, K, r, q, sigma, T);
    }

    static Greeks greeks(OptionType t, double S, double K, double r, double q,
                         double sigma, double T) {
        validate_inputs(S, K, sigma, T, r, q);
        Greeks g;
        if (T <= 0.0) {
            // Symmetric convention at the strike; one-sided limits elsewhere.
            if (S == K)
                g.delta = is_call(t) ? 0.5 : -0.5;
            else
                g.delta = is_call(t) ? (S > K ? 1.0 : 0.0)
                                     : (S < K ? -1.0 : 0.0);
            g.gamma = 0.0;
            g.vega  = 0.0;
            return g;
        }
        if (S == 0.0) {
            // At K == 0 a call is exactly S*exp(-qT), including at the
            // boundary S == 0, so its right-hand delta is exp(-qT). A
            // zero-strike put remains identically worthless.
            if (K == 0.0)
                g.delta = is_call(t) ? std::exp(-q * T) : 0.0;
            else
                g.delta = is_call(t) ? 0.0 : -std::exp(-q * T);
            return g;
        }
        if (sigma <= 0.0) {
            double df  = std::exp(-r * T);
            double fwd = forward(S, r, q, T);
            bool itm = is_call(t) ? (fwd > K) : (fwd < K);
            g.delta = itm ? (is_call(t) ? 1.0 : -1.0) * df * std::exp((r - q) * T) : 0.0;
            g.gamma = 0.0;
            g.vega  = 0.0;
            return g;
        }
        double sqT = std::sqrt(T);
        double D1  = d1(S, K, r, q, sigma, T);
        double dfQ = std::exp(-q * T);

        g.delta = is_call(t) ? dfQ * norm_cdf(D1) : -dfQ * norm_cdf(-D1);
        g.gamma = dfQ * norm_pdf(D1) / (S * sigma * sqT);
        g.vega  = S * dfQ * norm_pdf(D1) * sqT;   // per 1.00 change in sigma
        return g;
    }

    // Put-call parity: C - P = S e^{-qT} - K e^{-rT}. Used by tests.
    static double parity_rhs(double S, double K, double r, double q, double T) {
        return S * std::exp(-q * T) - K * std::exp(-r * T);
    }

private:
    static void validate_inputs(double S, double K, double sigma, double T,
                                double r = 0.0, double q = 0.0) {
        if (!std::isfinite(S) || !std::isfinite(K) || !std::isfinite(r) ||
            !std::isfinite(q) || !std::isfinite(sigma) || !std::isfinite(T))
            throw std::invalid_argument("BlackScholes: all inputs must be finite");
        if (S < 0.0)      throw std::invalid_argument("BlackScholes: spot must be >= 0");
        if (K < 0.0)      throw std::invalid_argument("BlackScholes: strike must be >= 0");
        if (sigma < 0.0)  throw std::invalid_argument("BlackScholes: volatility must be >= 0");
        if (T < 0.0)      throw std::invalid_argument("BlackScholes: maturity must be >= 0");
    }

public:
    // Value when the underlying is worthless.
    static double s0_zero_value(OptionType t, double K, double r, double T) {
        return is_call(t) ? 0.0 : K * std::exp(-r * T);
    }
};

}  // namespace quantcuda
