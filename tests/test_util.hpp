// Minimal test framework for QuantCUDA: registration + CHECK macros + main().
// Tests follow the AAA pattern (Arrange-Act-Assert) and exercise public
// interfaces only, so they survive internal refactors.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qtest {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

inline int& failure_count() {
    static int n = 0;
    return n;
}

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

class SkipTest : public std::runtime_error {
public:
    explicit SkipTest(const std::string& reason) : std::runtime_error(reason) {}
};

inline void report_failure(const char* file, int line, const std::string& msg) {
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++failure_count();
}

inline int run_all() {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    std::string current_suite;
    for (const auto& tc : registry()) {
        if (tc.suite != current_suite) {
            current_suite = tc.suite;
            std::printf("[%s]\n", current_suite.c_str());
        }
        int before = failure_count();
        try {
            tc.fn();
        } catch (const SkipTest& e) {
            if (failure_count() == before) {
                std::printf("  SKIP %s (%s)\n", tc.name.c_str(), e.what());
                ++skipped;
            } else {
                std::printf("  FAIL %s (skip requested after a failure)\n",
                            tc.name.c_str());
                ++failed;
            }
            continue;
        } catch (const std::exception& e) {
            report_failure(__FILE__, __LINE__, std::string("unexpected exception: ") +
                                                  e.what());
        } catch (...) {
            report_failure(__FILE__, __LINE__, "unexpected non-standard exception");
        }
        if (failure_count() == before) {
            std::printf("  PASS %s\n", tc.name.c_str());
            ++passed;
        } else {
            std::printf("  FAIL %s\n", tc.name.c_str());
            ++failed;
        }
    }
    std::printf("--------------------------------------------------\n");
    std::printf("%zu tests, %d passed, %d failed, %d skipped\n", registry().size(),
                passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}

}  // namespace qtest

#define QC_TEST(suite, name)                                              \
    static void suite##_##name##_body();                                  \
    static ::qtest::Registrar suite##_##name##_reg(#suite, #name,         \
                                                   suite##_##name##_body); \
    static void suite##_##name##_body()

#define QC_CHECK(cond)                                                     \
    do {                                                                   \
        if (!(cond)) ::qtest::report_failure(__FILE__, __LINE__,           \
                                             std::string("CHECK failed: ") + #cond); \
    } while (0)

#define QC_CHECK_NEAR(actual, expected, tol)                                  \
    do {                                                                      \
        double a_ = (actual), e_ = (expected), t_ = (tol);                    \
        if (!(std::fabs(a_ - e_) <= t_))                                      \
            ::qtest::report_failure(__FILE__, __LINE__,                       \
                std::string("NEAR failed: |") + #actual + " - " + #expected + \
                "| = " + std::to_string(std::fabs(a_ - e_)) + " > " +         \
                std::to_string(t_) + " (actual=" + std::to_string(a_) +      \
                ", expected=" + std::to_string(e_) + ")");                   \
    } while (0)

#define QC_CHECK_THROWS(expr)                                \
    do {                                                     \
        bool threw_ = false;                                 \
        try { (void)(expr); } catch (...) { threw_ = true; } \
        if (!threw_) ::qtest::report_failure(__FILE__, __LINE__, \
            std::string("expected throw: ") + #expr);        \
    } while (0)

#define QC_SKIP(reason) throw ::qtest::SkipTest(reason)
