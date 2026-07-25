// QuantCUDA Risk Engine - native C++ command-line application.
#include <cstdio>
#include <exception>

#include "quantcuda/cli.hpp"

int main(int argc, char** argv) {
    try {
        return quantcuda::run_cli(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "quantcuda: error: %s\n", e.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "quantcuda: unknown error\n");
        return 2;
    }
}
