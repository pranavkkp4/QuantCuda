// QuantCUDA Risk Engine - command-line interface.
#pragma once

namespace quantcuda {

// Parses argv and dispatches to price / risk / benchmark / validate / greeks /
// versions subcommands. Returns the process exit code.
int run_cli(int argc, char** argv);

}  // namespace quantcuda
