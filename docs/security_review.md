# OWASP Top 10 security review report

## Review summary

- **Scope:** all first-party code — `src/` (core, cuda, benchmark, cli),
  `include/quantcuda/`, `tests/`, `scripts/` — plus the one vendored
  third-party component (`nlohmann/json` v3.11.3, pinned).
- **Date:** 2026-07-24
- **Application profile:** offline native CLI. No network listeners, no
  authentication, no database, no web surface, no multi-tenant data. Most
  web-centric OWASP items are therefore not applicable; they are listed and
  marked, per the full-coverage rule.
- **Risk summary:** RED High 0 | YELLOW Medium 0 | GREEN Low 1 | PASS 9

## Findings (sorted by severity, descending)

### REMEDIATED — A05 — Resource exhaustion controls
- **Location:** JSON loader, CLI parser, CPU/GPU public entry points.
- **Fix (applied):** strict integer parsing; 100,000,000-path and
  10,000,000-scenario ceilings; checked paths × steps × instruments and
  scenarios × assets budgets; bounded configuration-file ingestion; 256-asset
  cap; `time_steps` ≤ 100,000; CUDA block size checked against both [32, 1024]
  and the selected device. Allocation failures still surface as actionable
  exceptions rather than undefined behavior.

### GREEN Low — A06 — Third-party components
- **Location:** `include/third_party/nlohmann/json.hpp` (v3.11.3).
- **Description:** single vendored dependency, pinned. CUDA-side Thrust/CUB
  and cuRAND ship with the CUDA toolkit (operator-managed).
- **Fix (policy):** versions pinned; no dynamic fetching at build or runtime;
  update nlohmann/json on patch releases.

### PASS A01 — Broken Access Control — not applicable (no access control surface; local single-user CLI)

### PASS A02 — Cryptographic Failures — no secrets, credentials, or sensitive data anywhere in the codebase; RNG is Monte-Carlo-grade Philox (documented: not for cryptographic use)

### PASS A03 — Injection — verified by scan: no `system()`/`popen`/shell execution anywhere in C++ or Python; no SQL; no template rendering; file paths are used only as file paths (ifstream/ofstream), never as commands

### PASS A04 — Insecure Design — workload caps and strict validation bound all resource usage; error paths carry no internal state beyond file:line diagnostics

### PASS A05 — Security Misconfiguration — release builds define NDEBUG; errors go to stderr without stack traces; no debug endpoints

### PASS A07 — Authentication Failures — not applicable (no authentication)

### PASS A08 — Software and Data Integrity Failures — input is JSON parsed by a memory-safe parser with strict schema validation (types, ranges, matrix shape/symmetry); no native deserialization of untrusted objects; Python scripts use `json` (safe), never `pickle`/`yaml.load`

### PASS A09 — Logging and Monitoring Failures — not applicable (no security events exist to log; errors are reported to stderr without sensitive data)

### PASS A10 — SSRF — not applicable (the application makes zero network requests; verified by scan)

## Input-validation inventory (defense in depth, verified by tests TC-JSN-*)

| Input | Validation |
|---|---|
| paths / scenarios | integer, > 0, ≤ 100,000,000 / 10,000,000; checked aggregate budgets |
| time_steps | integer, [1, 100000] |
| spot, strike, barrier, maturity | numeric, ≥ 0 |
| volatility | numeric, ≥ 0 |
| correlation | d×d, symmetric, entries in [−1,1], diagonal 1, positive-definite (Cholesky) |
| confidence levels | (0, 1) |
| horizon_years / horizon-days | (0, 1] / [1, 252] |
| instruments / assets | ≤ 100,000 / ≤ 256 |
| block-size | [32, 1024] |
| backend / precision / gpu-version | closed enumerations |
| configuration file | bounded size; strict object shapes and key whitelist |

## Remediation priority

1. (Done) Strict CLI/JSON numeric validation with overflow rejection.
2. (Done) Clean, non-crashing failure on CUDA OOM and missing GPU.
3. (Done) Zero compiler warnings under `-Wall -Wextra`.
4. Ongoing: keep the single vendored dependency pinned and reviewed.
