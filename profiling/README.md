# Nsight profiling workflow

Run these commands from the repository root after building a release CUDA
binary. The executable path is `./build/quantcuda` for a single-config POSIX
build, `.\build\quantcuda.exe` for Ninja on Windows, and
`.\build\Release\quantcuda.exe` for Visual Studio.

One directory per major kernel, per the project specification:

```text
profiling/
├── european_naive/        # V0 split kernels
├── european_fused/        # V2/V3 fused kernel
├── asian/                 # path kernel at 12/52/252 steps
├── barrier_branch/        # early-exit barrier kernel
├── barrier_mask/          # boolean-mask barrier kernel
├── reduction_atomic/      # per-thread atomicAdd
├── reduction_shuffle/     # warp-shuffle reduction
├── correlation_global/    # block kernel, L in global memory
└── correlation_shared/    # warp kernel, L in __constant__ memory
```

## Capturing a report (Nsight Compute)

```bash
ncu --set full \
    --kernel-name regex:european \
    --launch-count 3 \
    --export profiling/european_fused/report \
    ./build/quantcuda price --portfolio configs/european_call.json \
      --backend cuda --paths 10000000 --gpu-version v2
```

PowerShell / Visual Studio build:

```powershell
ncu --set full --kernel-name "regex:european" --launch-count 3 --export profiling/european_fused/report .\build\Release\quantcuda.exe price --portfolio configs\european_call.json --backend cuda --paths 10000000 --gpu-version v2
```

## Capturing a report (Nsight Systems, end-to-end timeline)

```bash
nsys profile -o profiling/reports/european_timeline \
    ./build/quantcuda price --portfolio configs/european_call.json \
      --backend cuda --paths 10000000 --gpu-version v6
```

PowerShell / Visual Studio build:

```powershell
nsys profile -o profiling/reports/european_timeline .\build\Release\quantcuda.exe price --portfolio configs\european_call.json --backend cuda --paths 10000000 --gpu-version v6
```

## Metrics to record for every kernel (per the specification)

| Metric | Nsight Compute section |
|---|---|
| Duration | GPU Speed Of Light |
| Achieved / theoretical occupancy | Occupancy |
| Registers per thread | Launch Statistics |
| Occupancy limited by registers / shared memory | Occupancy |
| DRAM throughput | Memory Workload Analysis |
| L1/L2 hit rates | Memory Workload Analysis |
| Global load/store efficiency | Memory Workload Analysis |
| Branch efficiency | Scheduler Statistics / Source Counters |
| Warp stall reasons | Warp State Statistics |
| Compute vs memory utilization | GPU Speed Of Light |
| Roofline position | Roofline Charts |

Every optimization entry in `docs/performance_report.md` follows the
Hypothesis / Change / Workload / Before / After / Conclusion format.
