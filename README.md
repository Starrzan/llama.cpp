# llama.cpp — Starrzan Fork

**Custom fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) optimized for Intel Arc (SYCL) with TurboQuant KV cache compression.**

## What's different from upstream

### TurboQuant KV Cache Compression (Phase 1)
Implements [TurboQuant+](https://github.com/TheTom/turboquant_plus) (ICLR 2026) for memory-efficient long-context inference.

| Type | Bits/value | Compression | Quality (PPL vs q8_0) |
|------|-----------|-------------|----------------------|
| turbo4 | 4.25 | 3.8x | +0.23% ✅ |
| turbo3 | 3.125 | 5.12x | +1.06% |
| turbo2 | 2.5 | 6.4x | +6.48% (use with Boundary V) |

**Phase 1 = CPU fallback on SYCL**. K/V dequantization runs on CPU, attention on GPU. Functional but not optimal for decode speed. Full SYCL kernel port deferred.

Safe config: `-ctk q8_0 -ctv turbo4 -fa on` (asymmetric, minimal quality loss)

See `lobo-vault/lobo-server/dev/turboquant-research.md` for full analysis.

### Intel Arc SYCL patches
- IntelLLVM (icpx/icx) compiler support
- MKL SYCL BLAS linking fixes for oneAPI 2026.x
- `GGML_SYCL_HOST_MEM_FALLBACK=ON` for production stability
- `GGML_SYCL_DNN=ON` for oneDNN accelerated ops
- dpct helper.hpp patches for Xe driver compatibility

## Target hardware
- Intel Arc 140T (Arrow Lake-H iGPU, SYCL backend)
- Intel Arc Pro B50/B70 via OCuLink (planned)

## Build

```bash
source /opt/intel/oneapi/setvars.sh --force

cmake -Bbuild \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=OFF \
  -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_HOST_MEM_FALLBACK=ON \
  -DGGML_SYCL_STMT=ON \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO=ON \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/2026.0 \
  -DBUILD_SHARED_LIBS=ON

cmake --build build -j4 --target llama-server
```

Must be started with `source /opt/intel/oneapi/setvars.sh --force`.

## Usage with TurboQuant

```bash
./build/bin/llama-server \
  -m ~/models/Qwen3.5-9b-Sushi-Coder-RL-Q4_K_M.gguf \
  -ctk q8_0 -ctv turbo4 -fa on \
  -ngl 99 -c 32768
```

## Documentation
- `lobo-vault/lobo-server/dev/turboquant-research.md` — full TurboQuant analysis
- `lobo-vault/lobo-server/dev/turboquant-sycl-analysis.md` — SYCL gap analysis
- `lobo-vault/lobo-server/dev/turboquant-implementation-plan.md` — implementation guide
- `lobo-vault/research/gpu-research/` — GPU hardware research

## License
MIT (same as upstream llama.cpp)
