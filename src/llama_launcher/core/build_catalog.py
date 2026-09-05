"""CMake build-option catalog for the Build tab.

Setting.flag holds the bare CMake variable (no -D); the renderer prepends -D.
Engine gate: "any" = option exists in BOTH repos; "llama.cpp"/"ik_llama.cpp" =
only that repo has it. The inventory follows both repos' root + ggml
CMakeLists.txt.

Deviations from a literal grep of those files:
- CMAKE_CUDA_ARCHITECTURES is a native CMake variable (not project-defined via
  option()/set() CACHE), so it never appears in either repo's CMakeLists.txt;
  kept with engine "any" and an empty default (CMake/nvcc auto-detects).
- GGML_SCHED_MAX_COPIES exists in both repos with different literal defaults
  (mainline "4", ik "1"); modeled as two engine-gated Settings (sched-max-copies
  for llama.cpp, sched-max-copies-ik for ik_llama.cpp) sharing the same flag,
  each defaulting to its own repo's literal, rather than one "any"-gated entry
  that would show an ik user mainline's default.
- GGML_MAX_CONTEXTS (ik only) has an empty-string CACHE default that means
  "use the compiled-in default of 64"; catalog uses the int sentinel 0 for
  that, matching the 0 = auto/default convention used elsewhere in this app.
- GGML_CUDA_FUSION (ik only) is a CACHE STRING with literal default "1" and
  help text "enable/disable fusion"; modeled as type "int" (0/1), since the
  source shows a plain toggle value.

EXCLUDED ON PURPOSE: warnings/sanitizers/gprof, install-dir cache vars,
WASM/Metal/macOS/Android/Hexagon/OpenVINO/virtGPU/zDNN/ET targets, Vulkan
debug+validate+shader-debug options, RISC-V/LoongArch/s390x ISA options, MUSA
sub-options, tests/examples toggles beyond the LLAMA_BUILD_* set carried below.
"""

from .settings_catalog import Setting

REPO_URL = {
    "llama.cpp": "https://github.com/ggml-org/llama.cpp",
    "ik_llama.cpp": "https://github.com/ikawrakow/ik_llama.cpp",
}
DEFAULT_BRANCH = {"llama.cpp": "master", "ik_llama.cpp": "main"}
ENGINE_SHORT = {"llama.cpp": "llama", "ik_llama.cpp": "ik"}

_ALL = [
    # Backend
    Setting(
        "cuda",
        "GGML_CUDA",
        "bool",
        False,
        "Backend",
        (),
        tooltip="Build the CUDA backend for NVIDIA GPUs. Required for GPU "
        "offload on NVIDIA hardware; off by default (CPU-only build).",
    ),
    Setting(
        "cuda-architectures",
        "CMAKE_CUDA_ARCHITECTURES",
        "string",
        "",
        "Backend",
        (),
        tooltip="Semicolon-separated list of CUDA compute-capability targets to "
        "compile kernels for, e.g. '86;89;120'. Narrowing this to just "
        "your GPU(s) speeds up the CUDA build a lot; empty lets CMake/nvcc "
        "auto-detect. A native CMake variable, not a project option.",
    ),
    Setting(
        "vulkan",
        "GGML_VULKAN",
        "bool",
        False,
        "Backend",
        (),
        tooltip="Build the Vulkan backend, usable on NVIDIA/AMD/Intel GPUs "
        "without vendor-specific SDKs. Off by default.",
    ),
    Setting(
        "hip",
        "GGML_HIP",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build the HIP backend for AMD GPUs via ROCm. Mainline's "
        "current AMD backend name (ik uses the older GGML_HIPBLAS "
        "toggle below). Off by default.",
    ),
    Setting(
        "hipblas",
        "GGML_HIPBLAS",
        "bool",
        False,
        "Backend",
        (),
        engine="ik_llama.cpp",
        tooltip="Build the hipBLAS backend for AMD GPUs via ROCm (ik_llama.cpp's "
        "name for AMD GPU support; mainline's equivalent is GGML_HIP "
        "above). Off by default.",
    ),
    Setting(
        "blas",
        "GGML_BLAS",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Use a BLAS library to speed up CPU prompt processing. Off by "
        "default outside macOS (where Apple's Accelerate is used "
        "instead); pair with blas-vendor to pick the implementation.",
    ),
    Setting(
        "blas-vendor",
        "GGML_BLAS_VENDOR",
        "string",
        "Generic",
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Which BLAS implementation to link when blas is on, e.g. "
        "'OpenBLAS', 'Intel10_64lp', 'Generic'. 'Generic' (the default "
        "off Apple) picks whatever BLAS CMake finds on the system.",
    ),
    Setting(
        "opencl",
        "GGML_OPENCL",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build the OpenCL backend, mainly used for Adreno GPUs on "
        "Android/Snapdragon devices. Off by default.",
    ),
    Setting(
        "sycl",
        "GGML_SYCL",
        "bool",
        False,
        "Backend",
        (),
        tooltip="Build the SYCL backend for Intel GPUs via oneAPI. Off by default.",
    ),
    Setting(
        "sycl-f16",
        "GGML_SYCL_F16",
        "bool",
        False,
        "Backend",
        (),
        tooltip="Use 16-bit floats for SYCL calculations, trading a little "
        "precision for speed on Intel GPUs. Off by default.",
    ),
    Setting(
        "sycl-target",
        "GGML_SYCL_TARGET",
        "string",
        "INTEL",
        "Backend",
        (),
        tooltip="Target device family for the SYCL backend, e.g. 'INTEL', "
        "'NVIDIA', 'AMD'. Defaults to 'INTEL'.",
    ),
    Setting(
        "musa",
        "GGML_MUSA",
        "bool",
        False,
        "Backend",
        (),
        tooltip="Build the MUSA backend for Moore Threads GPUs. Off by default.",
    ),
    Setting(
        "webgpu",
        "GGML_WEBGPU",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build the experimental WebGPU backend. Off by default.",
    ),
    Setting(
        "cpu-backend",
        "GGML_CPU",
        "bool",
        True,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build the CPU backend. On by default; only turn off for a "
        "GPU-only build (e.g. an RPC server node with no local "
        "inference).",
    ),
    Setting(
        "backend-dl",
        "GGML_BACKEND_DL",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build backends as dynamic libraries loaded at runtime instead "
        "of linked statically, so one binary can ship multiple backends "
        "and pick at load time. Requires BUILD_SHARED_LIBS. Off by "
        "default.",
    ),
    Setting(
        "cpu-all-variants",
        "GGML_CPU_ALL_VARIANTS",
        "bool",
        False,
        "Backend",
        (),
        engine="llama.cpp",
        tooltip="Build every CPU instruction-set variant (SSE4.2 through "
        "AVX512) into one binary that auto-selects the best at "
        "runtime, instead of one binary tuned for the build machine. "
        "Requires backend-dl. Off by default; slower to build, more "
        "portable to distribute.",
    ),
    # CUDA tuning
    Setting(
        "cuda-fa",
        "GGML_CUDA_FA",
        "bool",
        True,
        "CUDA tuning",
        (),
        engine="llama.cpp",
        tooltip="Compile the FlashAttention CUDA kernels. On by default; "
        "needed for --flash-attn on CUDA.",
    ),
    Setting(
        "cuda-fa-all-quants",
        "GGML_CUDA_FA_ALL_QUANTS",
        "bool",
        False,
        "CUDA tuning",
        (),
        tooltip="Compile FlashAttention kernels for every KV-cache quant type "
        "instead of just the common ones. Off by default; a much "
        "longer CUDA build in exchange for -ctk/-ctv flexibility.",
    ),
    Setting(
        "cuda-force-mmq",
        "GGML_CUDA_FORCE_MMQ",
        "bool",
        False,
        "CUDA tuning",
        (),
        tooltip="Always use the mmq (matrix-multiply-quantized) CUDA kernels "
        "instead of cuBLAS, even where cuBLAS would normally be chosen. "
        "Off by default.",
    ),
    Setting(
        "cuda-force-cublas",
        "GGML_CUDA_FORCE_CUBLAS",
        "bool",
        False,
        "CUDA tuning",
        (),
        tooltip="Always use cuBLAS instead of the mmq kernels, the opposite of "
        "cuda-force-mmq. Off by default.",
    ),
    Setting(
        "cuda-graphs",
        "GGML_CUDA_GRAPHS",
        "bool",
        True,
        "CUDA tuning",
        (),
        engine="llama.cpp",
        tooltip="Use CUDA graphs to reduce per-step kernel-launch overhead. On "
        "by default when built as part of llama.cpp; can help "
        "throughput on newer GPUs.",
    ),
    Setting(
        "cuda-no-vmm",
        "GGML_CUDA_NO_VMM",
        "bool",
        False,
        "CUDA tuning",
        (),
        tooltip="Disable CUDA virtual memory management even if the driver "
        "supports it. Off by default (VMM used when available).",
    ),
    Setting(
        "cuda-no-peer-copy",
        "GGML_CUDA_NO_PEER_COPY",
        "bool",
        False,
        "CUDA tuning",
        (),
        tooltip="Disable direct peer-to-peer copies between GPUs, falling back "
        "to host-staged transfers. Off by default; useful as a "
        "workaround when P2P is flaky on a given multi-GPU setup.",
    ),
    Setting(
        "cuda-peer-max-batch-size",
        "GGML_CUDA_PEER_MAX_BATCH_SIZE",
        "int",
        128,
        "CUDA tuning",
        (),
        0,
        4096,
        1,
        engine="ik_llama.cpp",
        tooltip="Batch-size threshold for using peer-to-peer GPU copies "
        "(ik_llama.cpp only; its CMake help text reads 'min batch size for "
        "GPU offload', which does not match the variable name, so treat this "
        "as a tuning knob rather than a precise cutoff). Default 128.",
    ),
    Setting(
        "cuda-compression-mode",
        "GGML_CUDA_COMPRESSION_MODE",
        "enum",
        "size",
        "CUDA tuning",
        (),
        enum=("none", "speed", "balance", "size"),
        tooltip="CUDA link binary compression mode (requires CUDA 12.8+). "
        "'size' (the default) favors a smaller binary; 'speed' favors "
        "faster linking.",
    ),
    Setting(
        "cuda-nccl",
        "GGML_CUDA_NCCL",
        "bool",
        True,
        "CUDA tuning",
        (),
        engine="llama.cpp",
        tooltip="Use NVIDIA's Collective Communications Library for multi-GPU "
        "reductions. On by default.",
    ),
    Setting(
        "cuda-f16",
        "GGML_CUDA_F16",
        "bool",
        False,
        "CUDA tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Use 16-bit floats for some CUDA calculations, trading a "
        "little precision for speed. Off by default.",
    ),
    Setting(
        "cuda-iqk-force-bf16",
        "GGML_CUDA_IQK_FORCE_BF16",
        "bool",
        False,
        "CUDA tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Use bf16 cuBLAS when no matching MMQ kernel is available for "
        "an iqk (i-quant) op, instead of falling back to fp16/fp32. "
        "Off by default.",
    ),
    Setting(
        "cuda-use-graphs",
        "GGML_CUDA_USE_GRAPHS",
        "bool",
        True,
        "CUDA tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="ik_llama.cpp's equivalent of cuda-graphs: use CUDA graphs to "
        "cut per-step launch overhead. On by default.",
    ),
    Setting(
        "cuda-fusion",
        "GGML_CUDA_FUSION",
        "int",
        1,
        "CUDA tuning",
        (),
        0,
        1,
        1,
        engine="ik_llama.cpp",
        tooltip="Enable (1, the default) or disable (0) ik_llama.cpp's CUDA "
        "kernel-fusion optimizations that combine adjacent ops into "
        "one kernel launch.",
    ),
    Setting(
        "cuda-min-batch-offload",
        "GGML_CUDA_MIN_BATCH_OFFLOAD",
        "int",
        32,
        "CUDA tuning",
        (),
        1,
        4096,
        1,
        engine="ik_llama.cpp",
        tooltip="Minimum batch size before a GPU offload is worth doing; "
        "smaller batches stay on CPU. Default 32.",
    ),
    Setting(
        "cuda-dmmv-x",
        "GGML_CUDA_DMMV_X",
        "int",
        32,
        "CUDA tuning",
        (),
        1,
        256,
        1,
        engine="ik_llama.cpp",
        tooltip="X stride used by the dmmv (dequantize-matmul-vec) CUDA "
        "kernels; a low-level kernel-tuning knob. Default 32.",
    ),
    Setting(
        "cuda-mmv-y",
        "GGML_CUDA_MMV_Y",
        "int",
        1,
        "CUDA tuning",
        (),
        1,
        16,
        1,
        engine="ik_llama.cpp",
        tooltip="Y block size used by the mmv (matmul-vec) CUDA kernels; a "
        "low-level kernel-tuning knob. Default 1.",
    ),
    Setting(
        "cuda-kquants-iter",
        "GGML_CUDA_KQUANTS_ITER",
        "int",
        2,
        "CUDA tuning",
        (),
        1,
        8,
        1,
        engine="ik_llama.cpp",
        tooltip="Iterations per thread-block for the Q2_K/Q6_K CUDA dequant "
        "kernels; a low-level kernel-tuning knob. Default 2.",
    ),
    Setting(
        "cuda-force-dmmv",
        "GGML_CUDA_FORCE_DMMV",
        "bool",
        False,
        "CUDA tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Always use the dmmv CUDA kernels instead of mmvq. Off by default.",
    ),
    # CPU & ISA
    Setting(
        "native-opt",
        "GGML_NATIVE",
        "bool",
        True,
        "CPU & ISA",
        (),
        tooltip="Optimize the build for the machine doing the compiling "
        "(-march=native), auto-enabling whatever instruction sets it "
        "supports. On by default; turn off when building a binary "
        "you'll run on a different, less-capable CPU.",
    ),
    Setting(
        "lto",
        "GGML_LTO",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Enable link-time optimization. Off by default; can improve "
        "runtime performance a little at the cost of a slower, "
        "more memory-hungry build.",
    ),
    Setting(
        "openmp",
        "GGML_OPENMP",
        "bool",
        True,
        "CPU & ISA",
        (),
        tooltip="Use OpenMP for CPU multithreading. On by default.",
    ),
    Setting(
        "llamafile",
        "GGML_LLAMAFILE",
        "bool",
        True,
        "CPU & ISA",
        (),
        tooltip="Use llamafile's optimized CPU matrix-multiplication "
        "kernels (sgemm). On by default; usually faster CPU prompt "
        "processing.",
    ),
    Setting(
        "cpu-repack",
        "GGML_CPU_REPACK",
        "bool",
        True,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Repack Q4_0 weights into runtime-optimized Q4_X_X layouts "
        "for faster CPU matmuls. On by default. (ik_llama.cpp achieves "
        "the same effect at runtime via -rtr instead of a build "
        "option.)",
    ),
    Setting(
        "cpu-hbm",
        "GGML_CPU_HBM",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Use memkind to allocate CPU tensors in high-bandwidth memory "
        "(HBM), for systems that have it (e.g. Xeon Phi/Knights "
        "Landing). Off by default.",
    ),
    Setting(
        "cpu-kleidiai",
        "GGML_CPU_KLEIDIAI",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Use Arm KleidiAI optimized kernels where applicable (Arm "
        "CPUs only). Off by default.",
    ),
    Setting(
        "cpu-arm-arch",
        "GGML_CPU_ARM_ARCH",
        "string",
        "",
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Override the ARM CPU architecture string passed to the "
        "compiler (e.g. 'armv8.2-a+dotprod+fp16'). Empty lets CMake "
        "pick automatically.",
    ),
    Setting(
        "avx",
        "GGML_AVX",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX support. Defaults off when native-opt is on "
        "(native detection already covers it); enable explicitly when "
        "native-opt is off and you want AVX targeted directly.",
    ),
    Setting(
        "avx2",
        "GGML_AVX2",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX2 support. Defaults off when native-opt is on "
        "(native detection already covers it); enable explicitly when "
        "native-opt is off.",
    ),
    Setting(
        "avx512",
        "GGML_AVX512",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX512F support. Off by default even with "
        "native-opt on, since not all 'native' CPUs have AVX512.",
    ),
    Setting(
        "avx512-vbmi",
        "GGML_AVX512_VBMI",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX512-VBMI support. Off by default.",
    ),
    Setting(
        "avx512-vnni",
        "GGML_AVX512_VNNI",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX512-VNNI support (faster int8 dot products). "
        "Off by default.",
    ),
    Setting(
        "avx512-bf16",
        "GGML_AVX512_BF16",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with AVX512-BF16 support. Off by default.",
    ),
    Setting(
        "avx-vnni",
        "GGML_AVX_VNNI",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with AVX-VNNI support (mainline's spelling; ik's "
        "equivalent is avxvnni below). Off by default.",
    ),
    Setting(
        "avxvnni",
        "GGML_AVXVNNI",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="ik_llama.cpp",
        tooltip="Compile with AVX-VNNI support (ik_llama.cpp's spelling; "
        "mainline's equivalent is avx-vnni above). Off by default.",
    ),
    Setting(
        "fma",
        "GGML_FMA",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with FMA (fused multiply-add) support. Defaults off "
        "when native-opt is on (native detection already covers it); "
        "enable explicitly when native-opt is off.",
    ),
    Setting(
        "f16c",
        "GGML_F16C",
        "bool",
        False,
        "CPU & ISA",
        (),
        tooltip="Compile with F16C support (hardware fp16<->fp32 conversion). "
        "Defaults off when native-opt is on; enable explicitly when "
        "native-opt is off.",
    ),
    Setting(
        "bmi2",
        "GGML_BMI2",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with BMI2 support. Defaults off when native-opt is "
        "on; enable explicitly when native-opt is off.",
    ),
    Setting(
        "sse42",
        "GGML_SSE42",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with SSE 4.2 support. Defaults off when native-opt is "
        "on; enable explicitly when native-opt is off (for very old "
        "target CPUs).",
    ),
    Setting(
        "amx-tile",
        "GGML_AMX_TILE",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with Intel AMX-TILE support (matrix tile registers "
        "on recent Xeon/Core CPUs). Off by default.",
    ),
    Setting(
        "amx-int8",
        "GGML_AMX_INT8",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with Intel AMX-INT8 support. Off by default.",
    ),
    Setting(
        "amx-bf16",
        "GGML_AMX_BF16",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="llama.cpp",
        tooltip="Compile with Intel AMX-BF16 support. Off by default.",
    ),
    Setting(
        "sve",
        "GGML_SVE",
        "bool",
        False,
        "CPU & ISA",
        (),
        engine="ik_llama.cpp",
        tooltip="Compile with Arm SVE (Scalable Vector Extension) support. "
        "Off by default.",
    ),
    # ik kernels
    Setting(
        "iqk-mul-mat",
        "GGML_IQK_MUL_MAT",
        "bool",
        True,
        "ik kernels",
        (),
        engine="ik_llama.cpp",
        tooltip="Use ik_llama.cpp's optimized matrix-multiplication kernels "
        "for i-quants. On by default; this is much of what makes ik "
        "faster than mainline on quantized models.",
    ),
    Setting(
        "iqk-flash-attention",
        "GGML_IQK_FLASH_ATTENTION",
        "bool",
        True,
        "ik kernels",
        (),
        engine="ik_llama.cpp",
        tooltip="Compile ik_llama.cpp's CPU FlashAttention kernels. On by default.",
    ),
    Setting(
        "iqk-fa-all-quants",
        "GGML_IQK_FA_ALL_QUANTS",
        "bool",
        True,
        "ik kernels",
        (),
        engine="ik_llama.cpp",
        tooltip="Compile the IQK FlashAttention kernels for every quant type "
        "instead of just the common ones. On by default; a longer "
        "build in exchange for -ctk/-ctv flexibility.",
    ),
    Setting(
        "expert-chunking",
        "GGML_EXPERT_CHUNKING",
        "bool",
        True,
        "ik kernels",
        (),
        engine="ik_llama.cpp",
        tooltip="Process MoE experts in chunks to reduce peak memory during "
        "inference. On by default.",
    ),
    # Vulkan tuning
    Setting(
        "vulkan-no-coopmat",
        "GGML_VULKAN_NO_COOPMAT",
        "bool",
        False,
        "Vulkan tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Don't use the Vulkan cooperative-matrix extension even if "
        "the driver supports it. Off by default; a compatibility "
        "fallback for buggy coopmat drivers.",
    ),
    Setting(
        "vulkan-no-coopmat2",
        "GGML_VULKAN_NO_COOPMAT2",
        "bool",
        False,
        "Vulkan tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Don't use the Vulkan cooperative-matrix2 extension even if "
        "supported. Off by default.",
    ),
    Setting(
        "vulkan-no-int-dot",
        "GGML_VULKAN_NO_INT_DOT",
        "bool",
        False,
        "Vulkan tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Don't use Vulkan's integer dot-product extension even if "
        "supported. Off by default.",
    ),
    Setting(
        "vulkan-no-bf16",
        "GGML_VULKAN_NO_BF16",
        "bool",
        False,
        "Vulkan tuning",
        (),
        engine="ik_llama.cpp",
        tooltip="Don't use Vulkan bf16 support even if the driver has it. "
        "Off by default.",
    ),
    # Features & networking
    Setting(
        "rpc",
        "GGML_RPC",
        "bool",
        False,
        "Features & networking",
        (),
        tooltip="Build RPC support so this build can act as (or connect to) a "
        "remote llama.cpp worker node for distributed inference. Off "
        "by default.",
    ),
    Setting(
        "curl",
        "LLAMA_CURL",
        "bool",
        False,
        "Features & networking",
        (),
        engine="ik_llama.cpp",
        tooltip="Use libcurl so the server/CLI can download a model directly "
        "from a URL (e.g. -hf). Off by default. Mainline has "
        "deprecated this flag with no replacement in the CMake files "
        "checked here, so it's offered on ik_llama.cpp only.",
    ),
    Setting(
        "ggml-curl",
        "GGML_CURL",
        "bool",
        False,
        "Features & networking",
        (),
        engine="ik_llama.cpp",
        tooltip="ggml-level counterpart of curl above: use libcurl to "
        "download a model from a URL. Off by default.",
    ),
    Setting(
        "openssl",
        "LLAMA_OPENSSL",
        "bool",
        True,
        "Features & networking",
        (),
        engine="llama.cpp",
        tooltip="Use OpenSSL so the server can support HTTPS. On by default.",
    ),
    Setting(
        "llguidance",
        "LLAMA_LLGUIDANCE",
        "bool",
        False,
        "Features & networking",
        (),
        tooltip="Include the LLGuidance library for structured/grammar-"
        "constrained output. Off by default; needs the LLGuidance "
        "dependency available at build time.",
    ),
    Setting(
        "build-server",
        "LLAMA_BUILD_SERVER",
        "bool",
        True,
        "Features & networking",
        (),
        tooltip="Build the llama-server example (what this app launches). On "
        "by default for a standalone build; turn off only for a "
        "library-only build.",
    ),
    Setting(
        "build-mtmd",
        "LLAMA_BUILD_MTMD",
        "bool",
        False,
        "Features & networking",
        (),
        engine="llama.cpp",
        tooltip="Build the mtmd (multimodal) library standalone, without "
        "pulling in the full tools tree. Off by default; not needed "
        "for a normal server build, which already includes mtmd.",
    ),
    Setting(
        "build-tools",
        "LLAMA_BUILD_TOOLS",
        "bool",
        True,
        "Features & networking",
        (),
        engine="llama.cpp",
        tooltip="Build llama.cpp's CLI tools (quantize, convert helpers, "
        "etc.), not just the server library. On by default for a "
        "standalone build.",
    ),
    Setting(
        "build-examples",
        "LLAMA_BUILD_EXAMPLES",
        "bool",
        True,
        "Features & networking",
        (),
        tooltip="Build the example programs. On by default for a standalone build.",
    ),
    Setting(
        "build-tests",
        "LLAMA_BUILD_TESTS",
        "bool",
        True,
        "Features & networking",
        (),
        tooltip="Build the test suite. On by default for a standalone build; "
        "turn off for a slightly faster build if you won't run the "
        "upstream tests.",
    ),
    # Build type & misc
    Setting(
        "build-type",
        "CMAKE_BUILD_TYPE",
        "enum",
        "Release",
        "Build type & misc",
        (),
        enum=("Release", "RelWithDebInfo", "Debug", "MinSizeRel"),
        tooltip="CMake's optimization/debug-info level. 'Release' (the "
        "default) is fully optimized with no debug info; "
        "'RelWithDebInfo' keeps optimizations and adds symbols; "
        "'Debug' disables optimization for debugging; 'MinSizeRel' "
        "optimizes for binary size.",
    ),
    Setting(
        "static",
        "GGML_STATIC",
        "bool",
        False,
        "Build type & misc",
        (),
        tooltip="Statically link the runtime libraries into the binary. Off "
        "by default; on makes the binary more portable at the cost of "
        "a larger file.",
    ),
    Setting(
        "ccache",
        "GGML_CCACHE",
        "bool",
        True,
        "Build type & misc",
        (),
        tooltip="Use ccache to speed up rebuilds when it's available on the "
        "system. On by default.",
    ),
    Setting(
        "sched-max-copies",
        "GGML_SCHED_MAX_COPIES",
        "int",
        4,
        "Build type & misc",
        (),
        1,
        16,
        1,
        engine="llama.cpp",
        tooltip="Max input-buffer copies ggml's scheduler keeps for pipeline "
        "parallelism across compute streams. Mainline defaults to 4.",
    ),
    Setting(
        "sched-max-copies-ik",
        "GGML_SCHED_MAX_COPIES",
        "int",
        1,
        "Build type & misc",
        (),
        1,
        16,
        1,
        engine="ik_llama.cpp",
        tooltip="Max input-buffer copies ggml's scheduler keeps for pipeline "
        "parallelism across compute streams. ik_llama.cpp defaults to "
        "1 (its scheduler doesn't pipeline the same way as mainline, "
        "whose default is 4).",
    ),
    Setting(
        "max-contexts",
        "GGML_MAX_CONTEXTS",
        "int",
        0,
        "Build type & misc",
        (),
        0,
        4096,
        1,
        engine="ik_llama.cpp",
        tooltip="Override ik_llama.cpp's compiled-in limit on simultaneous "
        "model contexts (used by --parallel/multi-slot serving). 0 "
        "leaves the CMake cache value empty, which falls back to the "
        "code's built-in default of 64.",
    ),
]

BUILD_CATALOG: dict = {s.key: s for s in _ALL}
