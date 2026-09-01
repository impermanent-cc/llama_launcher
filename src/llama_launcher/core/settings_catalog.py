from dataclasses import dataclass, field, replace

from .spec import DEFAULT_PORT


@dataclass(frozen=True)
class Setting:
    key: str
    flag: str
    type: str            # "bool" | "int" | "float" | "enum" | "string" | "int_or_token"
    default: object
    group: str
    aliases: tuple = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    enum: tuple = ()
    tokens: tuple = ()   # for int_or_token (e.g. ("auto", "all"))
    tooltip: str = ""
    danger: bool = False
    option_help: tuple = ()   # multiselect only: ((option, help_text), ...)
    suggestions: tuple = ()   # int only: preset values offered in an editable combo
    engine: str = "any"       # "any" | "ik_llama.cpp"  (which engine surfaces this flag)
                              # build_catalog additionally uses "llama.cpp" for
                              # mainline-only CMake options
    deprecated: bool = False  # upstream emits DEPRECATED warnings; kept for old images
    secret: bool = False      # mask the editor (password field) -- e.g. api-key


KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")

# ik_llama.cpp-only KV-cache quant types layered onto -ctk/-ctv when the engine is
# ik. Source-verified (common/common.cpp): only q6_0 and q8_KV are added beyond
# mainline in the default build. NOTE the capital "KV" in q8_KV; ik's parser is
# case-sensitive. The wider ik iq-quants need GGML_IQK_FA_ALL_QUANTS=ON images.
IK_EXTRA_KV_CACHE_TYPES = ("q6_0", "q8_KV")

# The shared spec-type enum uses mainline's spellings (common/speculative.cpp
# name map). ik's map (also common/speculative.cpp there) has no "draft-"
# prefix on the draft-model types, so these are renamed at emit time on an ik
# launch. The ngram-* names need no entry: ik normalizes '-' to '_' before its
# map lookup, which makes mainline's spellings land on the same keys.
IK_SPEC_TYPE_RENAMES = {
    "draft-simple": "draft",
    "draft-eagle3": "eagle3",
    "draft-dflash": "dflash",
    "draft-dspark": "dspark",
    "draft-mtp": "mtp",
}

# ik-only spec-type values, layered onto the shared enum like the KV-cache
# extras: offered by the UI only on the ik engine, dropped at emit time on a
# mainline launch (mainline's parser has no such type).
IK_EXTRA_SPEC_TYPES = ("suffix",)

_ALL = [
    # Model & Context
    Setting("ctx-size", "--ctx-size", "int", 0, "Model & Context", ("-c",), 0, 1048576, 1024,
            tooltip="Context window in tokens (prompt + generation). Bigger needs more "
                    "VRAM/RAM for the KV cache. 0 = use the model's trained maximum.",
            suggestions=(0, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)),
    Setting("n-predict", "--n-predict", "int", -1, "Model & Context", ("-n",), -2, 1048576, 1,
            tooltip="Maximum number of tokens to generate per response. -1 = unlimited "
                    "(until end-of-text or context fills); -2 = stop when context is full."),
    Setting("keep", "--keep", "int", 0, "Model & Context", (), -1, 1048576, 1,
            tooltip="When the context overflows, how many tokens from the start of the "
                    "prompt to always retain. -1 = keep the entire prompt; 0 = keep none."),
    Setting("swa-full", "--swa-full", "bool", False, "Model & Context", (),
            tooltip="Use a full-size sliding-window-attention (SWA) cache. Models with "
                    "hybrid local/global attention (e.g. Gemma) use SWA; this trades more "
                    "memory for fuller context reuse. Off by default."),
    Setting("context-shift", "--context-shift", "bool", False, "Model & Context", (),
            tooltip="Shift the context window to keep generating once it fills, instead of "
                    "stopping. Off by default."),

    # GPU & Memory
    Setting("n-gpu-layers", "--n-gpu-layers", "int_or_token", "auto", "GPU & Memory", ("-ngl",),
            0, 999, 1, tokens=("auto", "all"),
            tooltip="Number of model layers offloaded to the GPU. Higher = faster but more "
                    "VRAM. 'all' offloads everything; 'auto' lets llama.cpp choose; a big "
                    "number like 99 effectively means all."),
    Setting("n-cpu-moe", "--n-cpu-moe", "int", 0, "GPU & Memory", ("-ncmoe",), 0, 999, 1,
            tooltip="For Mixture-of-Experts models: keep the experts of the first N layers "
                    "on the CPU to fit a large model in limited VRAM. Higher N = less VRAM, "
                    "slower."),
    Setting("n-cpu-ffn", "--n-cpu-ffn", "int", 0, "GPU & Memory", ("-ncffn",), 0, 999, 1,
            tooltip="For dense (non-MoE) models: keep the FFN weights of the first N layers "
                    "on the CPU to fit a large model in limited VRAM. The dense counterpart "
                    "of --n-cpu-moe. Newer llama.cpp only."),
    Setting("cpu-moe", "--cpu-moe", "bool", False, "GPU & Memory", ("-cmoe",),
            tooltip="For Mixture-of-Experts models: keep ALL expert weights on the CPU, "
                    "leaving only attention/shared weights on the GPU. Saves the most VRAM "
                    "but is slower."),
    Setting("flash-attn", "--flash-attn", "enum", "auto", "GPU & Memory", ("-fa",),
            enum=("on", "off", "auto"),
            tooltip="FlashAttention: faster, lower-memory attention. 'auto' enables it when "
                    "supported; usually required for quantized KV cache (q8_0/q4_0)."),
    Setting("cache-type-k", "--cache-type-k", "enum", "f16", "GPU & Memory", ("-ctk",),
            enum=KV_CACHE_TYPES,
            tooltip="Quantization of the K KV-cache. Lower precision (e.g. q8_0, q4_0) saves "
                    "a lot of VRAM at a small quality cost; f16 (the default) is half precision, f32 is full."),
    Setting("cache-type-v", "--cache-type-v", "enum", "f16", "GPU & Memory", ("-ctv",),
            enum=KV_CACHE_TYPES,
            tooltip="Quantization of the V KV-cache. Lower precision (e.g. q8_0, q4_0) saves "
                    "a lot of VRAM at a small quality cost; f16 (the default) is half precision, f32 is full."),
    Setting("load-mode", "--load-mode", "enum", "auto", "GPU & Memory", ("-lm",),
            enum=("auto", "mmap", "none", "mlock", "mmap+mlock", "dio"),
            tooltip="How the model file is loaded (newer llama.cpp; supersedes "
                    "--no-mmap/--mlock). auto (the upstream default) memory-maps unless a "
                    "device does not support it; mmap forces memory-mapping; none loads "
                    "fully into RAM (slower, more RAM); mlock/mmap+mlock also lock it in "
                    "RAM so it's never swapped; dio uses DirectIO if available. Set to "
                    "anything other than auto and the legacy no-mmap/mlock flags below are "
                    "ignored. Needs an image new enough to know --load-mode; the auto "
                    "value itself needs images from 2026-08-11 or later."),
    # Upstream renamed this flag to --lazy-mode; --tensor-read-lazy is not
    # accepted by current builds (verified against b10711), so setting anything
    # other than the "auto" sentinel used to fail the launch outright. The KEY
    # stays "tensor-read-lazy" on purpose: it is internal, saved profiles are
    # written with it, and the form labels rows by flag, so renaming the key
    # would silently drop the value from every existing profile for no visible
    # gain.
    Setting("tensor-read-lazy", "--lazy-mode", "enum", "auto", "GPU & Memory", ("-lzm",),
            enum=("on", "auto", "off"),
            tooltip="On-demand reading of certain tensors, e.g. per-layer embeddings. "
                    "'on' reads their rows from disk on demand instead of keeping them "
                    "resident (requires mmap); 'auto' (default) does so only for tensors "
                    "larger than 4 GiB; 'off' keeps them resident. Newer llama.cpp only."),
    Setting("no-mmap", "--no-mmap", "bool", False, "GPU & Memory", (), deprecated=True,
            tooltip="Legacy (deprecated upstream in favor of --load-mode, but works on "
                    "all image versions). Disable memory-mapping of the model file, loading "
                    "it fully into RAM instead. Ignored when load-mode is set."),
    Setting("mlock", "--mlock", "bool", False, "GPU & Memory", (), deprecated=True,
            tooltip="Legacy (deprecated upstream in favor of --load-mode, but works on all "
                    "image versions). Lock the model in RAM so the OS never swaps it out. "
                    "Needs enough RAM and the privilege to lock memory. Ignored when "
                    "load-mode is set."),
    Setting("split-mode", "--split-mode", "enum", "layer", "GPU & Memory", ("-sm",),
            enum=("none", "layer", "row", "tensor"),
            tooltip="How to split the model across multiple GPUs. 'layer' (default) splits "
                    "by layer; 'row' splits tensors by row; 'tensor' splits weights and KV "
                    "(experimental); 'none' uses a single GPU only."),
    Setting("tensor-split", "--tensor-split", "string", "", "GPU & Memory", ("-ts",),
            tooltip="Proportions for spreading the model across GPUs, e.g. '3,1' puts 75% on "
                    "GPU0 and 25% on GPU1. Empty = split evenly."),
    Setting("main-gpu", "--main-gpu", "int", 0, "GPU & Memory", ("-mg",), 0, 64, 1,
            tooltip="Index of the primary GPU, used for small tensors and as the sole GPU "
                    "with split-mode 'none'/'row'. 0 = first GPU."),
    Setting("device", "--device", "string", "", "GPU & Memory", ("-dev",),
            tooltip="Comma-separated list of devices to use (e.g. 'CUDA0,CUDA1'), restricting "
                    "which GPUs are visible. Empty = use all available devices."),
    Setting("no-kv-offload", "--no-kv-offload", "bool", False, "GPU & Memory", (),
            tooltip="Keep the KV cache in system RAM instead of GPU VRAM. Frees VRAM for "
                    "weights but makes generation slower."),
    Setting("no-mmproj-offload", "--no-mmproj-offload", "bool", False, "GPU & Memory", (),
            tooltip="Keep the multimodal (vision/audio) projector on the CPU instead of the "
                    "GPU. llama.cpp offloads it by default; enable this to save VRAM for the "
                    "main model."),
    Setting("mmproj-device", "--mmproj-device", "string", "", "GPU & Memory", ("-mmdev",),
            tooltip="Which device runs the multimodal (vision/audio) projector, e.g. "
                    "'CUDA0'. 'none' keeps it on the CPU (like --no-mmproj-offload); empty "
                    "lets llama.cpp choose (auto). Run llama-server with --list-devices to "
                    "see the names. Newer llama.cpp only."),
    Setting("override-tensor", "--override-tensor", "string", "", "GPU & Memory", ("-ot",),
            tooltip="Map tensor-name patterns to buffer types, e.g. keep MoE experts on CPU "
                    "with 'exps=CPU'. Comma-separated for multiple. Empty = no override."),

    # Performance & Batching
    Setting("threads", "--threads", "int", -1, "Performance & Batching", ("-t",), -1, 256, 1,
            tooltip="CPU threads used during generation. Most useful for CPU or partial "
                    "offload; matching your physical core count is a good start. -1 = auto."),
    Setting("threads-batch", "--threads-batch", "int", -1, "Performance & Batching", ("-tb",),
            -1, 256, 1,
            tooltip="CPU threads used for prompt processing / batch decoding. -1 = use the "
                    "same value as --threads."),
    Setting("batch-size", "--batch-size", "int", 2048, "Performance & Batching", ("-b",),
            1, 1048576, 1,
            tooltip="Logical batch size: max tokens submitted together for prompt processing. "
                    "Larger can speed up long prompts but uses more memory. Default 2048."),
    Setting("ubatch-size", "--ubatch-size", "int", 512, "Performance & Batching", ("-ub",),
            1, 1048576, 1,
            tooltip="Physical (micro) batch size: tokens processed in one pass on the GPU. "
                    "Larger uses more VRAM; smaller reduces peak memory. Default 512."),
    Setting("parallel", "--parallel", "int", -1, "Performance & Batching", ("-np",), -1, 256, 1,
            tooltip="Number of concurrent request slots the server serves; the context is "
                    "divided among them. Higher = more parallel users, less context each. "
                    "-1 = auto."),
    Setting("no-cont-batching", "--no-cont-batching", "bool", False, "Performance & Batching", (),
            tooltip="Disable continuous batching (which is on by default). Continuous "
                    "batching boosts throughput with multiple slots; disabling it is rarely "
                    "needed."),
    Setting("numa", "--numa", "enum", "off", "Performance & Batching", (),
            enum=("off", "distribute", "isolate", "numactl"),
            tooltip="NUMA optimization strategy for multi-socket systems. 'off' does not "
                    "pass --numa. 'distribute' spreads threads across nodes; 'isolate' pins "
                    "to one node; 'numactl' follows the numactl map."),
    Setting("threads-http", "--threads-http", "int", -1, "Performance & Batching", (),
            -1, 256, 1,
            tooltip="Threads used to process HTTP requests. -1 = auto. Raise for many "
                    "concurrent clients."),

    # Caching
    Setting("cache-reuse", "--cache-reuse", "int", 0, "Caching", (), 0, 1048576, 1,
            tooltip="Reuse a cached prompt prefix via KV shifting when at least this many "
                    "tokens match, speeding up repeated/long prompts. 0 = off."),
    Setting("no-cache-prompt", "--no-cache-prompt", "bool", False, "Caching", (),
            tooltip="Disable reusing the previous prompt's KV cache (on by default). Each "
                    "request will reprocess the whole prompt from scratch."),
    Setting("cache-ram", "--cache-ram", "int", -1, "Caching", ("-cram",), -1, 1048576, 256,
            tooltip="RAM (MiB) for caching prompt states across requests. -1 = unlimited, "
                    "0 = disable the RAM cache."),
    Setting("ctx-checkpoints", "--ctx-checkpoints", "int", 32, "Caching", ("-ctxcp",),
            0, 1024, 1,
            tooltip="Maximum context checkpoints kept per slot, used by SWA models for fast "
                    "context reuse. Default 32."),
    Setting("checkpoint-min-step", "--checkpoint-min-step", "int", 8192, "Caching", ("-cms",),
            0, 1048576, 64,
            tooltip="Minimum spacing in tokens between context checkpoints. 0 = no minimum. "
                    "Default 8192."),

    # Sampling (core)
    Setting("temp", "--temp", "float", 0.80, "Sampling", (), 0.0, 2.0, 0.05,
            tooltip="Sampling temperature: higher = more random/creative, lower = more "
                    "focused/deterministic. 0 = greedy (always the top token). Typical 0.6-0.8."),
    Setting("top-k", "--top-k", "int", 40, "Sampling", (), 0, 200, 1,
            tooltip="Keep only the K most likely tokens before sampling. Lower = safer/"
                    "narrower, higher = more variety. 0 = disabled. Typical 40."),
    Setting("top-p", "--top-p", "float", 0.95, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Nucleus sampling: consider the smallest set of tokens whose "
                    "probabilities sum to P. Lower = safer/narrower. 1.0 = disabled."),
    Setting("min-p", "--min-p", "float", 0.05, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Drop tokens less likely than this fraction of the top token's "
                    "probability. Higher = stricter. 0.0 = disabled."),
    Setting("typical-p", "--typical-p", "float", 1.0, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Locally-typical sampling: keep tokens whose information content is near "
                    "the expected value, summing to P. Lower = narrower. 1.0 = disabled."),
    Setting("top-n-sigma", "--top-n-sigma", "float", -1.0, "Sampling", (), -1.0, 5.0, 0.1,
            tooltip="Keep only tokens within N standard deviations of the top logit. Lower = "
                    "stricter. Negative value = disabled."),
    Setting("seed", "--seed", "int", -1, "Sampling", (), -1, 2147483647, 1,
            tooltip="Random seed for sampling. Use a fixed value for reproducible output; "
                    "-1 = pick a random seed each run."),
    # Sampling: DRY
    Setting("dry-multiplier", "--dry-multiplier", "float", 0.0, "Sampling", (), 0.0, 5.0, 0.01,
            tooltip="DRY repetition penalty strength, which suppresses repeated multi-token "
                    "sequences. Higher = stronger suppression. 0 = off. Try ~0.8 to enable."),
    Setting("dry-base", "--dry-base", "float", 1.75, "Sampling", (), 1.0, 4.0, 0.05,
            tooltip="DRY base: controls how steeply the penalty grows with the length of a "
                    "repeated sequence. Higher = penalize long repeats more aggressively."),
    Setting("dry-allowed-length", "--dry-allowed-length", "int", 2, "Sampling", (), 1, 20, 1,
            tooltip="DRY: longest repeated sequence allowed before the penalty kicks in. "
                    "Lower = catches shorter repeats sooner. Default 2."),
    Setting("dry-penalty-last-n", "--dry-penalty-last-n", "int", 64, "Sampling", (), 0, 1048576, 1,
            tooltip="DRY look-back: how many recent tokens to scan for repeats. "
                    "0 = disabled. Default 64."),
    Setting("dry-sequence-breaker", "--dry-sequence-breaker", "string", "", "Sampling", (),
            tooltip="DRY sequence breaker. Setting this REPLACES llama.cpp's defaults "
                    "(newline : \" *); use 'none' to disable all breakers. Empty = keep "
                    "defaults. One breaker here; use raw-args for multiple."),
    # Sampling: Penalties
    Setting("repeat-penalty", "--repeat-penalty", "float", 1.0, "Sampling", (), 1.0, 2.0, 0.01,
            tooltip="Penalty applied to recently used tokens to reduce repetition. >1.0 "
                    "discourages repeats; 1.0 = off. Strong values can harm coherence."),
    Setting("repeat-last-n", "--repeat-last-n", "int", 64, "Sampling", (), 0, 1048576, 1,
            tooltip="How many recent tokens the repeat/frequency/presence penalties look "
                    "back over. 0 = disabled. Default 64."),
    Setting("frequency-penalty", "--frequency-penalty", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.01,
            tooltip="Penalize tokens in proportion to how often they already appeared. "
                    "Higher = less repetition of frequent words. 0 = off."),
    Setting("presence-penalty", "--presence-penalty", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.01,
            tooltip="Penalize tokens that have appeared at all, regardless of count, "
                    "encouraging new topics. Higher = more novelty. 0 = off."),
    # Sampling: Mirostat
    Setting("mirostat", "--mirostat", "int", 0, "Sampling", (), 0, 2, 1,
            tooltip="Mirostat adaptively targets a constant output 'surprise' instead of "
                    "top-k/top-p. 0 = off, 1 = Mirostat v1, 2 = Mirostat v2."),
    Setting("mirostat-lr", "--mirostat-lr", "float", 0.1, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Mirostat learning rate (eta): how quickly it adjusts toward the target "
                    "entropy. Higher = reacts faster. Default 0.1."),
    Setting("mirostat-ent", "--mirostat-ent", "float", 5.0, "Sampling", (), 0.0, 10.0, 0.1,
            tooltip="Mirostat target entropy (tau): the desired output randomness. Higher = "
                    "more varied/creative text. Default 5.0."),
    # Sampling: Dynamic temp
    Setting("dynatemp-range", "--dynatemp-range", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.05,
            tooltip="Dynamic temperature range: temperature varies within temp +/- this "
                    "amount based on token uncertainty. Higher = more adaptive. 0 = off."),
    Setting("dynatemp-exp", "--dynatemp-exp", "float", 1.0, "Sampling", (), 0.0, 4.0, 0.1,
            tooltip="Dynamic temperature exponent: shapes how sharply temperature responds "
                    "to uncertainty. Higher = more abrupt changes. Default 1.0."),
    # Sampling: XTC
    Setting("xtc-probability", "--xtc-probability", "float", 0.0, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="XTC (Exclude Top Choices): probability of removing high-likelihood "
                    "tokens to boost creativity. Higher = applied more often. 0 = off."),
    Setting("xtc-threshold", "--xtc-threshold", "float", 0.10, "Sampling", (), 0.0, 0.5, 0.01,
            tooltip="XTC threshold: only tokens above this probability are eligible for "
                    "removal. Lower = removes more aggressively. Above 0.5 disables XTC."),

    # Server & Tools
    Setting("port", "--port", "int", DEFAULT_PORT, "Server & Tools", (), 1, 65535, 1,
            tooltip="TCP port the server listens on (bound to 127.0.0.1). Connect clients to "
                    f"http://localhost:<port>. Default {DEFAULT_PORT}."),
    Setting("api-key", "--api-key", "string", "", "Server & Tools", (), secret=True,
            tooltip="Require this key in the Authorization header for API requests; supply "
                    "several comma-separated. Empty = no authentication."),
    Setting("jinja", "--jinja", "bool", False, "Server & Tools", (),
            tooltip="Use the Jinja2 chat-template engine to format conversations. Needed for "
                    "models with complex templates and for tool calling."),
    Setting("chat-template", "--chat-template", "string", "", "Server & Tools", (),
            tooltip="Override the chat template with a built-in one by name (e.g. 'chatml', "
                    "'llama3'). Empty = use the template embedded in the model."),
    Setting("chat-template-file", "--chat-template-file", "string", "", "Server & Tools", (),
            tooltip="Path to a custom Jinja chat-template file to use instead of the model's "
                    "built-in template. Empty = use the model's own template."),
    Setting("chat-template-kwargs", "--chat-template-kwargs", "string", "", "Server & Tools", (),
            tooltip="Extra parameters passed to the Jinja template as a JSON object string, "
                    "e.g. '{\"enable_thinking\":false}'. Must be a valid JSON object. "
                    "Empty = pass nothing extra."),
    Setting("tools", "--tools", "multiselect", "", "Server & Tools", (), danger=True,
            enum=("read_file", "write_file", "edit_file",
                  "file_glob_search", "grep_search", "exec_shell_command",
                  "get_info"),
            tooltip="Built-in server-side agent tools the model can call. DANGER: "
                    "exec_shell_command runs arbitrary commands inside the container; only "
                    "enable in trusted setups - your mounted folders are the only sandbox.",
            option_help=(
                ("read_file", "Read the contents of a file inside the mounted folders."),
                ("write_file", "Create or overwrite a file. Writes into any :rw mount (e.g. your workspace)."),
                ("edit_file", "Make targeted edits to an existing file. Writes into :rw mounts."),
                ("file_glob_search", "Find files by name pattern (glob), e.g. **/*.py."),
                ("grep_search", "Search inside file contents (like grep) across the mounted folders."),
                ("exec_shell_command", "DANGER: runs ARBITRARY shell commands inside the container. Trusted models only."),
                ("get_info", "Return runtime info: OS name/version and the working directory. Harmless."),
            )),
    Setting("reasoning", "--reasoning", "enum", "auto", "Server & Tools", ("-rea",),
            enum=("on", "off", "auto"),
            tooltip="Controls whether the model emits its reasoning/thinking output. 'auto' "
                    "follows the model's default; 'on' forces it, 'off' suppresses it."),
    Setting("reasoning-budget", "--reasoning-budget", "int", -1, "Server & Tools", (),
            -1, 1048576, 1,
            tooltip="Maximum tokens the model may spend on internal reasoning before "
                    "answering. -1 = unrestricted; 0 = no thinking."),
    Setting("reasoning-budget-message", "--reasoning-budget-message", "string", "",
            "Server & Tools", (),
            tooltip="Text injected before the end-of-thinking tag when the reasoning budget "
                    "is exhausted (nudges the model to stop thinking and answer). Only "
                    "meaningful with a reasoning-budget above 0. Empty = inject nothing."),
    Setting("metrics", "--metrics", "bool", False, "Server & Tools", (),
            tooltip="Expose the Prometheus /metrics endpoint (needed for the Monitor "
                    "tab's throughput numbers). Off by default in llama-server."),
    Setting("no-slots", "--no-slots", "bool", False, "Server & Tools", (),
            tooltip="Disable the /slots endpoint. /slots is ON by default and powers "
                    "per-slot + KV-cache monitoring, but it can expose prompt text; "
                    "enable this to turn it off."),
    Setting("props", "--props", "bool", False, "Server & Tools", (),
            tooltip="Allow changing global properties via POST /props. Reading /props "
                    "works without this."),
    Setting("no-webui", "--no-webui", "bool", False, "Server & Tools", (),
            tooltip="Disable the built-in Web UI. The HTTP/OpenAI-compatible API stays "
                    "available; only the browser chat UI is turned off."),
    Setting("reasoning-format", "--reasoning-format", "enum", "auto", "Server & Tools", (),
            enum=("auto", "none", "deepseek", "deepseek-legacy"),
            tooltip="How thinking/<think> content is parsed and returned. 'auto' follows the "
                    "template; 'none' leaves it inline; 'deepseek' moves it to "
                    "reasoning_content; 'deepseek-legacy' keeps tags and also fills it."),

    # Speculative Decoding
    Setting("spec-type", "--spec-type", "enum", "none", "Speculative Decoding", (),
            enum=("none", "draft-simple", "draft-eagle3", "draft-dflash",
                  "draft-dspark", "draft-mtp",
                  "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod",
                  "ngram-cache"),
            tooltip="Speculative-decoding strategy. 'draft-mtp' enables a model's built-in "
                    "multi-token-prediction head (e.g. Gemma 4) with no separate draft "
                    "model needed. 'none' disables speculation."),
    Setting("spec-draft-ngl", "--spec-draft-ngl", "int_or_token", "auto", "Speculative Decoding",
            ("-ngld",), 0, 999, 1, tokens=("auto", "all"),
            tooltip="Draft-model layers to offload to GPU for speculative decoding. "
                    "'auto' lets llama.cpp decide."),
    Setting("spec-draft-n-max", "--spec-draft-n-max", "int", 3, "Speculative Decoding", (),
            1, 64, 1,
            tooltip="Tokens the draft model proposes per step (speculative decoding). "
                    "Default 3."),
    Setting("spec-draft-n-min", "--spec-draft-n-min", "int", 0, "Speculative Decoding", (),
            0, 64, 1,
            tooltip="Minimum draft tokens to use per speculative step. Default 0."),
    # Negative form, like cors-credentials: llama.cpp offloads draft sampling to
    # the backend by default, so the bool defaults False (unchecked = keep the
    # default on, emit nothing) and checking it emits the --no- disable flag.
    Setting("spec-draft-backend-sampling", "--no-spec-draft-backend-sampling", "bool", False,
            "Speculative Decoding", (),
            tooltip="Disable offloading draft-model sampling to the backend during "
                    "speculative decoding. llama.cpp enables backend draft sampling by "
                    "default; check this to force it off."),
    Setting("cache-type-k-draft", "--cache-type-k-draft", "enum", "f16", "Speculative Decoding",
            ("-ctkd",), enum=KV_CACHE_TYPES,
            tooltip="K KV-cache quantization for the draft model. Lower precision "
                    "(e.g. q8_0) saves VRAM at a small quality cost."),
    Setting("cache-type-v-draft", "--cache-type-v-draft", "enum", "f16", "Speculative Decoding",
            ("-ctvd",), enum=KV_CACHE_TYPES,
            tooltip="V KV-cache quantization for the draft model. Lower precision "
                    "(e.g. q8_0) saves VRAM at a small quality cost."),
    # Embedding & Reranking
    Setting("embeddings", "--embeddings", "bool", False, "Embedding & Reranking", (),
            tooltip="Restrict the server to the embedding use case (serves /embeddings and "
                    "/v1/embeddings). Use only with dedicated embedding models."),
    Setting("pooling", "--pooling", "enum", "model default", "Embedding & Reranking", (),
            enum=("model default", "none", "mean", "cls", "last", "rank"),
            tooltip="Pooling type for embeddings. 'model default' does not pass --pooling "
                    "(llama-server uses the model's own default). Rerankers require 'rank'."),
    Setting("reranking", "--reranking", "bool", False, "Embedding & Reranking", (),
            tooltip="Enable the reranking endpoint (/rerank, /v1/rerank). A reranker also "
                    "needs pooling = rank and --embeddings enabled."),

    # Router (llama-server started with no model; see tools/server/README.md
    # "Using multiple models"). These are meaningful only on a router process.
    # The default MUST mirror upstream, not our advice: widgets emit only when
    # value != default, so a default of 1 meant "leave it alone" emitted nothing
    # and the server silently ran 4.
    Setting("models-max", "--models-max", "int", 4, "Router", (), 0, 64, 1,
            tooltip="Maximum number of models the router keeps loaded at once. "
                    "0 = unlimited. llama.cpp defaults to 4; SET THIS TO 1 on a "
                    "single consumer GPU, where two large models will not co-fit.",
            suggestions=(1, 2, 4, 0)),
    # Negative form, like cors-credentials: a bool defaulting to True can never
    # emit (checked == default stores nothing; unchecked renders no flag).
    Setting("models-autoload", "--no-models-autoload", "bool", False, "Router", (),
            tooltip="Disable automatic loading when a request names a model. "
                    "Checked = models must be loaded explicitly via /models/load; "
                    "unchecked leaves llama.cpp's autoload default on."),

    # Server lifecycle
    Setting("sleep-idle-seconds", "--sleep-idle-seconds", "int", -1, "Server & Tools", (),
            -1, 86400, 30,
            tooltip="Unload the model and its KV cache after this many idle seconds; "
                    "the next request reloads it automatically. -1 = never sleep. "
                    "/health, /props and /models do not reset the idle timer.",
            suggestions=(-1, 300, 600, 1800, 3600)),

    # CORS
    Setting("cors-origins", "--cors-origins", "string", "*", "Networking & CORS", (),
            tooltip="Comma-separated allowed CORS origins. The special value "
                    "'localhost' reflects the Origin header only when it is localhost. "
                    "Note: --tools/--agent/MCP clamp this to localhost."),
    Setting("cors-methods", "--cors-methods", "string", "", "Networking & CORS", (),
            tooltip="Comma-separated allowed CORS methods. Empty = llama.cpp default "
                    "(GET, POST, DELETE, OPTIONS)."),
    Setting("cors-headers", "--cors-headers", "string", "", "Networking & CORS", (),
            tooltip="Comma-separated allowed CORS headers. Empty = llama.cpp default (*)."),
    Setting("cors-credentials", "--no-cors-credentials", "bool", False, "Networking & CORS", (),
            tooltip="Disable CORS credentials. llama.cpp enables them by default; with "
                    "origins set to '*' the Origin header is echoed back and credentials "
                    "are always allowed."),
    Setting("sse-ping-interval", "--sse-ping-interval", "int", 15, "Networking & CORS", (),
            -1, 3600, 5,
            tooltip="Seconds between SSE keep-alive pings while a stream is silent, so a "
                    "long prompt-processing phase does not look like a dead connection. "
                    "-1 = disabled."),

    # MCP (built-in agent)
    Setting("mcp-servers-config", "--mcp-servers-config", "string", "", "Networking & CORS", (),
            tooltip="Path (inside the container) to a JSON file of MCP server definitions, "
                    "Cursor-compatible format. Experimental; clamps CORS origins to localhost.",
            danger=True),
    Setting("mcp-servers-json", "--mcp-servers-json", "string", "", "Networking & CORS", (),
            tooltip="Inline JSON of MCP server definitions, Cursor-compatible format. "
                    "Experimental; clamps CORS origins to localhost.",
            danger=True),

    # Chat behaviour
    Setting("reasoning-preserve", "--reasoning-preserve", "bool", False, "Server & Tools", (),
            tooltip="Preserve the reasoning trace across the whole history rather than only "
                    "the last assistant message. Needs a template advertising "
                    "'supports_preserve_reasoning'."),


    # -- Context Extension (RoPE / YaRN) --------------------------------------
    # Stretching a model past its trained context. Every value here defaults to
    # "loaded from the model", so an untouched form emits nothing and the model's
    # own trained settings win.
    Setting("rope-scaling", "--rope-scaling", "enum", "unset", "Context Extension", (),
            enum=("unset", "none", "linear", "yarn"),
            tooltip="How to stretch the model past its trained context length. 'unset' "
                    "leaves it to the model (usually linear). 'yarn' is the better choice "
                    "for large extensions and enables the YaRN values below."),
    Setting("rope-scale", "--rope-scale", "float", 0.0, "Context Extension", (), 0.0, 128.0, 0.5,
            tooltip="Context scaling factor N: expands the context by a factor of N. "
                    "0 = leave at the model's own value. Reciprocal of rope-freq-scale, so "
                    "set one or the other, not both."),
    Setting("rope-freq-base", "--rope-freq-base", "float", 0.0, "Context Extension", (),
            0.0, 10000000.0, 10000.0,
            tooltip="RoPE base frequency for NTK-aware scaling. 0 = load from the model. "
                    "Raising it is the manual way to extend context on models that were "
                    "not trained with a scaling method."),
    Setting("rope-freq-scale", "--rope-freq-scale", "float", 0.0, "Context Extension", (),
            0.0, 1.0, 0.05,
            tooltip="RoPE frequency scaling factor: expands context by 1/N. 0 = load from "
                    "the model. Inverse of rope-scale; set one, not both."),
    Setting("yarn-orig-ctx", "--yarn-orig-ctx", "int", 0, "Context Extension", (),
            0, 1048576, 1024,
            tooltip="YaRN: the model's ORIGINAL trained context size, the baseline the "
                    "extension is measured against. 0 = use the model's training context."),
    Setting("yarn-ext-factor", "--yarn-ext-factor", "float", -1.0, "Context Extension", (),
            -1.0, 1.0, 0.1,
            tooltip="YaRN extrapolation mix factor. -1 = leave at the model default; "
                    "0.0 = full interpolation (the usual choice for long-context use)."),
    Setting("yarn-attn-factor", "--yarn-attn-factor", "float", -1.0, "Context Extension", (),
            -1.0, 10.0, 0.1,
            tooltip="YaRN attention magnitude scale. -1 = model default. Rarely needs "
                    "changing; adjust only alongside a documented YaRN recipe."),
    Setting("yarn-beta-slow", "--yarn-beta-slow", "float", -1.0, "Context Extension", (),
            -1.0, 128.0, 1.0,
            tooltip="YaRN high correction dimension (alpha). -1 = model default."),
    Setting("yarn-beta-fast", "--yarn-beta-fast", "float", -1.0, "Context Extension", (),
            -1.0, 128.0, 1.0,
            tooltip="YaRN low correction dimension (beta). -1 = model default."),

    # -- Device memory auto-fit ------------------------------------------------
    # NOTE: upstream defaults --fit to 'on', so llama.cpp may already be shrinking
    # unset args (notably ctx-size) to make a model fit. That happens INSIDE the
    # server, after the launcher's own VRAM preflight has run, so the two can
    # disagree: the preflight can warn about a config the server then quietly
    # adjusts. Exposing the switch lets a user turn it off and get hard failures.
    Setting("fit", "--fit", "enum", "unset", "GPU & Memory", ("-fit",),
            enum=("unset", "on", "off"),
            tooltip="Let llama.cpp adjust arguments you left unset (context size above "
                    "all) so the model fits device memory. Upstream default is on. "
                    "Set 'off' to make an oversized config fail loudly instead of being "
                    "silently shrunk, which also makes the VRAM estimate above "
                    "authoritative."),
    Setting("fit-target", "--fit-target", "string", "", "GPU & Memory", ("-fitt",),
            tooltip="Memory margin in MiB to leave free per device for --fit. One value "
                    "is broadcast to every device, or give a comma-separated list per "
                    "device. Empty = upstream default (1024)."),
    Setting("fit-ctx", "--fit-ctx", "int", 0, "GPU & Memory", ("-fitc",), 0, 1048576, 1024,
            tooltip="Floor on the context size --fit is allowed to shrink to. 0 = upstream "
                    "default (4096). Stops auto-fit from silently giving you a context far "
                    "smaller than the workload needs."),

    # -- Memory / correctness knobs -------------------------------------------
    Setting("kv-unified", "--kv-unified", "bool", False, "GPU & Memory", ("-kvu",),
            tooltip="Share ONE unified KV buffer across all sequences instead of a "
                    "per-slot buffer. Upstream enables this when the slot count is auto. "
                    "Usually lowers KV memory with many slots."),
    Setting("no-op-offload", "--no-op-offload", "bool", False, "GPU & Memory", (),
            tooltip="Keep host tensor operations on the CPU instead of offloading them to "
                    "the device. llama.cpp offloads by default; enabling this is a "
                    "debugging/compatibility escape hatch."),
    Setting("check-tensors", "--check-tensors", "bool", False, "Model & Context", (),
            tooltip="Validate model tensor data for invalid values while loading. Slows "
                    "startup; worth it once when a new or self-quantized GGUF produces "
                    "garbage output."),
    Setting("override-kv", "--override-kv", "string", "", "Model & Context", (),
            tooltip="Override model metadata by key, comma-separated, as "
                    "KEY=TYPE:VALUE with type int, float, bool or str (e.g. "
                    "'tokenizer.ggml.add_bos_token=bool:false'). Advanced: wrong values "
                    "can break tokenization."),
    Setting("no-warmup", "--no-warmup", "bool", False, "Performance & Batching", (),
            tooltip="Skip the empty warmup run at startup. Gets the server listening "
                    "sooner, at the cost of a slower first real request."),
    Setting("no-repack", "--no-repack", "bool", False, "Performance & Batching", ("-nr",),
            tooltip="Disable weight repacking. Repacking is enabled upstream and normally "
                    "speeds up CPU inference; turn it off to rule it out when debugging a "
                    "CPU performance or correctness problem."),
    Setting("no-cache-idle-slots", "--no-cache-idle-slots", "bool", False, "Caching", (),
            tooltip="Stop saving idle slots to the prompt cache when a new task arrives. "
                    "Upstream caches them (it needs cache-ram set); disabling trades prompt "
                    "reuse for lower memory churn."),

    # -- CPU & Threading -------------------------------------------------------
    # Affinity/priority knobs that matter for CPU and hybrid CPU+GPU inference,
    # especially with MoE offload where CPU work is on the critical path.
    Setting("cpu-mask", "--cpu-mask", "string", "", "CPU & Threading", ("-C",),
            tooltip="CPU affinity mask as an arbitrarily long hex string. Complements "
                    "cpu-range. Empty = let the OS schedule freely."),
    Setting("cpu-range", "--cpu-range", "string", "", "CPU & Threading", ("-Cr",),
            tooltip="CPU range for affinity, written lo-hi (e.g. '0-15'). Pin generation "
                    "threads to physical cores to avoid SMT siblings stealing throughput."),
    Setting("cpu-strict", "--cpu-strict", "enum", "unset", "CPU & Threading", (),
            enum=("unset", "0", "1"),
            tooltip="Strict CPU placement: 1 keeps threads on the masked CPUs instead of "
                    "letting the scheduler migrate them. 'unset' = upstream default (0)."),
    Setting("prio", "--prio", "int", 0, "CPU & Threading", (), -1, 3, 1,
            tooltip="Process/thread priority: -1 low, 0 normal, 1 medium, 2 high, "
                    "3 realtime. Above normal usually needs elevated privileges."),
    Setting("poll", "--poll", "int", 50, "CPU & Threading", (), 0, 100, 10,
            tooltip="Polling level while waiting for work, 0 to 100. 0 sleeps instead of "
                    "spinning (lower CPU use when idle); upstream default is 50."),
    Setting("cpu-mask-batch", "--cpu-mask-batch", "string", "", "CPU & Threading", ("-Cb",),
            tooltip="CPU affinity mask for BATCH/prompt processing, if it should differ "
                    "from generation. Empty = same as cpu-mask."),
    Setting("cpu-range-batch", "--cpu-range-batch", "string", "", "CPU & Threading", ("-Crb",),
            tooltip="CPU range (lo-hi) for batch/prompt processing. Empty = same as "
                    "cpu-range."),
    Setting("cpu-strict-batch", "--cpu-strict-batch", "enum", "unset", "CPU & Threading", (),
            enum=("unset", "0", "1"),
            tooltip="Strict CPU placement for batch processing. 'unset' = same as "
                    "cpu-strict."),
    Setting("prio-batch", "--prio-batch", "int", 0, "CPU & Threading", (), -1, 3, 1,
            tooltip="Process/thread priority during batch processing, same scale as prio."),
    Setting("poll-batch", "--poll-batch", "int", 50, "CPU & Threading", (), 0, 100, 10,
            tooltip="Polling level while waiting for batch work. Defaults to the poll "
                    "value."),

    # -- Server identity & HTTP surface ---------------------------------------
    Setting("alias", "--alias", "string", "", "Server & Tools", ("-a",),
            tooltip="Model name reported to API clients, comma-separated for several. "
                    "Without it the model id is the full container path of the GGUF, "
                    "which is what /v1/models and every harness will show. Set a short "
                    "stable name here and clients keep working across model file changes."),
    Setting("tags", "--tags", "string", "", "Server & Tools", (),
            tooltip="Comma-separated model tags. Informational only; llama.cpp does not "
                    "route on them."),
    Setting("api-prefix", "--api-prefix", "string", "", "Networking & CORS", (),
            tooltip="Serve every endpoint under this path prefix, written without a "
                    "trailing slash (e.g. '/llama'). Needed when a reverse proxy mounts "
                    "the server somewhere other than the root."),
    Setting("path", "--path", "string", "", "Networking & CORS", (),
            tooltip="Directory of static files to serve. Empty = serve the built-in web "
                    "UI. Container launches need this path to be inside a mount."),
    Setting("timeout", "--timeout", "int", 3600, "Networking & CORS", ("-to",), 1, 86400, 60,
            tooltip="Server read/write timeout in seconds (upstream default 3600). Raise "
                    "it if very long generations are cut off mid-stream by the server "
                    "itself rather than by the client."),
    Setting("reuse-port", "--reuse-port", "bool", False, "Networking & CORS", (),
            tooltip="Allow several sockets to bind the same port (SO_REUSEPORT). Lets a "
                    "replacement instance bind before the old one has fully released the "
                    "port; otherwise leave off so a port clash is a loud error."),
    Setting("ssl-key-file", "--ssl-key-file", "string", "", "Networking & CORS", (),
            tooltip="PEM-encoded private key, which turns the server HTTPS. Must be set "
                    "together with ssl-cert-file, and inside a mount for container "
                    "launches. Without TLS an API key crosses the network in clear text."),
    Setting("ssl-cert-file", "--ssl-cert-file", "string", "", "Networking & CORS", (),
            tooltip="PEM-encoded certificate matching ssl-key-file. Set both or neither."),
    Setting("agent", "--agent", "bool", False, "Server & Tools", ("-ag",),
            tooltip="Enable the CORS proxy and ALL built-in tools. Upstream warns not to "
                    "enable this in untrusted environments; it also clamps CORS origins to "
                    "localhost. Leave off unless you specifically want agent features.",
            danger=True),
    Setting("slot-save-path", "--slot-save-path", "string", "", "Server & Tools", (),
            tooltip="Directory for saving and restoring per-slot KV cache. Empty = "
                    "disabled. Must be a writable mount on a container launch."),
    Setting("slot-prompt-similarity", "--slot-prompt-similarity", "float", 0.1,
            "Server & Tools", ("-sps",), 0.0, 1.0, 0.05,
            tooltip="How closely a request's prompt must match a slot's cached prompt to "
                    "reuse that slot (upstream default 0.10, 0.0 = disabled). Higher is "
                    "stricter and reuses less."),
    Setting("media-path", "--media-path", "string", "", "Server & Tools", (),
            tooltip="Directory local media may be loaded from via file:// URLs with "
                    "relative paths. Empty = disabled. Container launches need it mounted."),
    Setting("skip-chat-parsing", "--skip-chat-parsing", "bool", False, "Server & Tools", (),
            tooltip="Force a pure content parser even when a Jinja template is set, so "
                    "reasoning traces and tool calls stay inline in the content field "
                    "instead of being split out. Useful when a client wants the raw text."),
    Setting("no-prefill-assistant", "--no-prefill-assistant", "bool", False, "Server & Tools", (),
            tooltip="Treat a trailing assistant message as a complete message rather than "
                    "a prefix to continue. llama.cpp prefills by default; enable this when "
                    "a client sends full assistant turns it does not want extended."),
    Setting("lora-init-without-apply", "--lora-init-without-apply", "bool", False,
            "Server & Tools", (),
            tooltip="Load the LoRA adapters below but start with all of them inactive, so "
                    "they can be switched on and rescaled at runtime through the "
                    "/lora-adapters endpoint without restarting the server."),

    # -- Logging ---------------------------------------------------------------
    # The Monitor tab tails container logs, so prefix/timestamp/verbosity changes
    # alter what it parses. Defaults here leave llama.cpp's own format untouched.
    Setting("log-file", "--log-file", "string", "", "Logging", (),
            tooltip="Also write the log to this file. Empty = stdout/stderr only, which "
                    "is what the Monitor tab reads; a container path must be in a "
                    "writable mount."),
    Setting("log-colors", "--log-colors", "enum", "unset", "Logging", (),
            enum=("unset", "on", "off", "auto"),
            tooltip="ANSI colour in log output. 'auto' (upstream default) colours only a "
                    "terminal. Set 'off' when the captured log is being read by tooling."),
    Setting("verbosity", "--verbosity", "int", 0, "Logging", ("-lv",), 0, 10, 1,
            tooltip="Verbosity threshold: messages above this level are dropped. 0 is the "
                    "normal level; raise it to debug a launch that fails quietly."),
    Setting("log-disable", "--log-disable", "bool", False, "Logging", (),
            tooltip="Silence logging entirely. This also blanks the Monitor tab's log "
                    "pane and its throughput parsing, so it is rarely what you want.",
            danger=True),
    Setting("no-log-prefix", "--no-log-prefix", "bool", False, "Logging", (),
            tooltip="Drop the level prefix from log lines. Prefixes are on by default."),
    Setting("no-log-timestamps", "--no-log-timestamps", "bool", False, "Logging", (),
            tooltip="Drop timestamps from log lines. Timestamps are on by default and "
                    "make the Monitor log pane far easier to correlate with requests."),
    Setting("log-prompts-dir", "--log-prompts-dir", "string", "", "Logging", (),
            tooltip="Write every prompt to this directory, created if missing. Debugging "
                    "aid only: prompts are stored verbatim, so treat the directory as "
                    "sensitive. Empty = disabled.",
            danger=True),

    # -- Multimodal ------------------------------------------------------------
    Setting("image-min-tokens", "--image-min-tokens", "int", 0, "Multimodal", (), 0, 65536, 64,
            tooltip="Floor on the tokens one image may use, for vision models with dynamic "
                    "resolution. 0 = read from the model. Raising it keeps small images "
                    "legible at the cost of context."),
    Setting("image-max-tokens", "--image-max-tokens", "int", 0, "Multimodal", (), 0, 65536, 64,
            tooltip="Ceiling on the tokens one image may use, for dynamic-resolution "
                    "vision models. 0 = read from the model. Lowering it bounds how much "
                    "context a large image can consume."),
    Setting("mtmd-batch-max-tokens", "--mtmd-batch-max-tokens", "int", 1024, "Multimodal", (),
            64, 65536, 256,
            tooltip="Maximum image tokens encoded per batch (upstream default 1024). "
                    "Lower it if encoding a large image runs the GPU out of memory."),

    # -- Embedding -------------------------------------------------------------
    Setting("embd-normalize", "--embd-normalize", "int", 2, "Embedding & Reranking", (),
            -1, 8, 1,
            tooltip="Normalisation applied to returned embeddings: -1 none, 0 max absolute "
                    "int16, 1 taxicab, 2 euclidean (the default), above 2 = p-norm. Match "
                    "whatever your vector store expects."),

    # -- Sampling defaults -----------------------------------------------------
    # Server-wide defaults only; a request body still overrides them per call.
    Setting("samplers", "--samplers", "string", "", "Sampling", (),
            tooltip="Sampler chain in application order, separated by ';' (upstream "
                    "default 'penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;"
                    "temperature'). Empty = leave at the default."),
    Setting("sampler-seq", "--sampler-seq", "string", "", "Sampling", (),
            tooltip="The same chain in short letter form (upstream default 'edskypmxt'). "
                    "Set this or samplers, not both."),
    Setting("ignore-eos", "--ignore-eos", "bool", False, "Sampling", (),
            tooltip="Keep generating past the end-of-stream token. Effectively an infinite "
                    "'-inf' bias on EOS, so generation stops only at the token limit."),
    Setting("adaptive-target", "--adaptive-target", "float", -1.0, "Sampling", (), -1.0, 1.0, 0.05,
            tooltip="adaptive-p: target probability that tokens are selected near. "
                    "Negative = disabled (the default)."),
    Setting("adaptive-decay", "--adaptive-decay", "float", 0.9, "Sampling", (), 0.0, 0.99, 0.05,
            tooltip="adaptive-p: how fast the target adapts. Lower reacts faster, higher "
                    "is steadier. Only used when adaptive-target is enabled."),

    # -- Speculative decoding: draft-model placement and ngram tuning ----------
    Setting("spec-default", "--spec-default", "bool", False, "Speculative Decoding", (),
            tooltip="Turn on llama.cpp's default speculative-decoding configuration "
                    "instead of configuring each knob by hand."),
    Setting("spec-draft-p-min", "--spec-draft-p-min", "float", 0.0, "Speculative Decoding", (),
            0.0, 1.0, 0.05,
            tooltip="Minimum probability for a greedy draft token to be proposed. 0 = "
                    "upstream default. Raising it drafts less but wastes fewer rejected "
                    "tokens."),
    Setting("spec-draft-p-split", "--spec-draft-p-split", "float", 0.0, "Speculative Decoding", (),
            0.0, 1.0, 0.05,
            tooltip="Probability threshold at which the draft splits into a new branch. "
                    "0 = upstream default."),
    Setting("spec-draft-device", "--spec-draft-device", "string", "", "Speculative Decoding",
            ("-devd",),
            tooltip="Devices the DRAFT model runs on, comma-separated (e.g. 'CUDA1'). "
                    "Empty = same placement as the target model. Use it to park the draft "
                    "on a second, smaller GPU."),
    Setting("spec-draft-threads", "--spec-draft-threads", "int", 0, "Speculative Decoding",
            ("-td",), 0, 512, 1,
            tooltip="CPU threads for the draft model. 0 = same as the main threads value."),
    Setting("spec-draft-cpu-moe", "--spec-draft-cpu-moe", "bool", False, "Speculative Decoding",
            ("-cmoed",),
            tooltip="Keep ALL of the draft model's MoE expert weights on the CPU, freeing "
                    "VRAM for the target model."),
    Setting("spec-draft-n-cpu-moe", "--spec-draft-n-cpu-moe", "int", 0, "Speculative Decoding",
            ("-ncmoed",), 0, 512, 1,
            tooltip="Keep the draft model's first N MoE layers on the CPU. 0 = none. The "
                    "partial version of spec-draft-cpu-moe."),
    Setting("spec-draft-override-tensor", "--spec-draft-override-tensor", "string",
            "", "Speculative Decoding", ("-otd",),
            tooltip="Tensor-name pattern to buffer-type map for the DRAFT model, the same "
                    "form as override-tensor (e.g. 'exps=CPU'). Empty = no override."),
    Setting("spec-ngram-mod-n-min", "--spec-ngram-mod-n-min", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-mod: smallest ngram length used to draft. 0 = upstream default. "
                    "Only applies when spec-type is an ngram variant."),
    Setting("spec-ngram-mod-n-max", "--spec-ngram-mod-n-max", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-mod: largest ngram length used to draft. 0 = upstream default."),
    Setting("spec-ngram-mod-n-match", "--spec-ngram-mod-n-match", "int", 0,
            "Speculative Decoding", (), 0, 256, 1,
            tooltip="ngram-mod: lookup length (upstream default 24). 0 = leave alone."),
    Setting("spec-ngram-simple-size-n", "--spec-ngram-simple-size-n", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-simple: lookup ngram size N. 0 = upstream default."),
    Setting("spec-ngram-simple-size-m", "--spec-ngram-simple-size-m", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-simple: drafted length M. 0 = upstream default."),
    Setting("spec-ngram-simple-min-hits", "--spec-ngram-simple-min-hits", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-simple: minimum hits before a draft is accepted (upstream "
                    "default 1). 0 = leave alone."),
    Setting("spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-n", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k: lookup ngram size N. 0 = upstream default."),
    Setting("spec-ngram-map-k-size-m", "--spec-ngram-map-k-size-m", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k: drafted length M. 0 = upstream default."),
    Setting("spec-ngram-map-k-min-hits", "--spec-ngram-map-k-min-hits", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k: minimum hits before a draft is accepted (upstream "
                    "default 1). 0 = leave alone."),
    Setting("spec-ngram-map-k4v-size-n", "--spec-ngram-map-k4v-size-n", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k4v: lookup ngram size N. 0 = upstream default."),
    Setting("spec-ngram-map-k4v-size-m", "--spec-ngram-map-k4v-size-m", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k4v: drafted length M. 0 = upstream default."),
    Setting("spec-ngram-map-k4v-min-hits", "--spec-ngram-map-k4v-min-hits", "int", 0,
            "Speculative Decoding", (), 0, 64, 1,
            tooltip="ngram-map-k4v: minimum hits before a draft is accepted (upstream "
                    "default 1). 0 = leave alone."),

    # ik_llama.cpp-only flags (engine-gated). Shown only when engine == ik and
    # dropped from argv on a mainline launch (current_profile + command_builder).
    Setting("run-time-repack", "--run-time-repack", "bool", False, "ik_llama.cpp", ("-rtr",),
            engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Repack tensors kept in RAM to a row-interleaved "
                    "layout at load time; can speed up CPU/hybrid inference but DISABLES "
                    "mmap and raises load time and RAM. Off by default."),
    Setting("no-fused-moe", "--no-fused-moe", "bool", False, "ik_llama.cpp", ("-no-fmoe",),
            engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Disable fused MoE ops, which ik enables by default "
                    "for a MoE speedup. Leave unchecked to keep fused MoE on."),
    Setting("mla-use", "--mla-use", "enum", "auto", "ik_llama.cpp", ("-mla",),
            enum=("auto", "0", "1", "2", "3"), engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Multi-head Latent Attention for DeepSeek-style "
                    "models. 'auto' leaves ik's default; 0 disables; 1/2/3 select MLA "
                    "variants."),
    # Default MUST mirror current ik upstream (256 since ik PR #2312), not our
    # advice: widgets emit only when value != default, so with the old default
    # of 0 "no cap" emitted nothing and current ik silently ran 256.
    Setting("attention-max-batch", "--attention-max-batch", "int", 256, "ik_llama.cpp", ("-amb",),
            0, 65536, 64, engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Cap the K*Q attention buffer (MiB) to bound memory "
                    "on long contexts; only applies when flash-attn is off. 0 = no cap. "
                    "256 is the ik default (older ik builds defaulted to 0 = no cap). "
                    "ik raises values 1-127 to 128."),
    Setting("smart-expert-reduction", "--smart-expert-reduction", "string", "", "ik_llama.cpp",
            ("-ser",), engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Custom active-expert count for MoE models, form "
                    "'Kmin,t' (e.g. '4,0.5'). Empty = leave at the model default."),
    Setting("ctx-size-draft", "--ctx-size-draft", "int", 0, "ik_llama.cpp", ("-cd",),
            0, 1048576, 1024, engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only (mainline removed this flag). Prompt-context size "
                    "for the speculative-decoding draft model. 0 = inherit the target "
                    "context for DFlash/DSpark (their capacity knob), otherwise load from "
                    "the model."),
    Setting("swa-compress", "--swa-compress", "bool", False, "ik_llama.cpp", (),
            engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Allocate sliding-window-attention layers at the "
                    "window size instead of the full context, saving KV-cache memory on "
                    "SWA models (e.g. Gemma). Off by default."),
    Setting("indexer-cache-type-k", "--indexer-cache-type-k", "enum", "f16", "ik_llama.cpp",
            ("-ictk",), enum=("f16", "q8_0"), engine="ik_llama.cpp",
            tooltip="ik_llama.cpp only. Data type of the indexer K-cache used by "
                    "DeepSeek-style sparse attention (DSA) models. q8_0 saves memory; "
                    "f16 is the default."),
]

# Flags mainline llama.cpp accepts but ik_llama.cpp does NOT.
#
# ik is a fork that diverged, and it rejects a large slice of the shared surface.
# Without this gate those settings reached an ik launch and died on "unknown
# argument" the moment a user set one (they are only emitted when actually
# present in a profile's settings, so it broke the setting, not every launch).
#
# PROBE-DERIVED, not parsed from --help: ik's help under-reports what its parser
# accepts (--n-gpu-layers, --n-predict, --embeddings and --alias are all accepted
# while undocumented), so every entry here was confirmed by running the flag
# against ghcr.io/ikawrakow/ik-llama-cpp:cu12-server and checking for "unknown
# argument". Regenerate with tests/fixtures/regen_ik_flags.sh.
MAINLINE_ONLY_FLAGS: frozenset = frozenset({
    "--agent", "--api-prefix", "--cache-reuse", "--checkpoint-min-step",
    "--cors-headers", "--cors-methods", "--cors-origins", "--cpu-mask",
    "--cpu-mask-batch", "--cpu-range", "--cpu-range-batch", "--cpu-strict",
    "--cpu-strict-batch", "--fit-ctx", "--fit-target", "--kv-unified", "--lazy-mode",
    "--load-mode", "--log-colors", "--log-prompts-dir", "--mcp-servers-config",
    "--mcp-servers-json", "--media-path", "--mmproj-device", "--models-max",
    "--mtmd-batch-max-tokens", "--n-cpu-ffn", "--no-cache-idle-slots",
    "--no-cache-prompt", "--no-cors-credentials", "--no-log-prefix",
    "--no-log-timestamps", "--no-models-autoload", "--no-op-offload", "--no-repack",
    "--no-spec-draft-backend-sampling", "--no-webui", "--poll", "--poll-batch",
    "--prio", "--prio-batch", "--props", "--reasoning-preserve", "--reranking",
    "--reuse-port", "--sampler-seq", "--sleep-idle-seconds", "--spec-default",
    "--spec-draft-cpu-moe", "--spec-draft-device", "--spec-draft-n-cpu-moe",
    "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-ngl",
    "--spec-draft-override-tensor", "--spec-draft-p-min", "--spec-draft-p-split",
    "--spec-draft-threads", "--spec-ngram-map-k-min-hits", "--spec-ngram-map-k-size-m",
    "--spec-ngram-map-k-size-n", "--spec-ngram-map-k4v-min-hits",
    "--spec-ngram-map-k4v-size-m", "--spec-ngram-map-k4v-size-n",
    "--spec-ngram-mod-n-match", "--spec-ngram-mod-n-max", "--spec-ngram-mod-n-min",
    "--spec-ngram-simple-min-hits", "--spec-ngram-simple-size-m",
    "--spec-ngram-simple-size-n", "--sse-ping-interval", "--swa-full", "--tags",
    "--tools", "--typical-p",
})

# Applied here rather than as engine="llama.cpp" on 75 separate Setting() calls:
# one list is auditable against the probe output, and a regenerated list is a
# one-hunk diff instead of 75 scattered ones.
_ALL = [replace(s, engine="llama.cpp")
        if s.engine == "any" and s.flag in MAINLINE_ONLY_FLAGS else s
        for s in _ALL]

CATALOG: dict = {s.key: s for s in _ALL}

# Settings that exist ONLY on a router process.
ROUTER_ONLY_KEYS: frozenset = frozenset({"models-max", "models-autoload"})

# Settings a router profile may set. Everything not listed here is model-level
# and belongs on member profiles instead.
#
# This split is load-bearing, not cosmetic: llama.cpp resolves preset options as
# `router CLI args > per-model preset section > [*] global`, so CLI args WIN. A
# router carrying `-c 8192` would silently override every member's own `c` value.
# MainWindow.active_catalog() consumes these to filter the settings form and to
# filter what current_profile() collects, which is what makes the trap
# unreachable -- the form really is filtered, so keep it that way.
HOST_KEYS: frozenset = frozenset({
    "models-max", "models-autoload", "sleep-idle-seconds",
    "port", "api-key", "threads-http", "metrics", "no-webui",
    "tools",
    "cors-origins", "cors-methods", "cors-headers", "cors-credentials",
    "sse-ping-interval", "mcp-servers-config", "mcp-servers-json",
    # The router process is the one serving HTTP, so the HTTP surface, TLS and
    # logging belong to it rather than to a member. Anything model-level (alias,
    # slot paths, sampling) stays off this list and is set per member.
    "api-prefix", "path", "timeout", "reuse-port", "agent",
    "ssl-key-file", "ssl-cert-file",
    "log-file", "log-colors", "verbosity", "log-disable",
    "no-log-prefix", "no-log-timestamps", "log-prompts-dir",
}) & frozenset(CATALOG)


def router_catalog() -> dict:
    """Settings shown on a router profile's form."""
    return {k: s for k, s in CATALOG.items() if k in HOST_KEYS}


def member_catalog() -> dict:
    """Settings shown on an ordinary (member/single-server) profile's form."""
    return {k: s for k, s in CATALOG.items() if k not in ROUTER_ONLY_KEYS}


def for_engine(catalog: dict, engine: str) -> dict:
    """Drop settings tagged for a different engine. 'any' settings always pass;
    an engine-specific setting passes only when it matches `engine`."""
    return {k: s for k, s in catalog.items()
            if s.engine == "any" or s.engine == engine}
