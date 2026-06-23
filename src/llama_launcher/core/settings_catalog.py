from dataclasses import dataclass, field


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


KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")

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
    Setting("no-mmap", "--no-mmap", "bool", False, "GPU & Memory", (),
            tooltip="Disable memory-mapping of the model file, loading it fully into RAM "
                    "instead. Slower startup and more RAM, but can help with -mlock or odd "
                    "filesystems."),
    Setting("mlock", "--mlock", "bool", False, "GPU & Memory", (),
            tooltip="Lock the model in RAM so the OS never swaps it out, keeping latency "
                    "steady. Needs enough RAM and the privilege to lock memory."),
    Setting("split-mode", "--split-mode", "enum", "layer", "GPU & Memory", ("-sm",),
            enum=("none", "layer", "row"),
            tooltip="How to split the model across multiple GPUs. 'layer' (default) splits "
                    "by layer; 'row' splits tensors by row; 'none' uses a single GPU only."),
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
    Setting("dry-penalty-last-n", "--dry-penalty-last-n", "int", -1, "Sampling", (), -1, 1048576, 1,
            tooltip="DRY look-back: how many recent tokens to scan for repeats. -1 = the "
                    "whole context; 0 = disabled."),
    # Sampling: Penalties
    Setting("repeat-penalty", "--repeat-penalty", "float", 1.0, "Sampling", (), 1.0, 2.0, 0.01,
            tooltip="Penalty applied to recently used tokens to reduce repetition. >1.0 "
                    "discourages repeats; 1.0 = off. Strong values can harm coherence."),
    Setting("repeat-last-n", "--repeat-last-n", "int", 64, "Sampling", (), -1, 1048576, 1,
            tooltip="How many recent tokens the repeat/frequency/presence penalties look "
                    "back over. -1 = whole context; 0 = disabled. Default 64."),
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
    Setting("port", "--port", "int", 8080, "Server & Tools", (), 1, 65535, 1,
            tooltip="TCP port the server listens on (bound to 127.0.0.1). Connect clients to "
                    "http://localhost:<port>. Default 8080."),
    Setting("api-key", "--api-key", "string", "", "Server & Tools", (),
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
    Setting("tools", "--tools", "multiselect", "", "Server & Tools", (), danger=True,
            enum=("read_file", "write_file", "edit_file", "apply_diff",
                  "file_glob_search", "grep_search", "exec_shell_command",
                  "get_datetime"),
            tooltip="Built-in server-side agent tools the model can call. DANGER: "
                    "exec_shell_command runs arbitrary commands inside the container; only "
                    "enable in trusted setups - your mounted folders are the only sandbox.",
            option_help=(
                ("read_file", "Read the contents of a file inside the mounted folders."),
                ("write_file", "Create or overwrite a file. Writes into any :rw mount (e.g. your workspace)."),
                ("edit_file", "Make targeted edits to an existing file. Writes into :rw mounts."),
                ("apply_diff", "Apply a patch/diff to a file. Writes into :rw mounts."),
                ("file_glob_search", "Find files by name pattern (glob), e.g. **/*.py."),
                ("grep_search", "Search inside file contents (like grep) across the mounted folders."),
                ("exec_shell_command", "DANGER: runs ARBITRARY shell commands inside the container. Trusted models only."),
                ("get_datetime", "Return the current date and time. Harmless."),
            )),
    Setting("reasoning", "--reasoning", "enum", "auto", "Server & Tools", ("-rea",),
            enum=("on", "off", "auto"),
            tooltip="Controls whether the model emits its reasoning/thinking output. 'auto' "
                    "follows the model's default; 'on' forces it, 'off' suppresses it."),
    Setting("reasoning-budget", "--reasoning-budget", "int", -1, "Server & Tools", (),
            -1, 1048576, 1,
            tooltip="Maximum tokens the model may spend on internal reasoning before "
                    "answering. -1 = unrestricted; 0 = no thinking."),
]

CATALOG: dict = {s.key: s for s in _ALL}
