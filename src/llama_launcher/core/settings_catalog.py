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


KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")

_ALL = [
    # Model & Context
    Setting("ctx-size", "--ctx-size", "int", 0, "Model & Context", ("-c",), 0, 1048576, 1024,
            tooltip="Context size; 0 = use model's trained length"),
    Setting("n-predict", "--n-predict", "int", -1, "Model & Context", ("-n",), -2, 1048576, 1,
            tooltip="Max tokens to generate; -1 = infinite"),
    Setting("keep", "--keep", "int", 0, "Model & Context", (), -1, 1048576, 1,
            tooltip="Tokens to keep from the initial prompt; -1 = all"),

    # GPU & Memory
    Setting("n-gpu-layers", "--n-gpu-layers", "int_or_token", "auto", "GPU & Memory", ("-ngl",),
            0, 999, 1, tokens=("auto", "all"), tooltip="Layers to offload to VRAM"),
    Setting("n-cpu-moe", "--n-cpu-moe", "int", 0, "GPU & Memory", ("-ncmoe",), 0, 999, 1,
            tooltip="Keep MoE expert weights of first N layers on CPU (MoE models)"),
    Setting("cpu-moe", "--cpu-moe", "bool", False, "GPU & Memory", ("-cmoe",),
            tooltip="Keep ALL MoE expert weights on CPU"),
    Setting("flash-attn", "--flash-attn", "enum", "auto", "GPU & Memory", ("-fa",),
            enum=("on", "off", "auto"), tooltip="Flash attention mode"),
    Setting("cache-type-k", "--cache-type-k", "enum", "f16", "GPU & Memory", ("-ctk",),
            enum=KV_CACHE_TYPES, tooltip="KV cache type for K"),
    Setting("cache-type-v", "--cache-type-v", "enum", "f16", "GPU & Memory", ("-ctv",),
            enum=KV_CACHE_TYPES, tooltip="KV cache type for V"),
    Setting("no-mmap", "--no-mmap", "bool", False, "GPU & Memory", (),
            tooltip="Disable mmap (load whole model into RAM)"),
    Setting("mlock", "--mlock", "bool", False, "GPU & Memory", (),
            tooltip="Lock model in RAM (no swap)"),
    Setting("split-mode", "--split-mode", "enum", "layer", "GPU & Memory", ("-sm",),
            enum=("none", "layer", "row"), tooltip="Multi-GPU split mode"),
    Setting("tensor-split", "--tensor-split", "string", "", "GPU & Memory", ("-ts",),
            tooltip="Per-GPU proportions, e.g. 3,1"),
    Setting("main-gpu", "--main-gpu", "int", 0, "GPU & Memory", ("-mg",), 0, 64, 1,
            tooltip="Main GPU for split-mode none/row"),
    Setting("device", "--device", "string", "", "GPU & Memory", ("-dev",),
            tooltip="Comma-separated devices to offload to (restrict GPUs)"),
    Setting("no-kv-offload", "--no-kv-offload", "bool", False, "GPU & Memory", (),
            tooltip="Keep KV cache in system RAM instead of GPU"),

    # Performance & Batching
    Setting("threads", "--threads", "int", -1, "Performance & Batching", ("-t",), -1, 256, 1,
            tooltip="Generation threads; -1 = auto"),
    Setting("threads-batch", "--threads-batch", "int", -1, "Performance & Batching", ("-tb",),
            -1, 256, 1, tooltip="Batch/prompt threads; -1 = same as threads"),
    Setting("batch-size", "--batch-size", "int", 2048, "Performance & Batching", ("-b",),
            1, 1048576, 1, tooltip="Logical max batch size"),
    Setting("ubatch-size", "--ubatch-size", "int", 512, "Performance & Batching", ("-ub",),
            1, 1048576, 1, tooltip="Physical (micro) batch size"),
    Setting("parallel", "--parallel", "int", -1, "Performance & Batching", ("-np",), -1, 256, 1,
            tooltip="Number of server slots; -1 = auto"),
    Setting("no-cont-batching", "--no-cont-batching", "bool", False, "Performance & Batching", (),
            tooltip="Disable continuous batching (on by default)"),

    # Caching
    Setting("cache-reuse", "--cache-reuse", "int", 0, "Caching", (), 0, 1048576, 1,
            tooltip="Min chunk size (tokens) to reuse via KV shifting"),
    Setting("no-cache-prompt", "--no-cache-prompt", "bool", False, "Caching", (),
            tooltip="Disable prompt caching (on by default)"),
    Setting("cache-ram", "--cache-ram", "int", -1, "Caching", ("-cram",), -1, 1048576, 256,
            tooltip="Prompt cache size in MiB; -1 = unlimited, 0 = off"),

    # Sampling (core)
    Setting("temp", "--temp", "float", 0.80, "Sampling", (), 0.0, 2.0, 0.05,
            tooltip="Temperature; <=0 = greedy"),
    Setting("top-k", "--top-k", "int", 40, "Sampling", (), 0, 200, 1, tooltip="Top-k; 0 = off"),
    Setting("top-p", "--top-p", "float", 0.95, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Top-p; 1.0 = off"),
    Setting("min-p", "--min-p", "float", 0.05, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Min-p; 0.0 = off"),
    Setting("typical-p", "--typical-p", "float", 1.0, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Typical-p; 1.0 = off"),
    Setting("top-n-sigma", "--top-n-sigma", "float", -1.0, "Sampling", (), -1.0, 5.0, 0.1,
            tooltip="Top-n-sigma; negative = off"),
    Setting("seed", "--seed", "int", -1, "Sampling", (), -1, 2147483647, 1,
            tooltip="RNG seed; -1 = random"),
    # Sampling: DRY
    Setting("dry-multiplier", "--dry-multiplier", "float", 0.0, "Sampling", (), 0.0, 5.0, 0.01,
            tooltip="DRY multiplier; 0 = off"),
    Setting("dry-base", "--dry-base", "float", 1.75, "Sampling", (), 1.0, 4.0, 0.05,
            tooltip="DRY base"),
    Setting("dry-allowed-length", "--dry-allowed-length", "int", 2, "Sampling", (), 1, 20, 1,
            tooltip="DRY allowed length"),
    Setting("dry-penalty-last-n", "--dry-penalty-last-n", "int", -1, "Sampling", (), -1, 1048576, 1,
            tooltip="DRY look-back; -1 = ctx, 0 = off"),
    # Sampling: Penalties
    Setting("repeat-penalty", "--repeat-penalty", "float", 1.0, "Sampling", (), 1.0, 2.0, 0.01,
            tooltip="Repeat penalty; 1.0 = off"),
    Setting("repeat-last-n", "--repeat-last-n", "int", 64, "Sampling", (), -1, 1048576, 1,
            tooltip="Repeat window; -1 = ctx, 0 = off"),
    Setting("frequency-penalty", "--frequency-penalty", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.01,
            tooltip="Frequency penalty; 0 = off"),
    Setting("presence-penalty", "--presence-penalty", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.01,
            tooltip="Presence penalty; 0 = off"),
    # Sampling: Mirostat
    Setting("mirostat", "--mirostat", "int", 0, "Sampling", (), 0, 2, 1, tooltip="0=off,1=v1,2=v2"),
    Setting("mirostat-lr", "--mirostat-lr", "float", 0.1, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="Mirostat learning rate (eta)"),
    Setting("mirostat-ent", "--mirostat-ent", "float", 5.0, "Sampling", (), 0.0, 10.0, 0.1,
            tooltip="Mirostat target entropy (tau)"),
    # Sampling: Dynamic temp
    Setting("dynatemp-range", "--dynatemp-range", "float", 0.0, "Sampling", (), 0.0, 2.0, 0.05,
            tooltip="Dynamic temperature range; 0 = off"),
    Setting("dynatemp-exp", "--dynatemp-exp", "float", 1.0, "Sampling", (), 0.0, 4.0, 0.1,
            tooltip="Dynamic temperature exponent"),
    # Sampling: XTC
    Setting("xtc-probability", "--xtc-probability", "float", 0.0, "Sampling", (), 0.0, 1.0, 0.01,
            tooltip="XTC probability; 0 = off"),
    Setting("xtc-threshold", "--xtc-threshold", "float", 0.10, "Sampling", (), 0.0, 0.5, 0.01,
            tooltip="XTC threshold; >0.5 disables"),

    # Server & Tools
    Setting("port", "--port", "int", 8080, "Server & Tools", (), 1, 65535, 1,
            tooltip="Listen port (host bound to 127.0.0.1)"),
    Setting("api-key", "--api-key", "string", "", "Server & Tools", (),
            tooltip="API key(s), comma-separated"),
    Setting("jinja", "--jinja", "bool", False, "Server & Tools", (),
            tooltip="Use Jinja chat-template engine"),
    Setting("chat-template", "--chat-template", "string", "", "Server & Tools", (),
            tooltip="Built-in chat template name"),
    Setting("chat-template-file", "--chat-template-file", "string", "", "Server & Tools", (),
            tooltip="Custom Jinja chat-template file"),
    Setting("tools", "--tools", "multiselect", "", "Server & Tools", (), danger=True,
            enum=("read_file", "write_file", "edit_file", "apply_diff",
                  "file_glob_search", "grep_search", "exec_shell_command",
                  "get_datetime"),
            tooltip="Built-in server tools (e.g. 'all'). DANGER: exec_shell_command runs "
                    "arbitrary commands inside the container."),
    Setting("reasoning", "--reasoning", "enum", "auto", "Server & Tools", ("-rea",),
            enum=("on", "off", "auto"), tooltip="Reasoning/thinking mode"),
    Setting("reasoning-budget", "--reasoning-budget", "int", -1, "Server & Tools", (),
            -1, 1048576, 1, tooltip="Token budget for thinking; -1 = unrestricted"),
]

CATALOG: dict = {s.key: s for s in _ALL}
