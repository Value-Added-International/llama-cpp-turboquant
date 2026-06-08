## NAME[¶](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#NAME)

llama-batched-bench - llama-batched-bench

## DESCRIPTION[¶](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#DESCRIPTION)

**\-----** common params **\-----**

**\-h**, **\--help**, **\--usage** print usage and exit **\--version** show version and build info **\-cl**, **\--cache-list** show list of models in cache **\--completion-bash** print source-able bash completion script for llama.cpp **\-t**, **\--threads** N number of CPU threads to use during generation (default: <num\_cpus>)

(env: LLAMA\_ARG\_THREADS)

## \-tb, --threads-batch N number of threads to use during batch and prompt processing (default:[¶](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#-tb,_--threads-batch_N_number_of_threads_to_use_during_batch_and_prompt_processing_(default:)

same as **\--threads**)

**\-C**, **\--cpu-mask** M CPU affinity mask: arbitrarily long hex. Complements cpu-range

(default: "")

**\-Cr**, **\--cpu-range** lo-hi range of CPUs for affinity. Complements **\--cpu-mask** **\--cpu-strict** <0|1> use strict CPU placement (default: 0) **\--prio** N set process/thread priority : low(**\-1**), normal(0), medium(1), high(2),

realtime(3) (default: 0)

**\--poll** <0...100> use polling level to wait for work (0 - no polling, default: 50) **\-Cb**, **\--cpu-mask-batch** M CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch

(default: same as **\--cpu-mask**)

**\-Crb**, **\--cpu-range-batch** lo-hi ranges of CPUs for affinity. Complements **\--cpu-mask-batch** **\--cpu-strict-batch** <0|1> use strict CPU placement (default: same as **\--cpu-strict**) **\--prio-batch** N set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime

(default: 0)

**\--poll-batch** <0|1> use polling to wait for work (default: same as **\--poll**) **\-c**, **\--ctx-size** N size of the prompt context (default: 0, 0 = loaded from model)

(env: LLAMA\_ARG\_CTX\_SIZE)

**\-n**, **\--predict**, **\--n-predict** N number of tokens to predict (default: **\-1**, **\-1** = infinity)

(env: LLAMA\_ARG\_N\_PREDICT)

**\-b**, **\--batch-size** N logical maximum batch size (default: 2048)

(env: LLAMA\_ARG\_BATCH)

**\-ub**, **\--ubatch-size** N physical maximum batch size (default: 512)

(env: LLAMA\_ARG\_UBATCH)

**\--keep** N number of tokens to keep from the initial prompt (default: 0, **\-1** =

all)

**\--swa-full** use full-size SWA cache (default: false)

\[(more

info)\]([https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)) (env: LLAMA\_ARG\_SWA\_FULL)

**\-fa**, **\--flash-attn** \[on|off|auto\] set Flash Attention use ('on', 'off', or 'auto', default: 'auto')

(env: LLAMA\_ARG\_FLASH\_ATTN)

**\-p**, **\--prompt** PROMPT prompt to start generation with; for system message, use **\-sys** **\--perf**, **\--no-perf** whether to enable internal libllama performance timings (default:

[false)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#false))

(env: LLAMA\_ARG\_PERF)

**\-f**, **\--file** FNAME a file containing the prompt (default: none) **\-bf**, **\--binary-file** FNAME binary file containing the prompt (default: none) **\-e**, **\--escape**, **\--no-escape** whether to process escapes sequences (\\n, \\r, \\t, \\', \\", \\\\)

(default: true)

**\--rope-scaling** {none,linear,yarn} RoPE frequency scaling method, defaults to linear unless specified by

[the model](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#the)

(env: LLAMA\_ARG\_ROPE\_SCALING\_TYPE)

**\--rope-scale** N RoPE context scaling factor, expands context by a factor of N

(env: LLAMA\_ARG\_ROPE\_SCALE)

**\--rope-freq-base** N RoPE base frequency, used by NTK-aware scaling (default: loaded from

[model)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#model))

(env: LLAMA\_ARG\_ROPE\_FREQ\_BASE)

**\--rope-freq-scale** N RoPE frequency scaling factor, expands context by a factor of 1/N

(env: LLAMA\_ARG\_ROPE\_FREQ\_SCALE)

**\--yarn-orig-ctx** N YaRN: original context size of model (default: 0 = model training

[context size)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#context)

(env: LLAMA\_ARG\_YARN\_ORIG\_CTX)

**\--yarn-ext-factor** N YaRN: extrapolation mix factor (default: **\-1**.00, 0.0 = full

[interpolation)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#interpolation))

(env: LLAMA\_ARG\_YARN\_EXT\_FACTOR)

**\--yarn-attn-factor** N YaRN: scale sqrt(t) or attention magnitude (default: **\-1**.00)

(env: LLAMA\_ARG\_YARN\_ATTN\_FACTOR)

**\--yarn-beta-slow** N YaRN: high correction dim or alpha (default: **\-1**.00)

(env: LLAMA\_ARG\_YARN\_BETA\_SLOW)

**\--yarn-beta-fast** N YaRN: low correction dim or beta (default: **\-1**.00)

(env: LLAMA\_ARG\_YARN\_BETA\_FAST)

**\-kvo**, **\--kv-offload**, **\-nkvo**, **\--no-kv-offload**

[whether to enable KV cache offloading (default: enabled)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#whether)

(env: LLAMA\_ARG\_KV\_OFFLOAD)

**\--repack**, **\-nr**, **\--no-repack** whether to enable weight repacking (default: enabled)

(env: LLAMA\_ARG\_REPACK)

**\--no-host** bypass host buffer allowing extra buffers to be used

(env: LLAMA\_ARG\_NO\_HOST)

**\-ctk**, **\--cache-type-k** TYPE KV cache data type for K

[allowed values: f32, f16, bf16, q8\_0, q4\_0, q4\_1, iq4\_nl, q5\_0, q5\_1](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#allowed)

(default: f16) (env: LLAMA\_ARG\_CACHE\_TYPE\_K)

**\-ctv**, **\--cache-type-v** TYPE KV cache data type for V

[allowed values: f32, f16, bf16, q8\_0, q4\_0, q4\_1, iq4\_nl, q5\_0, q5\_1](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#allowed~2)

(default: f16) (env: LLAMA\_ARG\_CACHE\_TYPE\_V)

**\-dt**, **\--defrag-thold** N KV cache defragmentation threshold (DEPRECATED)

(env: LLAMA\_ARG\_DEFRAG\_THOLD)

**\-np**, **\--parallel** N number of parallel sequences to decode (default: 1)

(env: LLAMA\_ARG\_N\_PARALLEL)

**\--mlock** force system to keep model in RAM rather than swapping or compressing

(env: LLAMA\_ARG\_MLOCK)

**\--mmap**, **\--no-mmap** whether to memory-map model. (if mmap disabled, slower load but may

[reduce pageouts if not using mlock) (default: enabled)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#reduce)

(env: LLAMA\_ARG\_MMAP)

**\-dio**, **\--direct-io**, **\-ndio**, **\--no-direct-io**

[use DirectIO if available. (default: disabled)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#use)

(env: LLAMA\_ARG\_DIO)

**\--numa** TYPE attempt optimizations that help on some NUMA systems

[\- distribute: spread execution evenly over all nodes](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#distribute:)

\- isolate: only spawn threads on CPUs on the node that execution started on - numactl: use the CPU map provided by numactl if run without this previously, it is recommended to drop the system page cache before using this see [https://github.com/ggml-org/llama.cpp/issues/1437](https://github.com/ggml-org/llama.cpp/issues/1437) (env: LLAMA\_ARG\_NUMA)

**\-dev**, **\--device** <dev1,dev2,..> comma-separated list of devices to use for offloading (none = don't

[offload)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#offload))

use **\--list-devices** to see a list of available devices (env: LLAMA\_ARG\_DEVICE)

**\--list-devices** print list of available devices and exit **\-ot**, **\--override-tensor** <tensor name pattern>=<buffer type>,...

[override tensor buffer type](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#override)

(env: LLAMA\_ARG\_OVERRIDE\_TENSOR)

**\-cmoe**, **\--cpu-moe** keep all Mixture of Experts (MoE) weights in the CPU

(env: LLAMA\_ARG\_CPU\_MOE)

**\-ncmoe**, **\--n-cpu-moe** N keep the Mixture of Experts (MoE) weights of the first N layers in the

[CPU](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#CPU)

(env: LLAMA\_ARG\_N\_CPU\_MOE)

**\-ngl**, **\--gpu-layers**, **\--n-gpu-layers** N max. number of layers to store in VRAM, either an exact number,

'auto', or 'all' (default: auto)

(env: LLAMA\_ARG\_N\_GPU\_LAYERS)

**\-sm**, **\--split-mode** {none,layer,row,tensor}

[how to split the model across multiple GPUs, one of:](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#how)

\- none: use one GPU only - layer (default): split layers and KV across GPUs (pipelined) - row: split weight across GPUs by rows (parallelized) - tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL) (env: LLAMA\_ARG\_SPLIT\_MODE)

**\-ts**, **\--tensor-split** N0,N1,N2,... fraction of the model to offload to each GPU, comma-separated list of

[proportions, e.g. 3,1](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#proportions,)

(env: LLAMA\_ARG\_TENSOR\_SPLIT)

**\-mg**, **\--main-gpu** INDEX the GPU to use for the model (with split-mode = none), or for

[intermediate results and KV (with split-mode = row) (default: 0)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#intermediate)

(env: LLAMA\_ARG\_MAIN\_GPU)

**\-fit**, **\--fit** \[on|off\] whether to adjust unset arguments to fit in device memory ('on' or

'off', default: 'on')

(env: LLAMA\_ARG\_FIT)

**\-fitt**, **\--fit-target** MiB0,MiB1,MiB2,...

[target margin per device for **\--fit**, comma-separated list of values,](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#target)

single value is broadcast across all devices, default: 1024 (env: LLAMA\_ARG\_FIT\_TARGET)

**\-fitc**, **\--fit-ctx** N minimum ctx size that can be set by **\--fit** option, default: 4096

(env: LLAMA\_ARG\_FIT\_CTX)

**\--check-tensors** check model tensor data for invalid values (default: false) **\--override-kv** KEY=TYPE:VALUE,... advanced option to override model metadata by key. to specify multiple

[overrides, either use comma-separated values.](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#overrides,)

types: int, float, bool, str. example: **\--override-kv** tokenizer.ggml.add\_bos\_token=bool:false,tokenizer.ggml.add\_eos\_token=bool:false

**\--op-offload**, **\--no-op-offload** whether to offload host tensor operations to device (default: true) **\--lora** FNAME path to LoRA adapter (use comma-separated values to load multiple

adapters)

## \--lora-scaled FNAME:SCALE,... path to LoRA adapter with user defined scaling (format:[¶](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#--lora-scaled_FNAME:SCALE,..._path_to_LoRA_adapter_with_user_defined_scaling_(format:)

[FNAME:SCALE,...)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#FNAME:SCALE,...))

note: use comma-separated values

**\--control-vector** FNAME add a control vector

note: use comma-separated values to add multiple control vectors

**\--control-vector-scaled** FNAME:SCALE,...

[add a control vector with user defined scaling SCALE](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#add)

note: use comma-separated values (format: FNAME:SCALE,...)

**\--control-vector-layer-range** START END

layer range to apply the control vector(s) to, start and end inclusive

**\-m**, **\--model** FNAME model path to load

(env: LLAMA\_ARG\_MODEL)

**\-mu**, **\--model-url** MODEL\_URL model download url (default: unused)

(env: LLAMA\_ARG\_MODEL\_URL)

**\-dr**, **\--docker-repo** \[<repo>/\]<model>\[:quant\]

[Docker Hub model repository. repo is optional, default to ai/. quant](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#Docker)

is optional, default to :latest. example: gemma3 (default: unused) (env: LLAMA\_ARG\_DOCKER\_REPO)

**\-hf**, **\-hfr**, **\--hf-repo** <user>/<model>\[:quant\]

[Hugging Face model repository; quant is optional, case-insensitive,](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#Hugging)

default to Q4\_K\_M, or falls back to the first file in the repo if Q4\_K\_M doesn't exist. mmproj is also downloaded automatically if available. to disable, add **\--no-mmproj** example: ggml-org/GLM-4.7-Flash-GGUF:Q4\_K\_M (default: unused) (env: LLAMA\_ARG\_HF\_REPO)

**\-hff**, **\--hf-file** FILE Hugging Face model file. If specified, it will override the quant in

[**\--hf-repo** (default: unused)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#hf)

(env: LLAMA\_ARG\_HF\_FILE)

**\-hfv**, **\-hfrv**, **\--hf-repo-v** <user>/<model>\[:quant\]

[Hugging Face model repository for the vocoder model (default: unused)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#Hugging~2)

(env: LLAMA\_ARG\_HF\_REPO\_V)

**\-hffv**, **\--hf-file-v** FILE Hugging Face model file for the vocoder model (default: unused)

(env: LLAMA\_ARG\_HF\_FILE\_V)

**\-hft**, **\--hf-token** TOKEN Hugging Face access token (default: value from HF\_TOKEN environment

[variable)](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#variable))

(env: HF\_TOKEN)

**\--log-disable** Log disable **\--log-file** FNAME Log to file

(env: LLAMA\_ARG\_LOG\_FILE)

**\--log-colors** \[on|off|auto\] Set colored logging ('on', 'off', or 'auto', default: 'auto')

'auto' enables colors when output is to a terminal

(env: LLAMA\_ARG\_LOG\_COLORS)

**\-v**, **\--verbose**, **\--log-verbose** Set verbosity level to infinity (i.e. log all messages, useful for

debugging)

**\--offline** Offline mode: forces use of cache, prevents network access

(env: LLAMA\_ARG\_OFFLINE)

**\-lv**, **\--verbosity**, **\--log-verbosity** N Set the verbosity threshold. Messages with a higher verbosity will be

[ignored. Values:](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#ignored.)

\- 0: generic output - 1: error - 2: warning - 3: info - 4: trace (more info) - 5: debug

(default: 3)

(env: LLAMA\_ARG\_LOG\_VERBOSITY)

**\--log-prefix**, **\--no-log-prefix** Enable prefix in log messages

(env: LLAMA\_ARG\_LOG\_PREFIX)

**\--log-timestamps**, **\--no-log-timestamps** Enable timestamps in log messages

(env: LLAMA\_ARG\_LOG\_TIMESTAMPS)

**\--spec-draft-type-k**, **\-ctkd**, **\--cache-type-k-draft** TYPE

[KV cache data type for K for the draft model](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#KV)

allowed values: f32, f16, bf16, q8\_0, q4\_0, q4\_1, iq4\_nl, q5\_0, q5\_1 (default: f16) (env: LLAMA\_ARG\_SPEC\_DRAFT\_CACHE\_TYPE\_K)

**\--spec-draft-type-v**, **\-ctvd**, **\--cache-type-v-draft** TYPE

[KV cache data type for V for the draft model](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#KV~2)

allowed values: f32, f16, bf16, q8\_0, q4\_0, q4\_1, iq4\_nl, q5\_0, q5\_1 (default: f16) (env: LLAMA\_ARG\_SPEC\_DRAFT\_CACHE\_TYPE\_V)

**\-----** sampling params **\-----**

**\--samplers** SAMPLERS samplers that will be used for generation in the order, separated by

';'

(default: penalties;dry;top\_n\_sigma;top\_k;typ\_p;top\_p;min\_p;xtc;temperature)

**\-s**, **\--seed** SEED RNG seed (default: **\-1**, use random seed for **\-1**) **\--sampler-seq**, **\--sampling-seq** SEQUENCE

[simplified sequence for samplers that will be used (default:](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#simplified)

edskypmxt)

**\--ignore-eos** ignore end of stream token and continue generating (implies

**\--logit-bias** EOS-inf)

**\--temp**, **\--temperature** N temperature (default: 0.80) **\--top-k** N top-k sampling (default: 40, 0 = disabled)

(env: LLAMA\_ARG\_TOP\_K)

**\--top-p** N top-p sampling (default: 0.95, 1.0 = disabled) **\--min-p** N min-p sampling (default: 0.05, 0.0 = disabled) **\--top-nsigma**, **\--top-n-sigma** N top-n-sigma sampling (default: **\-1**.00, **\-1**.0 = disabled) **\--xtc-probability** N xtc probability (default: 0.00, 0.0 = disabled) **\--xtc-threshold** N xtc threshold (default: 0.10, 1.0 = disabled) **\--typical**, **\--typical-p** N locally typical sampling, parameter p (default: 1.00, 1.0 = disabled) **\--repeat-last-n** N last n tokens to consider for penalize (default: 64, 0 = disabled, **\-1**

\= ctx\_size)

**\--repeat-penalty** N penalize repeat sequence of tokens (default: 1.00, 1.0 = disabled) **\--presence-penalty** N repeat alpha presence penalty (default: 0.00, 0.0 = disabled) **\--frequency-penalty** N repeat alpha frequency penalty (default: 0.00, 0.0 = disabled) **\--dry-multiplier** N set DRY sampling multiplier (default: 0.00, 0.0 = disabled) **\--dry-base** N set DRY sampling base value (default: 1.75) **\--dry-allowed-length** N set allowed length for DRY sampling (default: 2) **\--dry-penalty-last-n** N set DRY penalty for the last n tokens (default: **\-1**, 0 = disable, **\-1** =

context size)

**\--dry-sequence-breaker** STRING add sequence breaker for DRY sampling, clearing out default breakers

('\\n', ':', '"', '\*') in the process; use "none" to not use any

sequence breakers

**\--adaptive-target** N adaptive-p: select tokens near this probability (valid range 0.0 to

1.0; negative = disabled) (default: **\-1**.00)

\[(more info)\]([https://github.com/ggml-org/llama.cpp/pull/17927](https://github.com/ggml-org/llama.cpp/pull/17927))

**\--adaptive-decay** N adaptive-p: decay rate for target adaptation over time. lower values

[are more reactive, higher values are more stable.](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#are)

(valid range 0.0 to 0.99) (default: 0.90)

**\--dynatemp-range** N dynamic temperature range (default: 0.00, 0.0 = disabled) **\--dynatemp-exp** N dynamic temperature exponent (default: 1.00) **\--mirostat** N use Mirostat sampling.

[Top K, Nucleus and Locally Typical samplers are ignored if used.](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#Top)

(default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)

**\--mirostat-lr** N Mirostat learning rate, parameter eta (default: 0.10) **\--mirostat-ent** N Mirostat target entropy, parameter tau (default: 5.00) **\-l**, **\--logit-bias** TOKEN\_ID(+/-)BIAS modifies the likelihood of token appearing in the completion,

[i.e. \`--logit-bias 15043+1\` to increase likelihood of token ' Hello',](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#i.e.)

or \`--logit-bias 15043-1\` to decrease likelihood of token ' Hello'

**\--grammar** GRAMMAR BNF-like grammar to constrain generations (see samples in grammars/

dir)

**\--grammar-file** FNAME file to read grammar from **\-j**, **\--json-schema** SCHEMA JSON schema to constrain generations ([https://json-schema.org/),](https://json-schema.org/),) e.g.

\`{}\` for any JSON object

For schemas w/ external $refs, use **\--grammar** + example/json\_schema\_to\_grammar.py instead

**\-jf**, **\--json-schema-file** FILE File containing a JSON schema to constrain generations

([https://json-schema.org/),](https://json-schema.org/),) e.g. \`{}\` for any JSON object

For schemas w/ external $refs, use **\--grammar** + example/json\_schema\_to\_grammar.py instead

**\-bs**, **\--backend-sampling** enable backend sampling (experimental) (default: disabled)

(env: LLAMA\_ARG\_BACKEND\_SAMPLING)

**\-----** speculative params **\-----**

**\-----** example-specific params **\-----**

**\-kvu**, **\--kv-unified**, **\-no-kvu**, **\--no-kv-unified**

[use single unified KV buffer shared across all sequences (default:](https://manpages.debian.org/unstable/llama.cpp-tools-extra/llama-batched-bench.1.en.html#use~2)

enabled if number of slots is auto) (env: LLAMA\_ARG\_KV\_UNIFIED)

**\-pps** is the prompt shared across parallel sequences (default: false) **\-tgs** is the text generation separated across the different sequences

(default: false)

**\-npp** n0,n1,... number of prompt tokens **\-ntg** n0,n1,... number of text generation tokens **\-npl** n0,n1,... number of parallel prompts **\--output-format** {md,jsonl} output format for batched-bench results (default: md)

example usage:

obj-x86\_64-linux-gnu/bin/llama-batched-bench **\-m** model.gguf **\-c** 2048 **\-b** 2048 **\-ub** 512 **\-npp** 128,256,512 **\-ntg** 128,256 **\-npl** 1,2,4,8,16,32 \[-pps\]