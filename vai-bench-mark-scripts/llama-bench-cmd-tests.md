


## BASIC TEST
./llama-bench \
-m /mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf \
-p 512,2048,4096,16384  -fa 1 \
-b 1024,2048,4096 \
-ub 1024,2048,4096,16384 \
-ctk q8_0 \
-ctv q4_0,turbo3 \
-ngl 99 --no-warmup


## . The MTP Draft KV Cache
--cache-type-k-draft q4_0 --cache-type-v-draft q4_0




## MTP Models

  LOG=/home/none/git/llama-cpp-turboquant/lamma-bench-logs/llama-bench-$(date +%Y%m%d-%H%M%S).jsonl
./llama-bench \
  -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa 1 \
  -ctk q8_0,turbo4 -ctv q4_0,turbo4,turbo3 \
  -p 512,2048,4096,8192 \
  -b 2048 -ub 512 -n 512 \
  --mtp-n-max 0,2 \
  -ctkd q4_0,turbo4 \
  -ctvd q4_0,turbo4,turbo3 \
  -o md -oe jsonl 2>"$LOG"
echo "Log: $LOG"


## Non-MTP Models

   LOG=/home/none/git/llama-cpp-turboquant/lamma-bench-logs/llama-bench-$(date +%Y%m%d-%H%M%S).jsonl
 ./llama-bench \
  -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa 1 \
  -ctk q8_0,turbo4 -ctv q4_0,turbo4,turbo3 \
  -p 512,2048,4096,8192 \
  -b 2048 -ub 512 -n 512 \
  -o md -oe jsonl 2>"$LOG"
echo "Log: $LOG"


1. DeepSeek V4 Pro (or Flash)
Why it pairs perfectly: DeepSeek’s V4 architecture is built explicitly for agentic code implementation. If Gemma 4 26B acts as your "Product Manager / Architect", DeepSeek acts as the "Senior Engineer." It is highly resilient against syntax hallucination and excels at generating complex, multi-line blocks of code based on a provided plan.

2. Qwen 3.6 Plus / Max
Why it pairs perfectly: For massive code bases, Qwen's latest 3.6 generation has specialized repository-level intelligence. It takes the architectural pseudocode or plan that Gemma 4 creates and translates it cleanly into production-ready code while keeping track of imports and dependencies across files.

# Conceptual Orchestration Configuration
models:
  architect:
    model: gemma-4-26b-a4b-it
    temperature: 0.5  # Slightly higher for creative problem-solving/planning
    system_prompt: "You are an expert software architect. Analyze the request and write a detailed, step-by-step implementation plan. Do not write the code itself, only the pseudocode, logic gates, and file structures."

  coder:
    model: qwen-3.6-coder-7b # or gemma-4-e4b
    temperature: 0.0  # Completely deterministic for raw code writing
    system_prompt: "You are a precise code-generation engine. Take the provided architectural plan and implement the exact source code. Output only clean code inside markdown blocks. No explanations."


## Nvidia tests
1. turbo4 K + q4_0 V — best prefill throughput
    # llama-bench
./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo4 -ctv q4_0 -p 512,2048,4096,8192 -n 512

# llama-server
./llama-server -m model.gguf -ngl 99 --flash-attn --cache-type-k turbo4 --cache-type-v q4_0


2. q8_0 K + turbo4 V — excellent prefill, best tg throughput
# llama-bench
./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo4 -p 512,2048,4096,8192 -n 512

# llama-server
./llama-server -m model.gguf -ngl 99 --flash-attn --cache-type-k q8_0 --cache-type-v turbo4


3. q8_0 K + turbo3 V — balanced, lowest stddev (most stable)

# llama-bench
./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv turbo3 -p 512,2048,4096,8192 -n 512

# llama-server
./llama-server -m model.gguf -ngl 99 --flash-attn --cache-type-k q8_0 --cache-type-v turbo3


## Bench parmamters

./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo4 -ctv q4_0 -p 512,2048,4096,8192 -n 512


## model cards and model locations:

1. unslothgemma-4-26B-A4B-it-qat
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/unslothgemma-4-26B-A4B-it-qat.md
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

2. unslothgemma-4-31B-it-qat
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/unslothgemma-4-31B-it-qat.md
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf

3. unslothgemma-4-12B-it-qat
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/unslothgemma-4-12B-it-qat.md
/run/media/none/1d178b00-4086-4162-ba58-a602ac516109/downloads/unsloth/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf


## Lama Bench Locations

/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-bench


/home/none/git/llama-cpp-turboquant/build-x64-linux-hip-rocm713-gfx1103-release/bin/llama-bench


/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-bench

## Log Locations:
LOG=/home/none/git/llama-cpp-turboquant/lamma-bench-logs/llama-bench-[MODEL]-[Nvidia|ROCM|VULKAN]$(date +%Y%m%d-%H%M%S).jsonl
