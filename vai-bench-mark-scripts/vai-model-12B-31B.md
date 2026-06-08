
## Bench parmamters

./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo4 -ctv q4_0 -p 512,2048,4096,8192 -n 512

## Unsloth-Gemma-4-QAT overview

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/unsloth-Gemma-4-QAT.md


## model cards and model locations:

1. unslothgemma-4-26B-A4B-it-qat

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/unslothgemma-4-26B-A4B-it-qat.md

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf

2. unslothgemma-4-31B-it-qat

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/unslothgemma-4-31B-it-qat.md

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf

3. unslothgemma-4-12B-it-qat

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/unslothgemma-4-12B-it-qat.md

/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf


## Lama Bench Locations

/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-bench


/home/none/git/llama-cpp-turboquant/build-x64-linux-hip-rocm713-gfx1103-release/bin/llama-bench


/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-bench

## Log Locations:
LOG=/home/none/git/llama-cpp-turboquant/lamma-bench-logs/llama-bench-[MODEL]-[Nvidia|ROCM|VULKAN]$(date +%Y%m%d-%H%M%S).jsonl