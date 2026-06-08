/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server \
    -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    -a qwen3.6-35b-a3b \
    \
    -dev Vulkan0 \
    -ngl 99 \
    -sm layer \
    -fa 1 \
    -ctk q8_0 \
    -ctv q8_0 \
    \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    --repeat-penalty 1.05 \
    -b 2048 \
    -ub 512 \
    -t 8 \
    -tb 8 \
    -c 131072 \
    -n -1 \
    --parallel 1 \
    --cache-prompt \
    --cache-reuse 256 \
    \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --reasoning auto \
    --reasoning-format deepseek \
    --reasoning-budget -1 \
    \
    --host 127.0.0.1 \
    --port 8082 \
    --metrics \
    --props

/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server \
    -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/gemma-4-26B-A4B-it-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    -a gemma-4-26b-a4b \
    \
    -dev Vulkan0 \
    -sm layer \
    -fa 1 \
    -ctk q8_0 \
    -ctv q8_0 \
    \
    -b 2048 \
    -ub 512 \
    -t 8 \
    -tb 8 \
    -c 0 \
    -n -1 \
    --parallel 1 \
    --cache-prompt \
    --cache-reuse 256 \
    \
    --fit on \
    --fit-target 1024 \
    \
    --jinja \
    --reasoning auto \
    --reasoning-format deepseek \
    --reasoning-budget -1 \
    \
    --host 127.0.0.1 \
    --port 8080 \
    --metrics \
    --props

/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-server \
    -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf \
    -ngl 32 -sm layer -fa on -nkvo \
    -ctk turbo4 -ctv turbo4 \
    -c 124928 \
    -b 2048 -ub 512 \
    --jinja \
    --temp 0.6 --top-p 0.95 --top-k 20 \
    --repeat-penalty 1.1 \
    --port 8081
# If it still loops occasionally on deep reasoning chains, --repeat-penalty 1.15 is the next step. G
