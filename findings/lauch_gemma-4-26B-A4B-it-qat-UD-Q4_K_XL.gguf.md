# Models Vulkan

## Gemma 4 26B

/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server \
  -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  -a gemma-4-26b-a4b \
  \
  -dev Vulkan0 \
  -ngl 99           \
  -sm layer         \
  -fa 1             \
  -ctk q8_0         \
  -ctv q8_0         \
  \
  -b 2048           \
  -ub 512           \
  -t 8              \
  -tb 8             \
  -c 131072         \
  -n -1             \
  --parallel 1      \
  --cache-prompt    \
  --cache-reuse 256 \
  --jinja           \
  --reasoning auto  \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  \
  --host 127.0.0.1  \
  --port 8080       \
  --metrics         \
  --props


  # verify

  curl -s http://127.0.0.1:8080/props | python3 -m json.tool | grep -E 'n_ctx|chat_template_tool'

## Qwen 8.4 tps


### warnings
41.16.183.095 W ~llama_context:    Vulkan0 compute buffer size of 565.2208 MiB, does not match expectation of 501.0332 MiB

### Command

/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server \
  -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -a qwen3.6-35b-a3b \
  --no-mmap \
  \
  -dev Vulkan0               \
  -ngl 99                    \
  -sm layer                  \
  -fa 1                      \
  -ctk q8_0                  \
  -ctv q8_0                  \
  \
  -b 2048                    \
  -ub 512                    \
  -t 8                       \
  -tb 8                      \
  -c 131072                  \
  -n -1                      \
  --parallel 1               \
  --cache-prompt             \
  --cache-reuse 256          \
  \
  --jinja                                          \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --reasoning auto                                 \
  --reasoning-format deepseek                      \
  --reasoning-budget -1                            \
  \
  --host 127.0.0.1  \
  --port 8081       \
  --metrics         \
  --props




## suggested
/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server \
  -m /mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -a qwen3.6-35b-a3b \
  --no-mmap \
  -dev Vulkan0 \
  -ngl 99 \
  -sm layer \
  -fa 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  -b 2048 \
  -ub 512 \
  -t 8 \
  -tb 8 \
  -c 8192 \
  -n -1 \
  --parallel 1 \
  --cache-prompt \
  --cache-reuse 256 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --reasoning auto \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --host 127.0.0.1 \
  --port 8081 \
  --metrics \
  --props
