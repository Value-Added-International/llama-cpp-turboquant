[2610061200:R3:T1:bash_tool][2610061200:R3:T2:bash_tool][2610061200:R3:T3:bash_tool][2610061200:R3:T4:bash_tool][2610061200:R3:T5:bash_tool][2610061200:R3:T6:bash_tool][2610061200:R3:T7:bash_tool][2610061200:R3:T8:bash_tool][2610061200:R3]

Correction to R1: the version comparison against "b9354" was wrong. Your binaries report `9426 (73a7a59c4)`, but that is not upstream b9426.

What the checks show:

1. Commit `73a7a59c4` does not exist in upstream ggml-org/llama.cpp. Upstream tag b9426 is commit `1738129be` (2026-05-30). Your build number counts commits in a different history, so it can't be compared to upstream b-tags.
2. Both of your switch lists offer `turbo2/turbo3/turbo4` KV types, and upstream master has none of these. So both binaries come from the same fork, and the NVIDIA and Vulkan switch lists are byte-identical.
3. I pulled `src/llama-vocab.cpp` at `73a7a59c4`. It has no `minicpm5` entry, which is the root cause. Upstream master has it.
4. Two upstream commits matter for this model:
   - `9777256c3` "convert: add MiniCPM5 tokenizer support (#23384)" first appears in tag b9354. This is needed to load the model.
   - `c818263f2` "chat : implement minicpm5 parser (#24889)" first appears in tag b9833. This is needed for `tool_calls` parsing, which matters because your llama-swap entry sets `tools: true`.
5. Upstream master is currently at b11434.

Fix: you need a build at upstream ≥ b9833, which covers both the tokenizer and the tool-call parser. Rebuilding in place from upstream would drop the `turbo*` KV types, so any model on this host that uses `-ctk/-ctv turbo*` would break. You have two options:

A. Rebase the fork onto upstream ≥ b9833. Run `git -C <fork-src> remote -v` to identify which fork it is.
B. Build upstream into separate prefixes, keep the fork binaries for the turbo models, and add new llama-swap macros for the upstream builds.

Option B, using your existing CUDA archs (86/89 from the startup log). Adjust the source path to yours:

```bash
git clone https://github.com/ggml-org/llama.cpp.git /opt/src/llama.cpp-upstream
cd /opt/src/llama.cpp-upstream
git checkout master
```

```bash
cmake -B build-nvidia -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_INSTALL_PREFIX=/opt/vai-llama/nvidia-upstream
cmake --build build-nvidia -j --target llama-server
cmake --install build-nvidia
```

```bash
cmake -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DCMAKE_INSTALL_PREFIX=/opt/vai-llama/vulkan-upstream
cmake --build build-vulkan -j --target llama-server
cmake --install build-vulkan
```

Then verify both builds contain the two commits:

```bash
cd /opt/src/llama.cpp-upstream
git merge-base --is-ancestor 9777256c3 HEAD && echo tokenizer-ok
git merge-base --is-ancestor c818263f2 HEAD && echo toolparser-ok
/opt/vai-llama/nvidia-upstream/bin/llama-server --version
/opt/vai-llama/vulkan-upstream/bin/llama-server --version
```

The launch commands from R2 then work unchanged once you swap the binary path to `/opt/vai-llama/nvidia-upstream/bin/llama-server` or `/opt/vai-llama/vulkan-upstream/bin/llama-server`. `--device Vulkan0` is correct; your `--list-devices` output confirms Vulkan0 is the 780M.

Stopgap on the current fork (not recommended for real use): force the pre-tokenizer through `--override-kv tokenizer.ggml.pre=str:llama-bpe`. The model will load, but tokenization will not be exact. I ran the upstream `minicpm5` split rules and the fork's `llama3` rules over 200,007 strings, and 3,345 split differently. Every mismatch was a run of whitespace directly before a digit, for example `"  4"` or `" \t91"`. `llama3` splits that run, while `minicpm5` keeps it whole. That pattern is common in indented code and tables, and the result is non-canonical tokens. Tool calls also won't be parsed on the fork (that needs the b9833 parser). Use this only to smoke-test that the model runs:

```bash
/opt/vai-llama/nvidia/bin/llama-server \
  -m /mnt/512_ssd_internal/vai-llms/coding/small/MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic-Q8_0.gguf \
  -a MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic \
  --override-kv tokenizer.ggml.pre=str:llama-bpe \
  --host 0.0.0.0 --port 50011 \
  -c 131072 -np 4 --kv-unified \
  -ngl 99 -fit off \
  -t 4 -tb 8 -cb --no-warmup \
  --jinja -fa on -ctk q8_0 -ctv q8_0 \
  --cache-prompt -ctxcp 32 -cram 16384 \
  -b 2048 -ub 512 \
  --reasoning-format deepseek --reasoning-budget 8192 \
  --temp 1.0 --top-p 0.95 --min-p 0.0 \
  --metrics --props \
  --slot-save-path /mnt/512_ssd_internal/vai-llms/swap
```
