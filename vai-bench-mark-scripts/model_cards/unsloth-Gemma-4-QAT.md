Gemma 4 QAT (Quantization-Aware Training) is Google DeepMind’s new [Gemma 4](https://unsloth.ai/docs/models/gemma-4) variants designed to **reduce memory requirements while preserving model quality**. This makes it possible to run larger models, such as **Gemma 4 26B-A4B**, locally on consumer GPUs with as little as **16GB of RAM**.

Gemma 4 QAT is trained with quantization in mind, allowing 4-bit format to have ~**72% lower memory usage** with **near original performance**. 2 special mobile quants of E2B and E4B are also provided which uses a mixture of quant widths.

Converting to `Q4_0` from QAT naively gets only 70.2% top-1 % accuracy for 26B-A4B. [We applied our Unsloth Dynamic method](https://unsloth.ai/docs/models/gemma-4/qat#qat-analysis) to push it up to **85.6% (+15.6%) whilst also being** [**200MB smaller**](https://unsloth.ai/docs/models/gemma-4/qat#usage-guide)!

Gemma 4 QAT includes: **E2B**, **E4B**, **12B, 26B-A4B**, and **31B.** They are multimodal, hybrid-thinking models that support 140+ languages and up to **256K context.**

[Run Gemma 4 QAT](https://unsloth.ai/docs/models/gemma-4/qat#run-gemma-4-qat-tutorials)[Fine-tune Gemma 4](https://unsloth.ai/docs/models/gemma-4/train)

**Gemma-4-E2B** QAT runs on 3GB RAM, **E4B** on 5GB**, 12B** on 7GB**, 26-A4B** on 15GB and **31B** on 18GB**.**

We name our Gemma 4 QAT GGUFs as `UD-Q4_K_XL` as we found q4\_0 to degrade accuracy despite being bigger. See our [Gemma 4 QAT GGUFs](https://huggingface.co/collections/unsloth/gemma-4-qat).

To compare `int4` quantization, see the original vs. QAT size differences below. QAT uses ~72% less memory whilst retaining nearly all its original accuracy:

### 

Usage Guide

Gemma 4 QAT variants for E2B and E4B are designed for phones and laptops, while the larger 26B-A4B and 31B QAT models now work on laptops rather than just strong home GPUs.

There is **only one GGUF file** for each Gemma 4 model because we found that precisions higher than the uploaded `UD-Q4_K_XL` version degrade accuracy rather than improve it. Use the original non QAT Q4\_0 quants [here](https://huggingface.co/collections/unsloth/gemma-4).

### 

Hardware requirements

**Table: Gemma 4 QAT Inference GGUF recommended hardware requirements** (units = total memory: RAM + VRAM, or unified memory).

### 

Recommended Settings

The QAT checkpoints use the same recommended Gemma 4 settings:

-   `temperature = 1.0`
    
-   `top_p = 0.95`
    
-   `top_k = 64`
    

Gemma 4's max context is **128K** for **E2B**, **E4B** and **256K** for **12B**, **26B A4B**, **31B**.

## 

QAT Analysis

We found that naively converting the QAT Q4\_0 checkpoint to Q4\_0 in llama.cpp land actually degraded accuracy and was not actually aligned with the BF16 QAT lattice for Q4\_0. We applied our Unsloth dynamic method to force a better agreement between the llama.cpp compatible Q4\_0 format and the true BF16 QAT Q4\_0 format, and managed to both make the quants smaller (Q6\_K wasn't needed for embeddings), and also more accurate!

Below is a table of KLD and Top 1% accuracy and Disk space. You can see our versions dramatically improve on 99.9% KLD and mean KLD. **E2B for example has a mean KLD of 0.00173 vs 0.05109 (29x better relatively) for a naive Q4\_0 quantization, and ours is even 22% smaller!**

The main issue is converting from QAT BF16 to llama.cpp's Q4\_0 format is not lossless. llama.cpp uses F16 scales, whilst QAT BF16 uses BF16 scales, and the scales are not determined optimally in llama.cpp land.

Naive conversion gets 24.77% byte exactness to BF16 QAT, whilst we found we can push it to 99.96% using some hacks!

## 

Mobile Mixture QAT

The Gemma-4 team also released special mobile mixture QAT versions of Gemma-4-E2B-it and Gemma-4-E4B-it. We also faithfully converted them to llama.cpp compatible format, and also recovered nearly all accuracy as well. We used TQ2\_0 for the 2-bit layers and did a negative scaler.

We made UD-Q2\_K\_XL quants for both E2B and E4B.

See [gemma-4-E2B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF) and [gemma-4-E4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF) for `UD-Q2_K_XL`.

## 

Run Gemma 4 QAT Tutorials

Because Gemma 4 GGUFs comes in several sizes, the recommended starting point for the small models is 8-bit and the larger models is **Dynamic 4-bit**. [Gemma 4 GGUFs](https://huggingface.co/collections/unsloth/gemma-4-qat):

[🦥 Unsloth Studio Guide](https://unsloth.ai/docs/models/gemma-4/qat#unsloth-studio-guide)[🦙 Llama.cpp Guide](https://unsloth.ai/docs/models/gemma-4/qat#llama.cpp-guide)

**You can run and train Gemma 4 QAT for free with a UI in our** [**Unsloth Studio**](https://unsloth.ai/docs/new/studio)✨ **notebook:**

### 

🦥 Unsloth Studio Guide

Gemma 4 QAT can now be run and trained in [Unsloth Studio](https://unsloth.ai/docs/new/studio), our new open-source web UI for local AI. Unsloth Studio lets you run models locally on **MacOS, Windows**, Linux and:

-   Search, download, [run GGUFs](https://unsloth.ai/docs/new/studio#run-models-locally) and safetensor models
    

-   Fast CPU + GPU inference via llama.cpp
    

1

#### 

Install Unsloth

Run in your terminal:

**MacOS, Linux, WSL:**

**Windows PowerShell:**

2

#### 

Launch Unsloth

**MacOS, Linux, WSL and Windows:**

Then open `http://127.0.0.1:8888` (or your specific URL) in your browser.

3

#### 

Search and download Gemma 4 QAT

On first launch you will need to create a password to secure your account and sign in again.

Then go to the [Studio Chat](https://unsloth.ai/docs/new/studio/chat) tab and search for Gemma 4 in the search bar and download your desired model and quant.

4

#### 

Run Gemma 4 QAT

Inference parameters should be auto-set when using Unsloth Studio, however you can still change it manually. You can also edit the context length, chat template and other settings.

For more information, you can view our [Unsloth Studio inference guide](https://unsloth.ai/docs/new/studio/chat).

### 

🦙 Llama.cpp Guide

For this guide there is no need to select quantization type since there is only one: `UD-Q4_K_XL`. See: [Gemma 4 QAT collection](https://huggingface.co/collections/unsloth/gemma-4-qat). For these tutorials, we will using [llama.cpp](llama.cpphttps://github.com/ggml-org/llama.cpp) for fast local inference, especially if you have a CPU.

1

Obtain the latest `llama.cpp` **on** [**GitHub here**](https://github.com/ggml-org/llama.cpp). You can follow the build instructions below as well. Change `-DGGML_CUDA=ON` to `-DGGML_CUDA=OFF` if you don't have a GPU or just want CPU inference. **For Apple Mac / Metal devices**, set `-DGGML_CUDA=OFF` then continue as usual - Metal support is on by default.

2

If you want to use `llama.cpp` directly to load models, you can follow commands below, according to each model. `UD-Q4_K_XL` is the ONLY quantization type. You can also download via Hugging Face (step 3). This is similar to `ollama run` . Use `export LLAMA_CACHE="folder"` to force `llama.cpp` to save to a specific location.

**26B-A4B:**

**31B:**

**E4B:**

**E2B:**

3

Download the model via (after installing `pip install huggingface_hub hf_transfer` ). You can choose `UD-Q4_K_XL` or other quantized versions like `Q8_0` . If downloads get stuck, see: [Hugging Face Hub, XET debugging](https://unsloth.ai/docs/basics/troubleshooting-and-faqs/hugging-face-hub-xet-debugging)

4

Then run the model in conversation mode (with vision `mmproj-F16`):

5

### 

Llama-server deployment

To deploy Gemma-4 on llama-server, use:

To [disable thinking / reasoning](https://unsloth.ai/docs/models/gemma-4/qat#how-to-enable-or-disable-reasoning-and-thinking), use `--chat-template-kwargs '{"enable_thinking":false}'`

If you're on **Windows** Powershell, use: `--chat-template-kwargs "{\"enable_thinking\":false}"`

Use 'true' and 'false' interchangeably.

Last updated 12 hours ago

Was this helpful?