# Qwentize.cpp 🥷👾

> **Started:** April 19, 2026
>
> A tutorial on quantizing [llama.cpp](https://github.com/ggml-org/llama.cpp) with **Qwen3-8B** (and Qwen3.6-35B if it doesn't break).

I really wanted to understand **quantization**, as well as get my hands on **llama.cpp**, so inspired by [Nunchaku](https://github.com/mit-han-lab/nunchaku), I decided to take on this mini-research tutorial-style hack!

[@douzog](https://github.com/douzog) 🥷👾

---

## What Is This?

A hands-on benchmark comparing **two quantized Qwen3-8B variants** running locally through llama.cpp.

| Model               | Bits | Avg tokens/sec | Avg prompt latency | Avg RAM | Avg load time |
| ------------------- | ---- | -------------- | ------------------ | ------- | ------------- |
| Qwen3-8B Q4\_K\_M   | 4    | ~14.3          | ~14.5 s            | ~0.89 GB | ~0.9 s        |
| Qwen3-8B Q2\_K      | 2    | ~15.6          | ~13.0 s            | ~0.36 GB | ~0.2 s        |

The notebook loads each quantized variant, runs identical prompts, and measures **speed (tok/s)**, **latency**, **RAM**, and **response quality** — then exports the results to `qwen_benchmark_results.csv` and visualizes them in `qwen_benchmark.png`.

---

## Why Not Nunchaku?

|                  | **llama.cpp / GGUF**                    | **Nunchaku / SVDQuant**                          |
| ---------------- | --------------------------------------- | ------------------------------------------------ |
| Target models    | LLMs (Qwen, Llama, etc.)               | Diffusion models (FLUX, SANA, etc.)              |
| Quantization     | Q4\_K\_M, Q2\_K (GGUF format)          | INT4 / NVFP4 via SVDQuant                        |
| Use case         | Text generation                         | Image generation                                 |
| Install          | Build from source                       | `pip install nunchaku`                            |

> **Note:** Nunchaku is a high-performance inference engine for 4-bit *diffusion* models — it targets image generation, not language models.

---

## Quick Start

### Prerequisites

- **Python 3.10+** with `pip`
- **cmake** (`brew install cmake` on Mac)
- **~20 GB free disk** for the 8B variants (or ~100 GB if you're brave enough for 35B)

### 1 · Clone & Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Apple Silicon (M1/M2/M3/M4) ✅
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(nproc)

# NVIDIA GPU — use this instead:
# cmake -B build -DGGML_CUDA=ON
# cmake --build build --config Release -j$(nproc)
```

### 2 · Download & Quantize Qwen

```bash
# Download from HuggingFace
hf download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b

# Convert to GGUF (full 16-bit)
pip install -r requirements.txt
python convert_hf_to_gguf.py models/qwen3-8b --outfile ../qwen3-8b-f16.gguf

# Quantize
./build/bin/llama-quantize ../qwen3-8b-f16.gguf ../qwen3-8b-q4km.gguf Q4_K_M
./build/bin/llama-quantize ../qwen3-8b-f16.gguf ../qwen3-8b-q2k.gguf  Q2_K
```

You should end up with:

```
qwen3-8b-f16.gguf    (~16 GB)
qwen3-8b-q4km.gguf   (~5 GB)
qwen3-8b-q2k.gguf    (~3 GB)
```

### 3 · Run the Benchmark

```bash
pip install llama-cpp-python psutil matplotlib numpy
jupyter notebook qwen_benchmark.ipynb
```

Run all cells — the notebook will:

1. Load each model variant one-by-one
2. Send 5 benchmark prompts (factual, coding, reasoning)
3. Measure tokens/sec, latency, and RAM
4. Generate comparison charts (`qwen_benchmark.png`)
5. Export raw results to `qwen_benchmark_results.csv`

---

## What Gets Measured

| Metric            | What it tells you                                    |
| ----------------- | ---------------------------------------------------- |
| **Tokens/sec**    | Raw generation speed                                 |
| **Elapsed time**  | Wall-clock time per prompt                           |
| **RAM usage**     | Memory footprint after loading                       |
| **Load time**     | Seconds to load the model into memory                |
| **Response text** | Side-by-side quality comparison at each quant level  |

---

## Quantization Cheat Sheet

```
Q  4  _K_M
│  │   │ │
│  │   │ └── M = Medium (balanced within the K family)
│  │   └──── K = K-quant method (smarter than basic Q4)
│  └──────── 4 = 4 bits per weight
└─────────── Q = Quantized
```

| Format      | Bits | 8B Size | RAM   | Quality   | Best For                |
| ----------- | ---- | ------- | ----- | --------- | ----------------------- |
| F16 (full)  | 16   | ~16 GB  | ~18 GB | Reference | Benchmarking baseline   |
| Q4\_K\_M    | 4    | ~5 GB   | ~6 GB  | ★★★★☆    | Daily use ✅            |
| Q2\_K       | 2    | ~3 GB   | ~4 GB  | ★★☆☆☆    | Edge / very low RAM     |

**What the Q2\_K tradeoff looks like:**

```
Q4_K_M  →  "The mitochondria is the powerhouse of the cell because..."
Q2_K    →  "The mitochondria powerhouse cell energy ATP..."
```

Q2\_K is ~2× smaller but noticeably degrades coherence on complex reasoning tasks.

---

## Repo Structure

```
Qwentify/
├── README.md                  ← you are here
├── qwen_benchmark.ipynb       ← full benchmark notebook
├── qwen_benchmark.png         ← generated chart (after running)
└── qwen_benchmark_results.csv ← generated CSV  (after running)
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `zsh: no such file or directory: ./build/bin/llama-quantize` | Build didn't complete — re-run the cmake steps |
| `cmake` not found | `brew install cmake` (Mac) or `sudo apt install cmake -y` (Linux) |
| Out of memory during quantization | F16 must fit in RAM. For 35B you need ~80 GB. Use a pre-quantized GGUF from HuggingFace instead. |

---

## Resources

- [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [Qwen on HuggingFace](https://huggingface.co/Qwen)
- [GGUF format spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Nunchaku (SVDQuant)](https://github.com/mit-han-lab/nunchaku)

---

<p align="center">
  Built with ☕ and curiosity by <a href="https://github.com/douzog">@douzog</a>
</p>
