# Qwen3.8-27B on a 12 GB gaming GPU: what each gigabyte buys you

I run **Mericanii**, an independent one-person research lab; this is our work.

> **STATUS: INTERNAL DRAFT — NOT PUBLISHED.** Links below are placeholders.
> Do not push this repo public until the checklist in `STRUCTURE.md` is clear.

**What this is:** the first independent **end-to-end task-accuracy**
measurements we can find of running a top-tier open coding model on ordinary
gaming GPUs. (Independent KL-divergence and perplexity measurements of this
exact model already exist — see [PAPER.md §1.1](PAPER.md) for how ours
differs.) We ran four sub-4-bit builds of **Qwen3.8-27B** on one **RTX 3060
12GB** — a card that costs about $250 used — and scored every one of them on
the **complete 164-task HumanEval** through the same harness. Everybody
publishes quantized GGUF files; almost nobody publishes what they cost you in
executed, graded code. Now somebody has.

Coding ability rises **49.4% → 82.3%** as the file grows 6.76 → 10.17 GiB —
approximately **10 HumanEval points per gigabyte across the measured range,
every adjacent step statistically significant under paired testing** (exact
McNemar test, p < 0.05 for all three steps; see [PAPER.md §3](PAPER.md)).
Every claim here is backed by **[PAPER.md](PAPER.md)** — read that for the
numbers, the method, and everything we got wrong.

---

## ⭐ Recommended setup — the flagship

> **UD-IQ3_XXS + built-in MTP speculation (n=2) + q8_0 KV cache, 16K context,
> on a 12 GB card.**
>
> **82.3% HumanEval · ~34 tok/s · 11642 MiB peak VRAM · fully on the GPU.**
>
> This is the best coding quality we could fit on a 12 GB card, and it needs a
> **headless GPU** — with a desktop session on the same card it will not fit.

```bash
# 1. weights (~10.2 GiB) and the fixed chat template
hf download unsloth/Qwen3.8-27B-GGUF \
  --include "*UD-IQ3_XXS*.gguf" --local-dir ./models
hf download froggeric/Qwen-Fixed-Chat-Templates \
  --include "chat_template.jinja" --local-dir ./models

# 2. serve
llama-server \
  -m ./models/Qwen3.8-27B-UD-IQ3_XXS.gguf \
  -ngl 99 -fa on -c 16384 \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 \
  --jinja --chat-template-file ./models/chat_template.jinja \
  --host 127.0.0.1 --port 8080
```

**Before you run it:**

- **Display attached, or want more context?** Use **UD-Q2_K_XL + MTP n=3** —
  72.0% HumanEval, 35.4 tok/s, 10740 MiB. Ten points cheaper, ~900 MiB roomier.
- `--spec-draft-n-max` **2 or 3, never 4** — acceptance collapses at n=4 and it
  is net slower.
- **Do not** download the separate external MTP file: +776 MiB for performance
  identical to the head already inside the GGUF.
- The stock chat template in these GGUFs is broken for this family. Use the
  fixed one above.
- Set `reasoning_effort` deliberately, or the model will spend 500–600 tokens
  on grade-school arithmetic.

---

## Headline results

RTX 3060 12GB · llama.cpp b10566 (`bb4caa754`) · `unsloth/Qwen3.8-27B-GGUF` ·
HumanEval 164 tasks, pass@1, temperature 0 · all builds fully resident.

| Build | bpw | File (GiB) | HumanEval (164) | Decode (tok/s) | Built-in MTP head? |
|---|---|---|---|---|---|
| UD-IQ2_XXS | 2.12 | 6.76 | 49.4% | 23.33 | ❌ stripped |
| UD-IQ2_S | 2.45 | 7.79 | 62.2% | 22.48 | ❌ stripped |
| UD-Q2_K_XL | 2.87 | 9.14 | 72.0% | 21.75 (35.4 with MTP) | ✅ |
| **UD-IQ3_XXS** | **3.20** | **10.17** | **82.3%** | **20.69 (33.8 with MTP)** | ✅ |

Decode column: `llama-bench` tg128 with no speculation. Parenthesised figures
are live-serving decode with the built-in MTP head at 16K context and a q8_0 KV
cache — a different measurement condition, not a delta on the same run
([PAPER.md §4.1](PAPER.md)).

**≈10 HumanEval points per GiB across the measured range, every adjacent step
significant under paired testing** (§3 has the confidence intervals and the
McNemar tests). On a 12 GB card, run the largest build that still fits the
context you need. Three more results worth knowing:

- **The built-in MTP speculation head is worth +68%, losslessly** — and builds
  below about 8.4 GiB silently ship without it. Going small is penalized twice.
- **No decode cliff to 98K context** (21.05 → 21.02 tok/s), and **KV cache
  precision costs zero speed** — quantize the cache, not the weights. Both
  measured against an allocated, near-empty cache; filled-cache re-measurement
  is pending ([PAPER.md §4.2–4.3](PAPER.md)).
- **Best-of-5 at temp 0.8 loses to one greedy attempt**: oracle ceiling 75.0%
  against an 82.3% baseline. Test-time compute is not free at 3.2 bpw. We ran
  it against pre-registered gates, and the gates killed the follow-up run.

---

## Limitations, up front

HumanEval only, and HumanEval is substantially quant-blind — treat our curve as
a **lower bound** on quantization damage. **No full-precision anchor**: a BF16
27B does not fit on a 3060, so our numbers are internally comparable but not
leaderboard-comparable, and our chat-template prompting protocol makes them
doubly so. Single seed, one run per point, one card, one runtime. **Agentic
performance is entirely unmeasured**, and the literature says that is precisely
where quantization damage concentrates. The full list is §7 of the paper, and
it is longer than this paragraph.

---

## What is in this repo

| Path | Contents |
|---|---|
| `PAPER.md` | The paper: full curve, speculation ablations, the pre-registered negative result, limitations, reproducibility. Start here. |
| `harness/` | The measurement harness. Python 3 stdlib + `requests`; needs `llama-server` and `llama-bench` on `$PATH`. |
| `configs/` | Config matrices for the sweep runner. |
| `results/full164/` | Raw per-run JSON and per-task pass/fail behind every number. |
| `results/full164/model_hashes.txt` | SHA-256 of every GGUF we measured. |
| `docs/` | Caveats and an FAQ: contamination, seeding, prompting protocol. |
| `STRUCTURE.md` | *(internal draft only — not part of the public repo)* |

Reproduce the headline number:

```bash
llama-server ... &                  # flagship config, as above
python3 harness/eval_quality.py --endpoint http://127.0.0.1:8080 \
  --limit-humaneval 164 --out results/humaneval_iq3xxs.json
```

---

## This is an open, ongoing effort

We will keep measuring, and we would rather be corrected than quoted wrongly.
Corrections, questions and suggestions are welcome — open an issue with the
config you ran and the output you got, or email **19thkingisreal@gmail.com**.
Corrections get recorded in the repo, not quietly edited into the text.

## Links

- The paper — [PAPER.md](PAPER.md)
- Weights (upstream, **not rehosted**) — <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- Fixed chat template — <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates>
- Base model — <https://huggingface.co/Qwen/Qwen3.8-27B>
- Inference runtime — <https://github.com/ggml-org/llama.cpp>
- Discussion thread — `[PLACEHOLDER: r/LocalLLaMA post URL]`
- Hugging Face community post — `[PLACEHOLDER: HF discussion URL]`
- Mericanii — `[PLACEHOLDER: site URL]`

## License

Our code and this paper: **Apache-2.0** (see `LICENSE`). **Model weights are not
redistributed here** — Qwen3.8-27B is Apache-2.0 and the GGUF builds we measured
are Unsloth's; both are linked above and remain with their upstream publishers.
Benchmark datasets are downloaded at runtime from their original sources under
their own licenses.

Measurements and paper by **Mericanii (Yugendren)**. Quantized builds by
Unsloth; fixed chat template by froggeric; base model by Qwen; inference by
llama.cpp.

**Citing this work:** see `CITATION.cff`.
