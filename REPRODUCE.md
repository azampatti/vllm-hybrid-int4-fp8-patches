# Reproducing the hybrid image — a fast, modern vLLM for qwen3.5-122b-a7b-int4

**Goal:** a vLLM **0.26.x** image that serves a hybrid INT4+FP8 Qwen3.5-MoE
checkpoint at the same speed as the old patched **0.19.1** image, instead of the
~30-35% slower unpatched build.

**Measured on DGX Spark GB10, single-stream benchmark script, LongCode tok/s:**

| image | vLLM | LongCode tok/s |
|---|---|---|
| patched 0.19.1 image | 0.19.1 | 71.0 |
| `vllm-node-20260730` (eugr base, unpatched) | 0.26.1rc1 | 51–53 |
| **hybrid image (this document)** | **0.26.1rc1** | **72.3 / 71.9** |

MTP speculative decoding stays healthy: mean acceptance length **3.79**, average
draft acceptance **93.1%**.

> **Two failure modes account for essentially all the pain here.** Read §5
> (stale tensors) and §7 (verification gates) before you start. §5 in particular
> is a **weight-file** problem that no image, flag, or config change can fix,
> and it produces an error message that reads like an engine bug.

---

## 1. Prerequisites

- DGX Spark / GB10 (SM121, `TORCH_CUDA_ARCH_LIST=12.1a`)
- Docker with `--gpus all`
- ~25 GB free disk for the image, plus ~65 GB per materialized model copy
- Your checkpoint directory (shards + index + tokenizer; this reference build
  used 49 shards plus `model_extra_tensors.safetensors` and
  `model_shared_expert_bf16.safetensors`)
- The patch tree from this repository

---

## 2. Step 1 — Base image from eugr's repo

Build the base per **eugr/spark-vllm-docker**'s own instructions. This document
assumes that image already exists; nothing here rebuilds it.

The reference build used:

```
build_script_commit: c1f31ebfb26f33a6fb122ed59c0cae6498b45070
vllm_version:        0.26.1rc1.dev30+g5773c4e60.d20260728
vllm_commit:         5773c4e60
flashinfer_commit:   2deed6c1
gpu_arch:            12.1a
build_args:          {vllm_ref: main, transformers_5: true, build_jobs: 16}
```

Verify your base and record these values — if a patch anchor later fails, this
is the first thing to compare:

```bash
docker run --rm --entrypoint cat <BASE_IMAGE> /workspace/build-metadata.yaml
```

**Do not** build the old `vllm-sm121` 0.19 base. That is the v2 path and is not
what this document reproduces.

> **The base image tag is consumed read-only and the result is written to a new
> tag.** No existing image is modified. Verify with `docker images` before and
> after — the base image ID must be unchanged.

---

## 3. Step 2 — Get the patch tree

You need this layout (all files are in `Prunning/vllm_node_patch/`):

```
vllm_node_patch/
├── hybridpatch.sh              <- one-shot builder (Step 3)
├── docker/Dockerfile.v3
├── patches/
│   ├── 01-hybrid-int4-fp8/patch_inc_hybrid.py
│   ├── 03-int8-lm-head/patch_int8_lmhead_v3.py
│   ├── 04-mtp-block-scope/patch_mtp_block_scope.py
│   └── 05-mtp-quant-config/patch_mtp_quant_config.py
├── check_stale_tensors.py      <- run BEFORE serving (Step 4)
└── probe_quant_resolution.py
```

`hybridpatch.sh` resolves its own directory, so it can be invoked by absolute
path from anywhere.

Every patch is an **anchored Python editor**, not a `.diff` and not a whole-file
replacement. Each one:

- is **idempotent** (re-running prints `SKIP:`)
- **fails loudly** — an anchor matching zero or ≠1 times exits 1 rather than
  half-applying
- `py_compile`s the file it edited before reporting `OK:`

This matters because upstream refactored the files these patches touch. A
whole-file `inc.py` replacement (what the 0.19 recipe did) would clobber the new
`config_parser` / `schemes` wiring and silently break INT4.

---

## 4. Step 3 — Build the thin layer

### The one-shot way (recommended)

```bash
cd Prunning/vllm_node_patch
./hybridpatch.sh vllm-node-20260730
```

That is the whole build. It takes the base image name, applies all four patches,
and writes a new image — by default `<base-repo>-hybrid:latest`. Override with
`-t my-image:v1`.

It refuses to proceed, with an explanation, when:

- the base image does not exist locally
- the base is vLLM ~0.19 (single-file `inc.py`) — wrong patch set
- the **output tag already exists** (pass `--force` to override)
- the output repo equals the base repo — it will not risk the base

After building it re-verifies the result independently of the build's own
asserts: all four patch markers present in the image, `vllm` imports cleanly, and
the base image ID is unchanged. Any failure exits non-zero.

Verified 2026-08-09: `./hybridpatch.sh vllm-node-20260730` produces an image
**byte-identical** (`f76465fef071`) to the image that measured 72.3
tok/s.

### The manual way

```bash
cd Prunning/vllm_node_patch
docker build -t <your-hybrid-image> -f docker/Dockerfile.v3 .
```

Takes ~15 seconds. Pass a different base with
`--build-arg VLLM_BASE=<your-base-tag>`.

Expected output — **all four `OK:` lines must appear**:

```
OK: hybrid INT4+FP8 dispatch applied (6 edits)
OK: INT8 LM Head v2 patch applied (_apply_head host site)
OK: MTP block-scope fix applied
OK: MTP quant-config propagation applied
v3 patched image: imports OK
```

The Dockerfile additionally `grep`s for each patch's marker and imports
`vllm.entrypoints.openai.api_server`, so a drifted anchor **breaks the build**
rather than shipping a silently unpatched image.

### What each patch does

| # | File patched | Why |
|---|---|---|
| **01** | `.../quantization/inc/inc.py` | Hybrid INT4+FP8 dispatch: sends dense `shared_expert` / `linear_attn` layers to `Fp8LinearMethod` instead of dropping them to unquantized. Port of albond patch 01. |
| **03** | `.../layers/logits_processor.py` | INT8 LM head v2 (Triton GEMV, autotuned). Port of albond patch 03. |
| **04** | `.../quantization/inc/config_parser.py` | vLLM strips the `mtp.` prefix when building the draft head, so its layers match no entry in `block_name_to_quantize` and resolve as unquantized. Re-qualifies the name. |
| **05** | `.../config/speculative.py` | On VL-wrapper checkpoints `quantization_config` sits on the **outer** config, but the draft is built from the promoted `text_config`, which never receives it. Upstream already does this carry-over for `step3p5` and `minimax_m3_vl`; the `qwen3_5` branch was missed. |

**Honesty about scope.** Patches **01, 04 and 05 are inert on this particular
checkpoint** and are *not* what produced the speed:

- **01** dispatches **FP8** dense layers. This checkpoint is INT4 + **BF16**
  shared expert — there are no `float8_e4m3fn` tensors, so detection finds
  nothing. Keep it: it is required for albond's INT4+FP8 hybrid checkpoint, and
  it is the patch that mattered on the 0.19 image.
- **04 / 05** were written while diagnosing a load failure that turned out to be
  §5 (stale weight files), not an engine bug. They fix real, independently
  verified defects, but **an ablation has not been run**, so it is not known
  whether this checkpoint loads without them.

They are all included because they are what the measured image contains. To
reproduce the numbers exactly, include them. Removing them is untested.

---

## 5. Step 4 — ⚠ Prepare the weights (the step that actually blocks you)

**Run this before anything else touches the model.** It takes ~5 seconds.

```bash
python3 check_stale_tensors.py ~/models/<model-dir>
```

Required output:

```
STALE (in files, NOT indexed): 0
MISSING (indexed, not in files): 0
OK: clean. Every tensor in the files is referenced by the index.
```

### Why this exists

Removing a key from `model.safetensors.index.json` does **not** remove the
tensor from the shard file. When a BF16 shared expert ships, the exporter drops
the 432 old packed keys (`qweight`/`qzeros`/`scales`) from the index — but they
are still physically in the files.

- **vLLM 0.19** is index-driven. It never sees them. Works.
- **vLLM 0.26** enumerates tensors from the **FILES** via `fastsafetensors`,
  finds `model.language_model.layers.0.mlp.shared_expert.down_proj.qweight`, and
  tries to load it into a module the config correctly declares BF16:

```
ValueError: There is no module or parameter named
'layers.0.mlp.shared_expert.down_proj.qweight' in Qwen3_5Model.
The available parameters belonging to layers.0.mlp.shared_expert.down_proj
(RowParallelLinear) are: {'layers.0.mlp.shared_expert.down_proj.weight'}
```

**Two traps in that message:**

1. It names **`Qwen3_5Model`** — the **MAIN** model. It is *not* the MTP head,
   even though a genuine draft-head failure prints an almost identical string.
   Chasing the draft head here costs hours.
2. **No flag, config edit, or engine patch fixes it.** Adding `mtp.`-prefixed
   `bits=4` entries to `extra_config` does nothing, and it reproduces with MTP
   disabled entirely. It is a property of the weight files.

### The fix

Materialize shards containing exactly what the index references:

```bash
python3 materialize_clean_shards.py            # your own re-write script; ~2 min
python3 check_stale_tensors.py ~/models/<clean-dir>   # must print STALE: 0
```

That script writes `<shard>.tmp` and atomically replaces each file one at a
time, skips files already materialized, and never touches the source directory.

Reference values from the build measured here:

| directory | indexed | in files | stale | loads on 0.26 |
|---|---|---|---|---|
| original directory | 114451 | 114884 | **432** | ❌ |
| re-materialized copy | 114451 | 114451 | **0** | ✅ |

Weights are otherwise byte-identical, so quality comparability is preserved.

> Also: **symlink targets must resolve inside the container.** Keep them within
> `${HOME}/models` → `/models`. Links written relative to the HF cache resolve on
> the host but not in the container, and the symptom is a misleading
> `Can't load image processor ... preprocessor_config.json` error.

### Optional: predict quantization resolution without loading 65 GB

```bash
docker run --rm -v ~/models:/models:ro \
  -v $PWD/probe_quant_resolution.py:/tmp/p.py:ro \
  --entrypoint bash <your-hybrid-image>:latest -c 'python3 /tmp/p.py /models/<dir>'
```

Expected — note the **last** row especially:

```
DRAFT  layers.0.mlp.shared_expert.down_proj    bits= 4 quantized=True
DRAFT  layers.0.mlp.shared_expert_gate         bits=16 quantized=False
MAIN   ...layers.0.mlp.experts.w2_weight       bits= 4 quantized=True
MAIN   ...layers.0.mlp.shared_expert.down_proj bits=16 quantized=False
```

If that last row flips to `bits=4`, the trained BF16 expert is being quantized
away at load time and any quality number is meaningless.

---

## 6. Step 5 — Launch

Raw launch command:

```bash
docker run --privileged --gpus all -d --name vllm-hybrid \
  --net=host --ipc=host \
  -v "${HOME}/models:/models" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm-v3:/root/.cache/vllm" \
  -v "${HOME}/dgx-spark-sglang-moe-configs:/root/.cache/dgx" \
  -v "${HOME}/spark-vllm-docker/mods/fix-qwen3.5-enhanced-chat-template/qwen3.5-enhanced.jinja:/workspace/qwen3.5-enhanced.jinja:ro" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  <your-hybrid-image> \
  vllm serve /models/<your-model-dir> \
  --served-model-name qwen3.5-122b-a7b-int4 \
  --port 8000 --host 0.0.0.0 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}' \
  --max-model-len 200K \
  --gpu-memory-utilization 0.75 \
  --load-format fastsafetensors \
  --attention-backend TRITON_ATTN \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --override-generation-config '{"temperature": 0.5, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}' \
  --tool-call-parser qwen3_coder \
  --chat-template /workspace/qwen3.5-enhanced.jinja
```

### Four things that differ from the 0.19 launcher — all forced by the base

1. **`vllm serve`, not `serve`.** This image's `ENTRYPOINT` is
   `/opt/nvidia/nvidia_entrypoint.sh`, not `["vllm"]`. A bare `serve` resolves to
   **Ray Serve's** CLI, which is also installed, and fails with a misleading
   `No such command '/models/...'`.
2. **Its own vLLM cache** (`~/.cache/vllm-v3`). Sharing a cache across vLLM
   versions risks reusing stale `torch.compile` artifacts from 0.19.
3. **`TRITON_ATTN`, pinned in *both* places.** `--attention-backend` reaches only
   the target model; the draft picks its own unless
   `SpeculativeConfig.attention_backend` is set in the `--speculative-config`
   JSON. With only the CLI flag, the draft silently selects FlashInfer.
4. **Not `--rm`.** A crashed engine takes its logs with it, and the root cause
   scrolls past the `RuntimeError` wrapper vLLM prints last. Remove the container
   on the *next* launch instead.

### Attention backend

`TRITON_ATTN` is the default because it measured better on every workload and
declares `AttentionCGSupport.ALWAYS`, so it avoids this downgrade:

```
WARNING [compilation.py:1459] CUDAGraphMode.FULL_AND_PIECEWISE is not supported
with spec-decode for attention backend FlashInferBackend ...
setting cudagraph_mode=PIECEWISE
```

| workload | FLASHINFER | TRITON_ATTN |
|---|---|---|
| LongCode | 71.9 / 71.1 | **72.3 / 71.9** |
| JSON | 63.8 / 61.7 | **65.9 / 64.4** |
| Math | 57.1 / 57.6 | **65.9 / 62.7** |

**But do not over-credit this.** The PIECEWISE downgrade was *not* the dominant
cost — removing it bought ~1% on LongCode, not the ~28-30% its mechanism
suggested. The patch set is what closed the gap. Use
`ATTN_BACKEND=FLASHINFER` for strict 0.19 flag parity.

---

## 7. Step 6 — Verification gates

Run all four. Each one has caught a real silent failure.

**Gate 1 — the server is up**

```bash
curl -s http://localhost:8000/health -o /dev/null -w "%{http_code}\n"   # 200
docker logs vllm-hybrid 2>&1 | grep "Application startup complete"
```

**Gate 2 — the trained BF16 expert actually loaded** (the important one)

```bash
docker logs vllm-hybrid 2>&1 | grep "Model loading took"
# Model loading took 64.16 GiB memory and 55.04 seconds
```

Must be **~64.1-64.2 GiB** (observed 64.13 and 64.16 across two launches),
above the **63.27 GiB** all-INT4 baseline. If it reports
~63.3, the BF16 shared expert did not load and you are benchmarking the
**untrained** model. Every quality number would be void.

**Gate 3 — MTP is live**

```bash
docker logs vllm-hybrid 2>&1 | grep "SpecDecoding metrics" | tail -1
# Mean acceptance length: 3.79 ... Avg Draft acceptance rate: 93.1%
```

Absent entirely ⇒ speculative decoding is off and throughput will be ~26 tok/s
instead of ~70.

**Gate 4 — no CUDA-graph downgrade**

```bash
docker logs vllm-hybrid 2>&1 | grep -c "setting cudagraph_mode=PIECEWISE"
# 0 with TRITON_ATTN, 1 with FLASHINFER
```

---

## 8. Step 7 — Benchmark

```bash
bash <your-benchmark-script>.sh
```

Expected steady-state (`Streams: 1`):

```
[Code]     ~68-70 tok/s
[JSON]     ~64-66 tok/s
[Math]     ~63-66 tok/s
[LongCode] ~72 tok/s        <- the headline number
```

**Ignore the first `[Q&A]` line of run 1** (~24-26 tok/s). That is cold-start
JIT, not a regression; run 2 recovers to ~59.

**These are throughput numbers only.** They say nothing about quality. Quality on
this project needs a **10-run trimmed** mean — 3 runs produced a 2.8pp false
positive twice. Do not quote a quality figure for this image until that is run.

---

## 9. What did NOT work (don't re-run these)

| attempt | result |
|---|---|
| INT8 shared expert (AutoRound) to remove the BF16 exception | **+1.9%** only |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | **0** |
| FlashInfer vs Triton attention on the *unpatched* node image | both ~50 |
| Adding `mtp.`-prefixed `bits=4` entries to `extra_config` | **nothing** — the lookup only ever sees the `mtp.`-stripped name |
| Re-pinning `starlette<1.0` (the 0.19 fix) | **harmful** — 0.26 runs on Starlette 1.3.1 natively |

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no module or parameter named '...shared_expert.down_proj.qweight' in Qwen3_5Model` | **Stale tensors in the weight FILES.** §5. Not an engine bug, not the MTP head. |
| `FAIL: anchor ... matched 0 times` during build | Base vLLM drifted. Compare `build-metadata.yaml` against §2 and re-derive that patch. |
| `No such command '/models/...'` | You used `serve` instead of `vllm serve`; Ray Serve's CLI shadowed it. |
| Model load reports ~63.3 GiB | BF16 shared expert did not load — wrong directory or wrong `extra_config`. |
| ~26 tok/s instead of ~70 | MTP off — check `--speculative-config` and Gate 3. |
| ~50 tok/s | Patches missing, or you are on the unpatched base image. |
| `Failed to infer device type` | You forgot `--gpus all`. |
| Engine dies, logs gone | You used `--rm`. |
| `cicc died due to signal 9` during FlashInfer JIT | Set `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1`. |

---

## 11. Provenance

- Base: eugr/spark-vllm-docker `c1f31eb`, vLLM `0.26.1rc1.dev30+g5773c4e60`
- Patches 01/03 ported from albond
  `DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/patches/`; 04/05 written for 0.26
- Recipe lineage: drewid74 `optimized-qwen35-hybrid-v2-runbook-public`, Phase 1
  Step 4 (the "thin layer"); Step 3 is replaced by eugr's modern base
- Measured on DGX Spark GB10, single-stream benchmark script
- Page-cache flush (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) was
  **skipped** in the reference run (no passwordless sudo). Cold-load timings are
  therefore pessimistic; steady-state throughput is unaffected.
