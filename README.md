# DGX Spark hybrid vLLM patches

Patches that let vLLM 0.26.x serve a **hybrid INT4 + FP8** Qwen3.5-MoE
checkpoint on a DGX Spark (GB10 / SM121), with a working MTP draft head.

Take an existing vLLM base image, run one script, get a new patched image.
The base is never modified.

## Prerequisite: build a vLLM base image

These patches are a thin layer on top of a working SM121 vLLM build. **Build
that first**, using [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
and its own instructions. Nothing here rebuilds it.

Verified against a base at vLLM `0.26.1rc1` (`build_args: vllm_ref=main,
transformers_5=true`, `gpu_arch 12.1a`).

## Build

```bash
git clone https://github.com/azampatti/vllm-hybrid-int4-fp8-patches.git && cd vllm-hybrid-int4-fp8-patches
bash ./hybridpatch.sh <your-base-image-tag>          # -> <base-repo>-hybrid:latest
bash ./hybridpatch.sh <your-base-image-tag> -t my-vllm:v1
```

The base image is read-only, and the script refuses to overwrite an existing
output tag without `--force`.

## What gets patched

| # | File | Fixes |
|---|---|---|
| 01 | `quantization/inc/inc.py` | Hybrid INT4+FP8 dispatch, so FP8 dense layers (shared expert, linear attention) aren't silently loaded unquantized |
| 03 | `layers/logits_processor.py` | INT8 Triton GEMV fast path for the LM head |
| 04 | `quantization/inc/config_parser.py` | vLLM strips the `mtp.` prefix when building the draft model, so head layers matched no declared block and load died on packed `.qweight` |
| 05 | `config/speculative.py` | Carries `quantization_config` into the MTP draft config on VL-wrapper checkpoints, where it lives only on the outer config |
| 06 | `models/qwen3_5_mtp.py` | Adds `VLLM_MTP_TOP_K`, letting the draft head route to more experts than the target model. Not an upstream vLLM variable; inert unless set |

Every patch is anchored and idempotent: an anchor that matches zero or more
than one time exits non-zero rather than half-applying, each patched file is
byte-compiled, and the build fails if `vllm.entrypoints.openai.api_server`
does not import. Re-running reports SKIP.

## Also here

- `REPRODUCE.md` — full runbook: launch flags, verification gates, and the two
  failure modes (stale tensors, unverified patches) that cost the most time.
- `check_stale_tensors.py` — **run this before debugging any vLLM ≥ 0.26 load
  failure.** Dropping a key from `model.safetensors.index.json` does not remove
  the tensor from the shard; newer vLLM reads the files and dies.
- `probe_quant_resolution.py` — predicts INC quant resolution in ~5s without
  loading 65 GB of weights.
- `overrides/qwen3_5_mtp.py` — full-file version of patch 06, for bind-mounting.
  See `OVERRIDE_MTP_TOPK_VERIFIER.md`; its self-check probe is known-broken on
  0.26 and can abort a valid run.
- `docker/Dockerfile.v3` — what `hybridpatch.sh` builds.
- `PORTING_NOTES.md` — how each patch was retargeted from vLLM 0.19.1, and what
  upstream changed in between.

## License

Apache-2.0, matching vLLM. These are derived patches against vLLM 0.26.1rc1.
