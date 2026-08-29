# MTP TOP-K patch — the self-check is broken on vLLM 0.26 (found 2026-08-26)

**Status: worked around by bind-mount. The PATCHER STILL NEEDS FIXING.**
The override itself is correct. Only its verification line is wrong — but it
was wrong in the direction that KILLS A VALID EXPERIMENT.

## Symptom

First launch of `qwen3-122b-r7kl-mtpk.sh` on `vllm-node-20260824-hybrid`:

```
MTP TOP-K OVERRIDE ACTIVE: draft head top_k=8 (target top_k=4)
MTP EFFECTIVE TOP-K: draft_head=None target_config=4
!!! NO-OP OR MISMATCH: asked for 8, engine reports '<nothing>'
```

The two lines contradict each other. The override HAD applied; the probe could
not see it, so the launcher's gate aborted a good run.

## Cause

The probe in `qwen3_5_mtp.py` reads:

```python
_eff = getattr(_blk, "top_k", None)
if _eff is None and hasattr(_blk, "experts"):
    _eff = getattr(_blk.experts, "top_k", None)
```

**`FusedMoE` in this build defines no `top_k` attribute at all.**
`grep -n "self\.top_k" vllm/model_executor/layers/fused_moe/layer.py` → nothing.
The value is passed to `FusedMoEFactory(top_k=...)` and stored on a config
dataclass:

- `fused_moe/layer.py:333` — `moe_config = FusedMoEConfig(experts_per_token=top_k, ...)`
- `fused_moe/config.py:1284` — `experts_per_token: int`
- the router also gets its own copy (`create_fused_moe_router(top_k=...)`)

So the canonical accessor is **`block.experts.moe_config.experts_per_token`**.

The override itself is unaffected: `qwen3_next.py:169` reads
`top_k=config.num_experts_per_tok` at CONSTRUCTION time, and the patch mutates
that config before the `ModuleList` is built and restores it in a `finally`.

## The fix to fold into the patcher

Replace the probe with a fallback chain, most-canonical first:

```python
_exp = getattr(_blk, "experts", None)
for _obj, _attr in ((getattr(_exp, "moe_config", None), "experts_per_token"),
                    (getattr(_exp, "router", None), "top_k"),
                    (_exp, "top_k"),
                    (_blk, "top_k")):
    if _obj is not None:
        _eff = getattr(_obj, _attr, None)
        if _eff is not None:
            break
```

Working copy: `vllm_node_patch/overrides/qwen3_5_mtp.py` (extracted from the
image, probe replaced, `ast.parse` clean). The launcher currently bind-mounts it
read-only over
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py`.
**The image is NOT modified**, so folding this into the patcher and rebuilding
supersedes the mount — drop the `-v` line from the launcher when that happens.

## The lesson, which is the reverse of the usual one

Every prior instance in LESSONS.md is a gate that PASSED while the thing it
guarded was broken. This is the mirror image: a gate that FAILED while the thing
it guarded worked. Both come from the check and the mechanism reading different
state. Here the mechanism wrote `config.num_experts_per_tok` and the check read
`FusedMoE.top_k` — two different objects, coupled only by a constructor
argument, and that coupling changed between vLLM versions while the check did not.

**A verifier that reads a different object than the mechanism writes is a
version-coupling bug waiting to happen.** Prefer asserting on the value the
mechanism actually set, or on observable behaviour, over introspecting a
downstream object's attribute name.
