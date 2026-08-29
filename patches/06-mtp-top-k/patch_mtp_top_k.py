#!/usr/bin/env python3
"""Draft-head expert-routing width override: VLLM_MTP_TOP_K (vLLM >= 0.26).

WHAT THIS IS
------------
Port of the `vllm-qwen35-v2-mtpk` local patch, which was never carried into the
v3 image -- that is why the measured +3pp was LOST on the move to v3, not
because the env var stopped taking effect. `VLLM_MTP_TOP_K` is NOT an upstream
vLLM variable; without this patch it is silently ignored.

THE MECHANISM
-------------
The Qwen3.5 MTP draft head is a 256-expert MoE layer that reads the SAME global
``config.num_experts_per_tok`` as the target model. Cutting the target to
top-k=4 therefore cut the draft head with it, even though the draft head was
never a training target and has no reason to route narrowly.

The override raises ``num_experts_per_tok`` for the duration of draft-layer
construction only, then restores it in a ``finally``. Everything built after
this point -- and the config object the rest of vLLM reads -- sees the real
value. Unset or 0 => byte-identical to stock.

WHY THE ANCHOR WRAPS MORE THAN THE ModuleList
---------------------------------------------
Upstream added its own save/restore around the same construction (an
``original_quant`` dance for GPTQ checkpoints that exclude MTP via
``quantization_config.dynamic`` "-:mtp" entries). The two restores must NEST,
not race, so the anchor spans that whole region and the top-k restore is placed
outside upstream's. The anchor is byte-identical in vllm-node-20260730 and
vllm-node-20260824, so a single anchor serves both.

`os` is imported locally rather than at module scope: the import block has
drifted between bases and a second edit there would be one more thing to break.

SILENT NO-OP GUARD
------------------
The `MTP EFFECTIVE TOP-K:` line introspects the layer that was actually BUILT
rather than trusting the code above it. A silent no-op is the dominant failure
mode in this project (LESSONS.md) -- read that line before trusting any number.

Distinct from `LogitsProcessor.get_top_k_tokens` in newer bases: that is vocab
top-k for logits, this is MoE expert-routing top-k. Unrelated code paths.

Idempotent; fails loudly if the anchor has drifted.
"""

import os
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/models/qwen3_5_mtp.py"
)

MARKER = "DGX_SPARK_MTP_TOP_K"

# Byte-identical in vllm-node-20260730 and vllm-node-20260824.
OLD = '''        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )
        vllm_config.quant_config = original_quant
'''

NEW = '''        # ---- BEGIN DGX_SPARK_MTP_TOP_K (Prunning/MTP_OPTIMIZATION_PLAN.md) ----
        # The draft head is a 256-expert MoE layer that reads the SAME global
        # config.num_experts_per_tok as the target model, so cutting the target
        # to top-k=4 cut the draft head with it. VLLM_MTP_TOP_K lets the head
        # route wider than the target. Unset -> identical to stock.
        import os as _dgx_os

        _mtp_top_k = int(_dgx_os.environ.get("VLLM_MTP_TOP_K", "0") or 0)
        _mtp_saved = getattr(self.config, "num_experts_per_tok", None)
        if _mtp_top_k > 0 and _mtp_saved is not None:
            self.config.num_experts_per_tok = _mtp_top_k
            logger.info(
                "MTP TOP-K OVERRIDE ACTIVE: draft head top_k=%d (target top_k=%d)",
                _mtp_top_k,
                _mtp_saved,
            )
        else:
            logger.info(
                "MTP TOP-K OVERRIDE INACTIVE: draft head top_k=%s (stock behaviour)",
                _mtp_saved,
            )
        try:
            self.layers = torch.nn.ModuleList(
                Qwen3_5DecoderLayer(
                    vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(self.num_mtp_layers)
            )
            vllm_config.quant_config = original_quant
        finally:
            # Restore unconditionally: everything built after this -- and the
            # config object the rest of vLLM reads -- must see the real value.
            # Placed OUTSIDE upstream's own quant_config restore above so the
            # two nest rather than race.
            if _mtp_top_k > 0 and _mtp_saved is not None:
                self.config.num_experts_per_tok = _mtp_saved
        # Prove the effective widths rather than trusting the code above. This
        # is the check that stops the whole experiment from being a silent
        # no-op -- the dominant failure mode in this project (LESSONS.md).
        try:
            _blk = self.layers[0].mlp
            _eff = getattr(_blk, "top_k", None)
            if _eff is None and hasattr(_blk, "experts"):
                _eff = getattr(_blk.experts, "top_k", None)
            logger.info(
                "MTP EFFECTIVE TOP-K: draft_head=%s target_config=%s",
                _eff,
                getattr(self.config, "num_experts_per_tok", None),
            )
        except Exception as _e:  # never let instrumentation break serving
            logger.warning("MTP EFFECTIVE TOP-K: could not introspect (%s)", _e)
        # ---- END DGX_SPARK_MTP_TOP_K ----
'''


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found")
        sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if MARKER in content:
        print("SKIP: MTP top-k override already applied")
        return

    count = content.count(OLD)
    if count != 1:
        print(f"FAIL: MTP draft-layer anchor matched {count} times, expected exactly 1")
        print("      Upstream qwen3_5_mtp.py has drifted; re-derive this patch.")
        sys.exit(1)

    content = content.replace(OLD, NEW)

    with open(TARGET, "w") as f:
        f.write(content)

    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: patched qwen3_5_mtp.py does not compile: {exc}")
        sys.exit(1)

    print("OK: MTP top-k override applied (VLLM_MTP_TOP_K)")


if __name__ == "__main__":
    main()
