#!/usr/bin/env python3
"""Carry quantization_config into the Qwen3.5 MTP draft config (vLLM >= 0.26).

THE BUG
-------
This checkpoint is a VL-wrapper config: ``model_type: qwen3_5_moe`` at the top
level, with ``quantization_config`` at the TOP level and a nested
``text_config`` (``qwen3_5_moe_text``) that does NOT carry it::

    top-level keys: [architectures, image_token_id, model_type,
                     quantization_config, text_config, vision_config, ...]
    quantization_config in text_config: False

vLLM builds the MTP draft model from the PROMOTED text_config. That promoted
config has no ``quantization_config``, so the draft model is constructed with
no quant method at all -- ``INCConfig.get_quant_method`` is never called for a
single draft layer (verified by instrumenting it: only main-model prefixes ever
appear). Every draft linear is therefore an unquantized RowParallelLinear
holding ``.weight``, and weight loading dies on the checkpoint's packed
``.qweight``::

    ValueError: There is no module or parameter named
    'layers.0.mlp.shared_expert.down_proj.qweight' in Qwen3_5Model.
    The available parameters ... are: {'...down_proj.weight'}

THE FIX
-------
Copy the outer ``quantization_config`` onto ``text_config`` when the latter
lacks it, in the ``qwen3_5``/``qwen3_5_moe`` MTP branch of
``SpeculativeConfig._maybe_override_draft_max_model_len``'s config rewrite.

This is NOT a novel workaround -- upstream already does exactly this, with the
same idiom, for ``step3p5``/``step3p7`` (speculative.py ~line 536) and for
``minimax_m3_vl`` (~line 594). The ``qwen3_5`` branch simply never got it,
presumably because the non-VL Qwen3.5 configs keep quantization_config and the
model config at the same level. Ours is the VL-wrapper shape, so it does not.

Distinct from patch 04: 04 fixes how a draft layer's bits are RESOLVED once INC
is consulted; 05 is why INC was never consulted at all. Both are required --
04 alone leaves the draft with no quant config, 05 alone leaves the draft
resolving its shared_expert as unquantized via the stripped-prefix path.

Idempotent; fails loudly if the anchor has drifted.
"""

import os
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"

MARKER = "DGX_SPARK_MTP_QUANT_CONFIG"

# Upstream drifted between bases. 0.26.1rc1.dev1123 widened the qwen3_5 branch
# to four model types and turned `is_moe` into a tuple membership test, which
# broke the original single-line anchor. Both shapes are supported: the first
# anchor that matches exactly once is used. The insertion point is identical in
# both -- immediately before `model_type` is rewritten to "qwen3_5_mtp".

# --- anchor A: vLLM >= 0.26.1rc1.dev1123 (four model types, tuple is_moe) ---
OLD_A = """            is_moe = hf_config.model_type in ("qwen3_5_moe", "qwen3_5_moe_text")
            hf_config.model_type = "qwen3_5_mtp"
"""

NEW_A = '''            is_moe = hf_config.model_type in ("qwen3_5_moe", "qwen3_5_moe_text")
            # DGX_SPARK_MTP_QUANT_CONFIG: on VL-wrapper checkpoints the
            # quantization_config lives on the OUTER config, while the draft
            # model is built from the promoted text_config. Without this carry
            # the draft head is built with no quant method and load fails on
            # the checkpoint's packed .qweight. Upstream already does this for
            # step3p5 and minimax_m3_vl; the qwen3_5 branch was missed.
            # Text-only checkpoints (qwen3_5_text / qwen3_5_moe_text) have no
            # text_config, so this is a no-op for them -- correct, since their
            # quantization_config is already on the config being used.
            _dgx_qc = getattr(hf_config, "quantization_config", None)
            _dgx_tc = getattr(hf_config, "text_config", None)
            if _dgx_qc is not None and _dgx_tc is not None:
                if getattr(_dgx_tc, "quantization_config", None) is None:
                    if hasattr(_dgx_tc, "update"):
                        _dgx_tc.update({"quantization_config": _dgx_qc})
                    else:
                        setattr(_dgx_tc, "quantization_config", _dgx_qc)
                    logger.info(
                        "DGX_SPARK_MTP_QUANT_CONFIG: propagated "
                        "quantization_config to qwen3_5 MTP text_config"
                    )
            hf_config.model_type = "qwen3_5_mtp"
'''

# --- anchor B: earlier 0.26 bases (two model types, equality is_moe) ---
OLD_B = """        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
"""

NEW_B = '''        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            # DGX_SPARK_MTP_QUANT_CONFIG: on VL-wrapper checkpoints the
            # quantization_config lives on the OUTER config, while the draft
            # model is built from the promoted text_config. Without this carry
            # the draft head is built with no quant method and load fails on
            # the checkpoint's packed .qweight. Upstream already does this for
            # step3p5 and minimax_m3_vl; the qwen3_5 branch was missed.
            # Text-only checkpoints (qwen3_5_text / qwen3_5_moe_text) have no
            # text_config, so this is a no-op for them -- correct, since their
            # quantization_config is already on the config being used.
            _dgx_qc = getattr(hf_config, "quantization_config", None)
            _dgx_tc = getattr(hf_config, "text_config", None)
            if _dgx_qc is not None and _dgx_tc is not None:
                if getattr(_dgx_tc, "quantization_config", None) is None:
                    if hasattr(_dgx_tc, "update"):
                        _dgx_tc.update({"quantization_config": _dgx_qc})
                    else:
                        setattr(_dgx_tc, "quantization_config", _dgx_qc)
                    logger.info(
                        "DGX_SPARK_MTP_QUANT_CONFIG: propagated "
                        "quantization_config to qwen3_5 MTP text_config"
                    )
            hf_config.model_type = "qwen3_5_mtp"
'''

ANCHORS = [
    ("A (>= dev1123, four model types)", OLD_A, NEW_A),
    ("B (earlier 0.26)", OLD_B, NEW_B),
]


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found")
        sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if MARKER in content:
        print("SKIP: MTP quant-config propagation already applied")
        return

    matched = None
    seen = []
    for label, old, new in ANCHORS:
        count = content.count(old)
        seen.append(f"{label}={count}")
        if count == 1:
            matched = (label, old, new)
            break
        if count > 1:
            print(f"FAIL: qwen3_5 MTP anchor {label} matched {count} times, expected exactly 1")
            sys.exit(1)

    if matched is None:
        print("FAIL: no qwen3_5 MTP anchor matched (" + ", ".join(seen) + ")")
        print("      Upstream speculative.py has drifted; re-derive this patch.")
        sys.exit(1)

    label, old, new = matched
    content = content.replace(old, new)

    with open(TARGET, "w") as f:
        f.write(content)

    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: patched speculative.py does not compile: {exc}")
        sys.exit(1)

    print(f"OK: MTP quant-config propagation applied (anchor {label})")


if __name__ == "__main__":
    main()
