#!/usr/bin/env python3
"""Fix MTP/draft-head quant resolution on vLLM >= 0.26 (`inc/config_parser.py`).

THE BUG
-------
vLLM builds the MTP head as its own Qwen3_5Model with the ``mtp.`` prefix
STRIPPED. It therefore asks INC about::

    layers.0.mlp.shared_expert.down_proj

while the checkpoint declares::

    block_name_to_quantize = ['model.language_model.layers', 'mtp.layers']

The strict ``layer_name.startswith(block)`` test in ``_resolve_raw`` matches
neither, so ``quantized`` becomes False, bits fall back to 16, vLLM builds an
unquantized RowParallelLinear expecting ``.weight`` -- and load dies on the
checkpoint's packed ``.qweight``::

    ValueError: There is no module or parameter named
    'layers.0.mlp.shared_expert.down_proj.qweight' in Qwen3_5Model

The checkpoint config is CORRECT: those MTP keys are legitimately INT4 because
the draft head was never a training target. vLLM 0.19 scoped the lookup
properly; 0.26 does not. This is orthogonal to patches 01/03.

Adding ``mtp.``-prefixed entries to ``extra_config`` does NOT help -- the lookup
only ever sees the stripped name, so a qualified key can never match. (Tried
2026-08-09 on the epoch3-node build; that observation is what pinned this down.)

THE FIX
-------
When the strict block match fails, re-qualify: find prefixes P such that
``P + "." + layer_name`` matches a declared block, and retry the whole
resolution with that qualified name. This restores both halves of the answer --
the ``quantized`` flag AND the per-layer ``extra_config`` bits -- rather than
just forcing ``quantized=True``, which would wrongly drag bits=16 layers
(``shared_expert_gate``, ``mlp.gate``, ``fc``) down to the default bits=4.

HEURISTIC, STATED PLAINLY: for this checkpoint both ``mtp`` and
``model.language_model`` are viable prefixes for a bare ``layers.0...`` name,
and nothing inside config_parser reveals which sub-model is being loaded. We
take the SHORTEST prefix, which selects ``mtp`` -- correct here, because the
main model never reaches this fallback (its names already match a block
strictly). If a future checkpoint has a differently-named draft head, verify
with `probe_quant_resolution.py` before trusting a load.

The fallback is entered ONLY when the strict match failed, so it can never
change a resolution that already succeeded.

Idempotent; fails loudly if the anchor has drifted.
"""

import os
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/quantization/inc/config_parser.py"
)

MARKER = "DGX_SPARK_MTP_BLOCK_SCOPE"

OLD = """        quantized = not isinstance(layer, ParallelLMHead)
        if self._config.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name)
                for name in self._config.block_name_to_quantize
            )
"""

NEW = '''        quantized = not isinstance(layer, ParallelLMHead)
        if self._config.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name)
                for name in self._config.block_name_to_quantize
            )
            # DGX_SPARK_MTP_BLOCK_SCOPE: see patch 04. vLLM loads the MTP head
            # with the "mtp." prefix stripped, so a bare "layers.0..." matches
            # no declared block and is misreported as unquantized. Re-qualify
            # and resolve again under the full name.
            if not quantized:
                qualified = self._dgx_requalify(layer_name)
                if qualified is not None:
                    return self._resolve_raw(layer, qualified)
'''

# The helper method, inserted just before _resolve_raw.
HELPER_OLD = """    def _resolve_raw(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int, bool]:
"""

HELPER_NEW = '''    def _dgx_requalify(self, layer_name: str) -> "str | None":
        """DGX_SPARK_MTP_BLOCK_SCOPE: rebuild a stripped sub-model prefix.

        Returns ``P + "." + layer_name`` for the shortest prefix P that makes
        the name match a declared block, or None when no prefix does. Matching
        is on dotted component boundaries only, so "layers" never matches
        "layersnorm".
        """
        blocks = self._config.block_name_to_quantize or []
        candidates = []
        for block in blocks:
            parts = block.split(".")
            # i is how many leading components were stripped off.
            for i in range(1, len(parts)):
                suffix = ".".join(parts[i:])
                if layer_name == suffix or layer_name.startswith(suffix + "."):
                    candidates.append((i, ".".join(parts[:i]) + "." + layer_name))
        if not candidates:
            return None
        # Shortest stripped prefix wins -- see the module docstring.
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][1]

    def _resolve_raw(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int, bool]:
'''

EDITS = [
    ("_dgx_requalify helper", HELPER_OLD, HELPER_NEW),
    ("block_name_to_quantize fallback", OLD, NEW),
]


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found")
        sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if MARKER in content:
        print("SKIP: MTP block-scope fix already applied")
        return

    for desc, old, new in EDITS:
        count = content.count(old)
        if count != 1:
            print(f"FAIL: anchor {desc!r} matched {count} times, expected exactly 1")
            sys.exit(1)
        content = content.replace(old, new)

    with open(TARGET, "w") as f:
        f.write(content)

    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: patched config_parser.py does not compile: {exc}")
        sys.exit(1)

    print("OK: MTP block-scope fix applied")


if __name__ == "__main__":
    main()
