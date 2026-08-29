#!/usr/bin/env python3
"""Port of albond patch 01 (Hybrid INT4+FP8 dispatch) to vLLM >= 0.26.

Upstream refactored `quantization/inc.py` (a single 537-line module in 0.19)
into a `quantization/inc/` package: `inc.py` + `config_parser.py` + `schemes/`.
The per-backend `apply_gptq_quant_layer` / `apply_awq_quant_layer` entry points
that the 0.19 patch hooked are gone; all dispatch is now funnelled through a
single `INCConfig.get_quant_method`.

That makes the port SMALLER than the original: instead of patching two
`apply_*_quant_layer` methods plus `get_quant_method`, we patch the two places
inside `get_quant_method` where a layer falls through to
`UnquantizedLinearMethod`, and hand those layers to `Fp8LinearMethod` when the
checkpoint carries FP8 weights for them.

Behaviour is otherwise identical to the 0.19 patch, with one deliberate
tightening noted at the DIVERGENCE comment below.

Idempotent: re-running is a no-op. Fails loudly (exit 1) if any anchor has
drifted -- never a silent partial apply.
"""

import os
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/quantization/inc/inc.py"
)

MARKER = "DGX_SPARK_HYBRID_INT4_FP8"


# --------------------------------------------------------------------------
# Anchors: (description, exact existing text, replacement text)
# --------------------------------------------------------------------------

# 1. Imports. Kept lazy inside methods where they risk an import cycle
#    (vllm.model_executor.layers.quantization.fp8 imports back into the
#    quantization package); only the safetensors dtype table is top-level.
IMPORTS_OLD = """from .config_parser import INCConfigParser
"""
IMPORTS_NEW = """from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

from .config_parser import INCConfigParser
"""

# 2. Extra state on the config object.
INIT_OLD = """        self.config_parser = INCConfigParser(self)
"""
INIT_NEW = """        self.config_parser = INCConfigParser(self)

        # DGX_SPARK_HYBRID_INT4_FP8: populated by maybe_update_config()
        self.fp8_config = None
        self.fp8_layers: set[str] = set()
"""

# 3. FP8 layer names are HF-namespaced; remap them alongside the others.
MAPPER_OLD = """        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)
"""
MAPPER_NEW = """        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)
        # DGX_SPARK_HYBRID_INT4_FP8
        if self.fp8_layers:
            self.fp8_layers = set(hf_to_vllm_mapper.apply_list(list(self.fp8_layers)))
"""

# 4. The two new methods, inserted ahead of get_quant_method.
#
#    NOTE on the signature of maybe_update_config: 0.19 called it
#    `(self, model_name, revision=None)`. 0.26's base_config.py declares
#    `(self, model_name, hf_config=None, revision=None)` and config/vllm.py
#    calls it with `hf_config=` as a keyword. Getting this wrong is the
#    "INCConfig hf_config error" in the runbook's troubleshooting table.
METHODS_OLD = """    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from .schemes.factory import resolve_scheme
"""
METHODS_NEW = '''    def maybe_update_config(
        self,
        model_name: str,
        hf_config=None,
        revision: str | None = None,
    ):
        """DGX_SPARK_HYBRID_INT4_FP8: detect FP8 layers in a hybrid checkpoint.

        The hybrid checkpoint stores the MoE experts as INT4 (auto-round) but
        leaves the dense shared_expert / linear_attn projections in blockwise
        FP8. Stock INC sees "not quantized" for those and drops them to
        UnquantizedLinearMethod, which is both wrong (weight_scale_inv is never
        applied) and slow. Here we scan the shard metadata for FP8 weights that
        have a matching weight_scale_inv and record them.
        """
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        from vllm.transformers_utils.config import get_safetensors_params_metadata

        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        fp8_weights: dict[str, dict] = {}
        for param_name, info in metadata.items():
            dtype_str = info.get("dtype", None)
            if dtype_str is None:
                continue
            torch_dtype = _SAFETENSORS_TO_TORCH_DTYPE.get(dtype_str)
            if torch_dtype == torch.float8_e4m3fn and param_name.endswith(".weight"):
                scale_name = param_name.replace(".weight", ".weight_scale_inv")
                if scale_name in metadata:
                    fp8_weights[param_name] = info

        if not fp8_weights:
            return

        # Infer block size from the first FP8 weight + scale pair.
        block_size = None
        for param_name, info in fp8_weights.items():
            scale_name = param_name.replace(".weight", ".weight_scale_inv")
            scale_info = metadata[scale_name]
            w_shape = info.get("shape", [])
            s_shape = scale_info.get("shape", [])
            if len(w_shape) == 2 and len(s_shape) == 2:
                block_size = [w_shape[0] // s_shape[0], w_shape[1] // s_shape[1]]
                break

        if block_size is None:
            return

        self.fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=block_size,
        )
        self.fp8_layers = {name.rsplit(".weight", 1)[0] for name in fp8_weights}
        logger.info(
            "Hybrid INT4+FP8: detected %d FP8 dense layers (block_size=%s)",
            len(self.fp8_layers),
            block_size,
        )

    def _is_layer_fp8(self, prefix: str) -> bool:
        """DGX_SPARK_HYBRID_INT4_FP8: does this layer live in FP8?"""
        if not self.fp8_layers:
            return False
        if prefix in self.fp8_layers:
            return True
        # Fused module matching: a vLLM qkv_proj / gate_up_proj covers several
        # checkpoint tensors, and is only FP8 if every shard is.
        fused_mapping = getattr(self, "packed_modules_mapping", {})
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard) for shard in fused_mapping[proj_name]
            ]
            return all(
                any(fp8_layer in sp for fp8_layer in self.fp8_layers)
                for sp in shard_prefixes
            )
        return any(fp8_layer in prefix for fp8_layer in self.fp8_layers)

    def _maybe_fp8_method(self, layer, prefix: str):
        """DGX_SPARK_HYBRID_INT4_FP8: Fp8LinearMethod, or None to fall through."""
        from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

        if not self.fp8_config:
            return None
        # DIVERGENCE from the 0.19 patch: gate on LinearBase.
        #
        # 0.19 returned Fp8LinearMethod for ANY layer type that matched, which
        # was harmless there because only linear layers reached that branch.
        # 0.26 routes RoutedExperts through the same code path, and handing a
        # RoutedExperts an Fp8LinearMethod would fail at weight-creation time.
        # ParallelLMHead is a VocabParallelEmbedding, not a LinearBase, so it
        # is excluded too -- the lm_head is covered by patch 03 instead.
        if not isinstance(layer, LinearBase):
            return None
        if not self._is_layer_fp8(prefix):
            return None
        return Fp8LinearMethod(self.fp8_config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from .schemes.factory import resolve_scheme
'''

# 5a. Dispatch point one: layers explicitly marked bits>=16 in extra_config.
DISPATCH1_OLD = """                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    if isinstance(layer, RoutedExperts):
"""
DISPATCH1_NEW = """                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    # DGX_SPARK_HYBRID_INT4_FP8: FP8 overrides "unquantized"
                    fp8_method = self._maybe_fp8_method(layer, prefix)
                    if fp8_method is not None:
                        return fp8_method
                    if isinstance(layer, RoutedExperts):
"""

# 5b. Dispatch point two: layers the config parser resolved as not quantized.
#     This is the one that catches shared_expert / linear_attn.
DISPATCH2_OLD = """        layer_config = self.config_parser.resolve(layer, prefix)
        if not layer_config.quantized:
            if isinstance(layer, (LinearBase, ParallelLMHead)):
"""
DISPATCH2_NEW = """        layer_config = self.config_parser.resolve(layer, prefix)
        if not layer_config.quantized:
            # DGX_SPARK_HYBRID_INT4_FP8: dispatch FP8 for hybrid dense layers
            fp8_method = self._maybe_fp8_method(layer, prefix)
            if "shared_expert" in prefix or "linear_attn" in prefix:
                logger.info(
                    "INC GPTQ dispatch: prefix=%s, fp8_match=%s, "
                    "fp8_config=%s, layer_type=%s",
                    prefix,
                    fp8_method is not None,
                    self.fp8_config is not None,
                    type(layer).__name__,
                )
            if fp8_method is not None:
                return fp8_method
            if isinstance(layer, (LinearBase, ParallelLMHead)):
"""

EDITS = [
    ("imports", IMPORTS_OLD, IMPORTS_NEW),
    ("__init__ fp8 state", INIT_OLD, INIT_NEW),
    ("apply_vllm_mapper fp8_layers remap", MAPPER_OLD, MAPPER_NEW),
    ("maybe_update_config / _is_layer_fp8 / _maybe_fp8_method", METHODS_OLD, METHODS_NEW),
    ("dispatch: extra_config bits>=16", DISPATCH1_OLD, DISPATCH1_NEW),
    ("dispatch: layer_config not quantized", DISPATCH2_OLD, DISPATCH2_NEW),
]


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found")
        print("      (vLLM < 0.26 keeps INC in a single inc.py -- use patch 01 for 0.19)")
        sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if MARKER in content:
        print("SKIP: hybrid INT4+FP8 dispatch already applied")
        return

    for desc, old, new in EDITS:
        count = content.count(old)
        if count != 1:
            print(f"FAIL: anchor {desc!r} matched {count} times, expected exactly 1")
            print("      Upstream inc.py has drifted; re-derive this patch.")
            sys.exit(1)
        content = content.replace(old, new)

    with open(TARGET, "w") as f:
        f.write(content)

    # Verify it still parses before we declare victory.
    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: patched inc.py does not compile: {exc}")
        sys.exit(1)

    print(f"OK: hybrid INT4+FP8 dispatch applied ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
