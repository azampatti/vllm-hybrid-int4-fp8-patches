#!/usr/bin/env python3
"""Predict INC quant resolution WITHOUT loading 65 GB of weights.

Answers, in ~5 seconds, the question a failed serve takes 20 minutes to answer:
does the engine agree with the checkpoint about which layers are quantized?

The draft/MTP rows are the ones that matter. vLLM strips the "mtp." prefix when
it builds the draft model, so if those resolve quantized=False while the
checkpoint stores packed INT4, the load WILL die with:

    ValueError: There is no module or parameter named
    'layers.0.mlp.shared_expert.down_proj.qweight' in Qwen3_5Model

Expected on a correctly patched image (see patch 04):

    DRAFT  layers.0.mlp.shared_expert.down_proj   bits= 4 quantized=True
    DRAFT  layers.0.mlp.shared_expert_gate        bits=16 quantized=False
    MAIN   ...layers.0.mlp.experts.w2_weight      bits= 4 quantized=True
    MAIN   ...layers.0.mlp.shared_expert.down_proj bits=16 quantized=False

That last MAIN row is the trained BF16 shared expert and MUST stay bits=16 --
if it flips to 4 the healing work is being thrown away at load time.

Usage (inside the image):
  docker run --rm -v ~/models:/models:ro -v $PWD/probe_quant_resolution.py:/tmp/p.py:ro \
    --entrypoint bash vllm-qwen35-v3:latest -c 'python3 /tmp/p.py /models/<dir>'
"""

import json
import sys

import torch.nn as nn

from vllm.model_executor.layers.quantization.inc import INCConfig

model_dir = sys.argv[1] if len(sys.argv) > 1 else (
    "/models/qwen35-122b-sar-opencode-topk4-healed-epoch3"
)

cfg = json.load(open(f"{model_dir}/config.json"))
quant_config = INCConfig.from_config(cfg["quantization_config"])
dummy = nn.Linear(4, 4)

DRAFT = [
    "layers.0.mlp.shared_expert.down_proj",
    "layers.0.mlp.shared_expert.gate_proj",
    "layers.0.mlp.shared_expert.up_proj",
    "layers.0.mlp.shared_expert_gate",
    "layers.0.mlp.gate",
]
MAIN = [
    "model.language_model.layers.0.mlp.experts.w2_weight",
    "model.language_model.layers.0.mlp.shared_expert.down_proj",
    "model.language_model.layers.0.mlp.shared_expert_gate",
]

print(f"model: {model_dir}")
print(f"blocks: {quant_config.block_name_to_quantize}")
print()
for tag, names in (("DRAFT", DRAFT), ("MAIN", MAIN)):
    for name in names:
        lc = quant_config.config_parser.resolve(dummy, name)
        print(f"{tag:6s} {name:58s} bits={lc.bits:2d} quantized={lc.quantized}")
