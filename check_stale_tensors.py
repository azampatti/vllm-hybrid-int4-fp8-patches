#!/usr/bin/env python3
"""Detect tensors present in the shard FILES but absent from the INDEX.

RUN THIS BEFORE DEBUGGING ANY vLLM >= 0.26 LOAD FAILURE. It takes ~5 seconds
and settles, in advance, the single most expensive failure mode on this project.

Removing a key from model.safetensors.index.json does NOT remove the tensor from
the shard file. Two loaders disagree about the result:

  vLLM 0.19 (vllm-qwen35-v2)  -- index-driven. Never sees the extra tensors.
  vLLM 0.26 (eugr / v3)       -- fastsafetensors enumerates the FILES, finds
                                 e.g. ...shared_expert.down_proj.qweight, and
                                 tries to load it into a module the config
                                 correctly declares BF16:

      ValueError: There is no module or parameter named
      'layers.0.mlp.shared_expert.down_proj.qweight' in Qwen3_5Model.
      The available parameters ... are: {'...down_proj.weight'}

NOTE the error names Qwen3_5Model -- that is the MAIN model, not the MTP head,
even though the message looks identical to a draft-head failure. No engine patch
or config change fixes this; only rewriting the weight files does.

Fix: topk4_heal/scripts/materialize_clean_shards.py (keeps exactly what the
index references). Expect ~432 dropped for a BF16-shared-expert variant.

Exit code 0 = clean, 1 = stale tensors found.

Usage:
  python3 check_stale_tensors.py ~/models/<model-dir>
"""

import glob
import json
import os
import struct
import sys


def tensor_keys(path: str) -> list[str]:
    """Read a safetensors header without loading any tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    # __metadata__ is the header's own metadata block, not a tensor.
    return [k for k in header if k != "__metadata__"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model_dir = os.path.expanduser(sys.argv[1])

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(f"FAIL: no index at {index_path}")
        return 2
    indexed = set(json.load(open(index_path))["weight_map"])

    in_files: set[str] = set()
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        print(f"FAIL: no .safetensors files in {model_dir}")
        return 2
    for shard in shards:
        in_files.update(tensor_keys(shard))

    stale = in_files - indexed
    missing = indexed - in_files

    print(f"model  : {model_dir}")
    print(f"shards : {len(shards)}")
    print(f"indexed: {len(indexed)}")
    print(f"in files: {len(in_files)}")
    print(f"STALE (in files, NOT indexed): {len(stale)}")
    print(f"MISSING (indexed, not in files): {len(missing)}")

    if missing:
        print("\n!! MISSING tensors -- this checkpoint is incomplete, not just stale:")
        for k in sorted(missing)[:10]:
            print("   ", k)

    if stale:
        print("\n!! WILL FAIL TO LOAD ON vLLM >= 0.26.")
        print("   Sample stale tensors:")
        for k in sorted(stale)[:8]:
            print("   ", k)
        print("\n   Fix: materialize shards containing exactly what the index")
        print("   references (topk4_heal/scripts/materialize_clean_shards.py),")
        print("   then re-run this check until STALE is 0.")
        return 1

    print("\nOK: clean. Every tensor in the files is referenced by the index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
