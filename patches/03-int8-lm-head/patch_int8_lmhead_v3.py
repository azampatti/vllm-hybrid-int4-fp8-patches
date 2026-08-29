#!/usr/bin/env python3
"""Port of albond patch 03 (INT8 LM Head v2) to vLLM >= 0.26.

The 0.19 patch replaced the body of `LogitsProcessor._get_logits`, whose first
statement was the `lm_head.quant_method.apply(...)` projection. 0.26 factored
that projection out into a new `LogitsProcessor._apply_head`, which also
implements a `head_dtype` upcast path -- so the old anchor no longer exists and
the original script exits "FAIL: pattern not found".

This version injects the same INT8 GEMV fast path at the top of `_apply_head`.
The Triton kernel, the INT8 quantisation of the lm_head weight, and the
autotune config list are carried over byte-identically from patch 03; only the
host site and the head_dtype handling are new.

Note the argument order differs from `_get_logits`: `_apply_head` takes
(lm_head, hidden_states, embedding_bias).

Idempotent; fails loudly if the anchor has drifted.
"""

import os
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/logits_processor.py"
)

MARKER = "DGX_SPARK_INT8_LMHEAD_V2"

OLD = '''        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
'''

NEW = '''        """Project hidden states through the lm_head, honoring head_dtype."""
        # DGX_SPARK_INT8_LMHEAD_V2: Batched 2D INT8 GEMV — single kernel launch
        if not hasattr(self, '_int8v2_initialized'):
            self._int8v2_initialized = True
            w = lm_head.weight.data
            if w.dtype in (torch.bfloat16, torch.float16) and w.shape[0] > 100000:
                scales = w.float().abs().amax(dim=1) / 127.0
                scales = scales.clamp(min=1e-12)
                w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
                lm_head._ww_int8 = w_int8
                lm_head._ww_scales = scales.to(torch.float16)
                orig_size = w.numel() * w.element_size()
                lm_head.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
                import sys as _sys
                print(f"DGX_SPARK_V2: LM Head -> INT8 Batched Triton ({list(w_int8.shape)}, saved {orig_size//1024//1024}MB)", file=_sys.stderr, flush=True)
                import triton
                import triton.language as tl
                # Backport from community PR (DGX Spark forum): autotune
                # picks BLOCK_M/BLOCK_K/num_warps/num_stages for SM121 instead
                # of the hardcoded 128/256/4/2. Arithmetic is byte-identical;
                # only the kernel-launch parameters change. ~6s once-per-process
                # autotune cost (per unique NUM_BATCH × 8 configs), then steady
                # state. Measured on Qwen3.5-122B/Spark: +0.4 to +2.6% across
                # prompt classes (avg +1.2%), no quality change.
                _AUTOTUNE_CONFIGS = [
                    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),  # = v2 baseline
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                ]
                @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=['M', 'K', 'NUM_BATCH'])
                @triton.jit
                def _k_v2(out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                          stride_ob, stride_xb, NUM_BATCH: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    # 1D grid: each block processes ALL batch elements
                    # Weight tile loaded ONCE, reused for all batch inputs
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    # One accumulator per batch element (unrolled by compiler)
                    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        # Load weight tile ONCE
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :], other=0).to(tl.float32)
                        # Reuse weight tile for each batch element
                        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                        acc0 += tl.sum(w * x0[None, :], axis=1)
                        if NUM_BATCH > 1:
                            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc1 += tl.sum(w * x1[None, :], axis=1)
                        if NUM_BATCH > 2:
                            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc2 += tl.sum(w * x2[None, :], axis=1)
                        if NUM_BATCH > 3:
                            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc3 += tl.sum(w * x3[None, :], axis=1)
                    # Scale and store
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 1:
                        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 2:
                        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 3:
                        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)
                lm_head._ww_kernel_v2 = _k_v2

        if hasattr(lm_head, '_ww_int8'):
            M, K = lm_head._ww_int8.shape
            x = hidden_states.view(-1, K)
            batch = x.shape[0]
            out = torch.empty(batch, M, dtype=torch.float16, device=x.device)
            # Autotune-aware grid: BLOCK_M is chosen by autotuner per (M,K,NUM_BATCH).
            grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
            if batch <= 4:
                # Small batch (decode): shared-weight kernel reads weights ONCE
                nb = batch
                lm_head._ww_kernel_v2[grid](
                    out, lm_head._ww_int8, x.to(torch.float16),
                    lm_head._ww_scales, M, K,
                    out.stride(0), x.stride(0), NUM_BATCH=nb)
            else:
                # Large batch (prefill/profile): fall back to per-row loop
                for b in range(batch):
                    lm_head._ww_kernel_v2[grid](
                        out[b:b+1], lm_head._ww_int8, x[b:b+1].to(torch.float16),
                        lm_head._ww_scales, M, K,
                        M, K, NUM_BATCH=1)
            logits = out.view(hidden_states.shape[:-1] + (M,))
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(logits.dtype)
            # New in 0.26: callers may ask for a wider head dtype. The INT8
            # kernel always emits fp16, so honour the request here rather than
            # letting a dtype mismatch surface downstream in _gather_logits.
            if self.head_dtype is not None and self.head_dtype != logits.dtype:
                logits = logits.to(self.head_dtype)
            return logits

        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
'''


def main() -> None:
    if not os.path.exists(TARGET):
        print(f"FAIL: {TARGET} not found")
        sys.exit(1)

    with open(TARGET) as f:
        content = f.read()

    if MARKER in content:
        print("SKIP: INT8 LM Head v2 already applied")
        return

    if "def _apply_head(" not in content:
        print("FAIL: no _apply_head found -- this looks like vLLM < 0.26.")
        print("      Use the original patches/03-int8-lm-head/patch_int8_lmhead.py.")
        sys.exit(1)

    count = content.count(OLD)
    if count != 1:
        print(f"FAIL: _apply_head anchor matched {count} times, expected exactly 1")
        print("      Upstream logits_processor.py has drifted; re-derive this patch.")
        sys.exit(1)

    content = content.replace(OLD, NEW)

    with open(TARGET, "w") as f:
        f.write(content)

    import py_compile

    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"FAIL: patched logits_processor.py does not compile: {exc}")
        sys.exit(1)

    print("OK: INT8 LM Head v2 patch applied (_apply_head host site)")


if __name__ == "__main__":
    main()
