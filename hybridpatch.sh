#!/usr/bin/env bash
#
# hybridpatch.sh — one-shot builder for the patched "hybrid" vLLM image.
#
# Takes an existing vLLM base image (e.g. an eugr/spark-vllm-docker build) and
# produces a NEW image carrying the DGX Spark Qwen3.5-122B patch set:
#
#   01  hybrid INT4+FP8 dispatch          (quantization/inc/inc.py)
#   03  INT8 LM head v2                   (layers/logits_processor.py)
#   04  MTP/draft-head quant scoping      (quantization/inc/config_parser.py)
#   05  MTP quantization_config carry     (config/speculative.py)
#   06  VLLM_MTP_TOP_K draft-head width   (models/qwen3_5_mtp.py)
#
# Usage:
#   ./hybridpatch.sh vllm-node-20260730
#   ./hybridpatch.sh vllm-node-20260730 -t my-image:v1
#   ./hybridpatch.sh vllm-node-20260730 --force        # overwrite output tag
#
# SAFETY
#   * The base image is read ONLY. It is never rewritten or re-tagged.
#   * The output tag defaults to "<base-repo>-hybrid:latest" and the script
#     REFUSES to overwrite an existing tag unless --force is given. This is
#     deliberate: an image that silently replaces the one your benchmarks came
#     from destroys the ability to reproduce them.
#
# Full background, launch flags and verification gates: REPRODUCE.md
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIST="/usr/local/lib/python3.12/dist-packages/vllm"

BASE=""
OUT_TAG=""
FORCE=0

# --------------------------------------------------------------------------
# Terminal colours, only when stdout is a TTY.
# --------------------------------------------------------------------------
if [ -t 1 ]; then
    B=$'\033[1m'; R=$'\033[0;31m'; G=$'\033[0;32m'; Y=$'\033[0;33m'; N=$'\033[0m'
else
    B=""; R=""; G=""; Y=""; N=""
fi
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%sFATAL:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
    sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)  usage 0 ;;
        -f|--force) FORCE=1; shift ;;
        -t|--tag)   [ $# -ge 2 ] || die "--tag needs a value"; OUT_TAG="$2"; shift 2 ;;
        -*)         die "unknown option: $1  (--help for usage)" ;;
        *)
            [ -z "$BASE" ] || die "unexpected extra argument: $1"
            BASE="$1"; shift ;;
    esac
done

[ -n "$BASE" ] || usage 1

# Default output tag: <base-repo>-hybrid:latest, ignoring any tag on the base.
if [ -z "$OUT_TAG" ]; then
    OUT_TAG="${BASE%%:*}-hybrid:latest"
fi
case "$OUT_TAG" in *:*) ;; *) OUT_TAG="${OUT_TAG}:latest" ;; esac

say ""
say "${B}hybridpatch${N} — build a patched vLLM image"
say "  base image:  $BASE"
say "  output tag:  $OUT_TAG"
say ""

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
say "${B}[1/5]${N} Preflight"

command -v docker >/dev/null 2>&1 || die "docker not found in PATH"

docker image inspect "$BASE" >/dev/null 2>&1 \
    || die "base image '$BASE' not found locally. Build it first (see REPRODUCE.md §2)."
BASE_ID="$(docker image inspect "$BASE" --format '{{.Id}}')"
ok "base image exists (${BASE_ID:7:12})"

if [ "${OUT_TAG%%:*}" = "${BASE%%:*}" ]; then
    die "output repo equals base repo ('${BASE%%:*}'). Refusing to risk overwriting the base. Use -t."
fi

if docker image inspect "$OUT_TAG" >/dev/null 2>&1; then
    if [ "$FORCE" -eq 1 ]; then
        warn "output tag '$OUT_TAG' exists and WILL BE REPLACED (--force)"
    else
        die "output tag '$OUT_TAG' already exists.
       Refusing to overwrite it — if benchmarks came from that image, replacing
       it silently would make them unreproducible.
       Pick another tag with -t, or pass --force if you really mean it."
    fi
else
    ok "output tag is free"
fi

# All four patch scripts plus the Dockerfile must be present.
DOCKERFILE="$SCRIPT_DIR/docker/Dockerfile.v3"
[ -f "$DOCKERFILE" ] || die "missing $DOCKERFILE"
missing=0
for p in \
    patches/01-hybrid-int4-fp8/patch_inc_hybrid.py \
    patches/03-int8-lm-head/patch_int8_lmhead_v3.py \
    patches/04-mtp-block-scope/patch_mtp_block_scope.py \
    patches/05-mtp-quant-config/patch_mtp_quant_config.py \
    patches/06-mtp-top-k/patch_mtp_top_k.py
do
    if [ -f "$SCRIPT_DIR/$p" ]; then ok "patch present: $p"; else
        printf '  %s✗%s MISSING: %s\n' "$R" "$N" "$p"; missing=1
    fi
done
[ "$missing" -eq 0 ] || die "patch tree incomplete — run this script from its own directory."

# --------------------------------------------------------------------------
# Base sanity: report what we are layering on, and catch a 0.19 base early.
# --------------------------------------------------------------------------
say ""
say "${B}[2/5]${N} Inspecting base"

BASE_VLLM="$(docker run --rm --entrypoint bash "$BASE" -c \
    'python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null' 2>/dev/null | tr -d '\r' || true)"
[ -n "$BASE_VLLM" ] || die "could not determine vLLM version inside '$BASE' — is this a vLLM image?"
ok "vLLM $BASE_VLLM"

if docker run --rm --entrypoint bash "$BASE" -c \
      "test -f $DIST/model_executor/layers/quantization/inc/inc.py" 2>/dev/null; then
    ok "INC is a package (inc/) — modern layout, patches target this"
else
    die "this base has the old single-file inc.py (vLLM ~0.19).
       These patches target the refactored inc/ package in vLLM >= 0.26.
       For a 0.19 base use albond's original patches instead."
fi

if docker run --rm --entrypoint cat "$BASE" /workspace/build-metadata.yaml >/dev/null 2>&1; then
    say ""
    say "  build-metadata.yaml (record this — compare it if a patch anchor ever fails):"
    docker run --rm --entrypoint cat "$BASE" /workspace/build-metadata.yaml 2>/dev/null | sed 's/^/      /'
else
    warn "no /workspace/build-metadata.yaml — not an eugr build? proceeding anyway"
fi

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
say ""
say "${B}[3/5]${N} Building patched image (~15s)"
say ""

BUILD_LOG="$(mktemp -t hybridpatch-build.XXXXXX.log)"
trap 'rm -f "$BUILD_LOG"' EXIT

if ! docker build \
        --build-arg "VLLM_BASE=$BASE" \
        -t "$OUT_TAG" \
        -f "$DOCKERFILE" \
        "$SCRIPT_DIR" 2>&1 | tee "$BUILD_LOG"; then
    die "docker build failed — see output above."
fi

# The Dockerfile greps for every marker, so a drifted anchor already fails the
# build. Surface which patch died anyway, since that is the useful bit.
if grep -q "^FAIL:" "$BUILD_LOG"; then
    say ""
    grep "^FAIL:" "$BUILD_LOG" | sed 's/^/      /'
    die "a patch failed to apply (see FAIL line above). Upstream vLLM has
       drifted from the anchors — re-derive that patch against $BASE_VLLM."
fi

say ""
applied="$(grep -c '^OK: ' "$BUILD_LOG" || true)"
skipped="$(grep -c '^SKIP: ' "$BUILD_LOG" || true)"
if [ "$applied" -eq 0 ] && [ "$skipped" -eq 0 ]; then
    # Docker reused cached layers, so the patch scripts did not re-run and their
    # output is absent from this log. That is not a failure: the layers were
    # built by these same scripts. The marker checks below are the real proof.
    ok "all layers served from Docker cache — patches already baked in"
else
    ok "patches applied: $applied  (skipped/already-present: $skipped)"
fi

# --------------------------------------------------------------------------
# Post-build verification — independent of the build's own asserts
# --------------------------------------------------------------------------
say ""
say "${B}[4/5]${N} Verifying the built image"

verify_marker() {
    local marker="$1" file="$2" label="$3"
    if docker run --rm --entrypoint bash "$OUT_TAG" -c "grep -q '$marker' '$file'" 2>/dev/null; then
        ok "$label"
    else
        die "$label — marker '$marker' NOT found in the built image."
    fi
}

verify_marker DGX_SPARK_HYBRID_INT4_FP8 \
    "$DIST/model_executor/layers/quantization/inc/inc.py"            "01 hybrid INT4+FP8 dispatch"
verify_marker DGX_SPARK_INT8_LMHEAD_V2 \
    "$DIST/model_executor/layers/logits_processor.py"                "03 INT8 LM head v2"
verify_marker DGX_SPARK_MTP_BLOCK_SCOPE \
    "$DIST/model_executor/layers/quantization/inc/config_parser.py"  "04 MTP block scoping"
verify_marker DGX_SPARK_MTP_QUANT_CONFIG \
    "$DIST/config/speculative.py"                                    "05 MTP quant-config carry"
verify_marker DGX_SPARK_MTP_TOP_K \
    "$DIST/model_executor/models/qwen3_5_mtp.py"                     "06 VLLM_MTP_TOP_K draft-head width"

if docker run --rm --entrypoint bash "$OUT_TAG" -c \
      'python3 -c "import vllm.entrypoints.openai.api_server"' >/dev/null 2>&1; then
    ok "vLLM API server imports cleanly"
else
    die "the patched image does not import vLLM — do NOT use this image."
fi

# The base must be byte-identical to what we started with.
NOW_ID="$(docker image inspect "$BASE" --format '{{.Id}}')"
if [ "$NOW_ID" = "$BASE_ID" ]; then
    ok "base image unchanged (${BASE_ID:7:12})"
else
    die "base image ID CHANGED — something rewrote '$BASE'. Investigate before use."
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
OUT_ID="$(docker image inspect "$OUT_TAG" --format '{{.Id}}')"
OUT_SIZE="$(docker image inspect "$OUT_TAG" --format '{{.Size}}' \
            | awk '{printf "%.1f GB", $1/1000000000}')"

say ""
say "${B}[5/5]${N} Done"
say ""
say "  ${G}${B}$OUT_TAG${N}  (${OUT_ID:7:12}, $OUT_SIZE)"
say "  built from $BASE — vLLM $BASE_VLLM"
say ""
say "  ${B}Before you serve anything, check the weights:${N}"
say "      python3 $SCRIPT_DIR/check_stale_tensors.py ~/models/<model-dir>"
say ""
say "  Tensors present in the shard FILES but missing from the index will fail"
say "  to load on vLLM >= 0.26 with an error naming Qwen3_5Model. That is a"
say "  weight-file problem — no image or flag fixes it. See REPRODUCE.md §5."
say ""
say "  ${B}Optional — widen the MTP draft head (patch 06):${N}"
say "      VLLM_MTP_TOP_K=8   # unset => stock routing, byte-identical"
say "  Confirm it took effect or it is a silent no-op; the startup log must"
say "  print 'MTP TOP-K OVERRIDE ACTIVE' AND 'MTP EFFECTIVE TOP-K: draft_head=8'."
say ""
say "  Then launch and check the four gates in REPRODUCE.md §7:"
say "      model load ~64 GiB · MTP acceptance present · no PIECEWISE · health 200"
say ""
