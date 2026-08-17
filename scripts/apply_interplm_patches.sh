#!/bin/bash
# Apply glp-prot's local patches to the InterPLM submodule.
#
# Why this exists: InterPLM is pinned as a submodule at upstream
# ElanaPearl/InterPLM (see .gitmodules), so we can't push fixes there and we don't
# bump the submodule pointer to a SHA that wouldn't exist upstream (that would break
# `git submodule update` on a fresh clone). Instead the fixes live as patch files in
# patches/ and get applied to the submodule's working tree by this script — run it
# once after `git submodule update --init`, and again whenever the submodule is reset.
#
# Currently applied:
#   interplm-nnsight-compat.patch — adapts interplm/sae/intervention.py to the
#     installed nnsight's API (`.input` singular now returns the unwrapped first arg,
#     `.save()` returns the resolved object directly). Without it, steering
#     (scripts/run_steering.py) crashes in generation with an expand() shape error.
#     See docs/steering_history.md / SETUP.md.
#
# Idempotent: patches already applied are detected and skipped.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

SUB=InterPLM
[ -d "$SUB/interplm" ] || { echo "ERROR: $SUB submodule not checked out — run 'git submodule update --init' first" >&2; exit 1; }

for patch in patches/interplm-*.patch; do
    [ -e "$patch" ] || { echo "no patches/interplm-*.patch found — nothing to do"; break; }
    if git -C "$SUB" apply --reverse --check "../$patch" >/dev/null 2>&1; then
        echo "already applied: $patch"
    elif git -C "$SUB" apply --check "../$patch" >/dev/null 2>&1; then
        git -C "$SUB" apply "../$patch"
        echo "applied: $patch"
    else
        echo "ERROR: $patch does not apply cleanly to $SUB (submodule may have drifted)" >&2
        exit 1
    fi
done
