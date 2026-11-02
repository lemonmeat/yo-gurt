#!/usr/bin/env bash
# Fetch only Simulation/SO101 (MuJoCo scene + URDF + meshes) from TheRobotStudio/SO-ARM100 at a
# pinned commit into third_party/SO-ARM100. The rest of that repo is CAD/STL for printing, so we
# do a partial, sparse clone instead of a submodule. third_party/ is gitignored.
#
# Usage: scripts/fetch_so_arm100.sh            (idempotent; re-run after changing SO_ARM100_REV)
set -euo pipefail

SO_ARM100_REV="${SO_ARM100_REV:-a758567c3978dfeefe282ede0500085a48fe8f78}"  # main, 2026-10-06
URL="https://github.com/TheRobotStudio/SO-ARM100.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/SO-ARM100"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --quiet --filter=blob:none --no-checkout "$URL" "$DEST"
fi

git -C "$DEST" sparse-checkout set --cone Simulation/SO101
git -C "$DEST" fetch --quiet --filter=blob:none origin "$SO_ARM100_REV"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$SO_ARM100_REV"

test -f "$DEST/Simulation/SO101/scene.xml" || { echo "scene.xml missing after fetch" >&2; exit 1; }
echo "SO-ARM100 @ ${SO_ARM100_REV:0:12} -> $DEST/Simulation/SO101"
