#!/bin/bash
# Reproducible submission commit: the tree of SRC (default HEAD) without the working-repo-only paths, committed with
# no parents, a fixed author/committer and the commit date of SRC -> the same SRC always gives the same hash.
# Result: branch submit-048 (replaced) checked out in the worktree /tmp/submit-048. Never pushes.
# Usage: bash research/diag/v16/build_orphan.sh [SRC]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=$(git rev-parse --verify "${1:-HEAD}^{commit}"); SHORT=$(git rev-parse --short "$SRC")
EXCLUDE=(research HANDOFF.md
         rust/team048_accel/benches/data/frame_points.f64 rust/team048_accel/benches/data/plane_points.f64
         rust/team048_accel/benches/data/frame_mask.u8 rust/team048_accel/benches/data/frame.json)
WT=/tmp/submit-048; BR=submit-048
IDX=$(mktemp); trap 'rm -f "$IDX"' EXIT
GIT_INDEX_FILE=$IDX git read-tree "$SRC"
GIT_INDEX_FILE=$IDX git rm -r -q --cached --ignore-unmatch "${EXCLUDE[@]}" >/dev/null
TREE=$(GIT_INDEX_FILE=$IDX git write-tree)
DATE=$(git show -s --format=%cI "$SRC")
C=$(GIT_AUTHOR_NAME=team048-local GIT_AUTHOR_EMAIL=local@invalid GIT_COMMITTER_NAME=team048-local GIT_COMMITTER_EMAIL=local@invalid \
    GIT_AUTHOR_DATE=$DATE GIT_COMMITTER_DATE=$DATE git commit-tree "$TREE" -m "Team 048 solution (DeftSort $SHORT)")
if git worktree list --porcelain | grep -qx "worktree $WT"; then git worktree remove --force "$WT"; fi
git branch -f "$BR" "$C" >/dev/null
git worktree add -q "$WT" "$BR"
# verification: tree = SRC minus EXCLUDE, no excluded path, no parents
EXP=$(git ls-tree -r "$SRC" | grep -v -E $'\t(research/|HANDOFF\\.md$|rust/team048_accel/benches/data/(frame_points\\.f64|plane_points\\.f64|frame_mask\\.u8|frame\\.json)$)' | sha256sum)
GOT=$(git ls-tree -r "$BR" | sha256sum)
[ "$EXP" = "$GOT" ] || { echo "STOP: tree mismatch"; exit 1; }
[ -z "$(git show -s --format=%P "$BR")" ] || { echo "STOP: orphan has parents"; exit 1; }
echo "orphan $C tree $TREE from $SHORT: $(git ls-tree -r --name-only "$BR" | wc -l) files, $(git ls-tree -r -l "$BR" | awk '{s+=$4} END {printf "%.2f MB", s/1e6}')"
