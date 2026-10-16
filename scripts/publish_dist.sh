#!/usr/bin/env bash
# Publish THIS market's results to the single-commit `dist` branch of the PRIVATE data repo.
#
# Markets run in parallel, so each job only replaces its own folder (dist/<market>/ and its logs) on
# top of the latest remote dist, and pushes with --force-with-lease: if another market published in
# between, the push is refused and we rebuild on the new tip and retry. History is always squashed to
# one commit, so the branch never grows.
set -euo pipefail
M="${MARKET:?MARKET not set}"
REMOTE="${DIST_REMOTE:-https://x-access-token:${DATA_TOKEN:-}@github.com/${DATA_REPO:-}.git}"
SRC="$(pwd)/dist"
for attempt in 1 2 3 4 5 6; do
  W="$(mktemp -d)"
  if git clone -q --depth 1 --branch dist "$REMOTE" "$W" 2>/dev/null; then
    LEASE="refs/heads/dist:$(git -C "$W" rev-parse HEAD)"
  else
    git -C "$W" init -q -b dist
    LEASE="refs/heads/dist:"
  fi
  rm -rf "$W/$M"; mkdir -p "$W/$M" "$W/_logs"
  cp -r "$SRC/$M/." "$W/$M/"
  for f in "$SRC"/_logs/"$M".log "$SRC"/_logs/"$M"-model.log; do [ -f "$f" ] && cp "$f" "$W/_logs/"; done
  cd "$W"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > updated_at.txt
  git config user.name "chip-radar-bot"
  git config user.email "chip-radar-bot@users.noreply.github.com"
  git checkout -q --orphan publish
  git add -A
  git commit -qm "data $M $(cat updated_at.txt)"
  if git push -q --force-with-lease="$LEASE" "$REMOTE" publish:dist; then
    echo "dist published ($M, attempt $attempt)"
    exit 0
  fi
  cd - >/dev/null
  echo "dist moved while publishing $M; retrying"
  sleep $((RANDOM % 20 + 5))
done
echo "::error::could not publish dist for $M"
exit 1
