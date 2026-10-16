#!/usr/bin/env bash
# Publish ./dist as a single-commit `dist` branch of the PRIVATE data repo (no history => never bloats).
set -euo pipefail
cd dist
rm -rf .git
git init -q -b dist
git config user.name "chip-radar-bot"
git config user.email "chip-radar-bot@users.noreply.github.com"
date -u +"%Y-%m-%dT%H:%M:%SZ" > updated_at.txt
git add -A
git commit -qm "data $(cat updated_at.txt)"
git push -qf "https://x-access-token:${DATA_TOKEN}@github.com/${DATA_REPO}.git" dist:dist
echo "dist published"
