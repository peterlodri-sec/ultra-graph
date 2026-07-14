#!/usr/bin/env bash
# Sync Obsidian vault after compact
# Run this after Obsidian Git performs a compact to persist wiki changes
# Also runs ruff auto-fix before committing
#
# Usage: bash .obsidian/sync.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Syncing Obsidian vault..."

# ── 1. Ruff auto-fix ─────────────────────────────────────────────────────────
echo "→ Running ruff check --fix..."
if command -v ruff >/dev/null 2>&1; then
    ruff check --fix ultragraph tests examples assets 2>/dev/null || true
    echo "✓ Ruff auto-fix complete"
else
    echo "ℹ Ruff not found, skipping auto-fix"
fi

# ── 2. Stage ruff changes ────────────────────────────────────────────────────
git add -u ultragraph/ tests/ examples/ assets/ 2>/dev/null || true

# ── 3. Stage wiki changes ────────────────────────────────────────────────────
git add -A wiki/ .obsidian/ .raw/ WIKI.md SETUP_SUMMARY.md 2>/dev/null || true

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "✓ No changes to commit"
else
    # Commit with timestamp (--no-verify to avoid hook recursion)
    git commit --no-verify -m "chore(wiki): sync Obsidian vault $(date '+%Y-%m-%d %H:%M')"
    echo "✓ Committed wiki changes"
fi

echo "✓ Sync complete"
