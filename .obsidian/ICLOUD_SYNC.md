# Obsidian iCloud Sync Setup

**Goal:** Sync ultra-graph wiki between GitHub and Obsidian in iCloud Drive

---

## Option 1: iCloud Drive (Recommended for macOS + iOS)

### Step 1: Move vault to iCloud

```bash
# Create Obsidian vault folder in iCloud
mkdir -p ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/

# Copy ultra-graph wiki to iCloud
cp -r ~/workspace/peterlodri-sec/ultra-graph/wiki ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/ultra-graph-wiki
cp -r ~/workspace/peterlodri-sec/ultra-graph/.obsidian ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/ultra-graph-wiki/.obsidian
```

### Step 2: Open in Obsidian

1. Open Obsidian
2. **Manage Vaults** → **Open folder as vault**
3. Navigate to: `iCloud Drive → obsidian-vaults → ultra-graph-wiki`
4. Click **Open**

### Step 3: Enable iCloud Sync

Obsidian automatically syncs via iCloud when the vault is in iCloud Drive. No additional setup needed.

### Step 4: Keep GitHub in Sync

Run this after Obsidian sessions:

```bash
# From your workspace directory
cd ~/workspace/peterlodri-sec/ultra-graph

# Pull latest from iCloud (if you edited on iPhone/iPad)
rsync -av --delete ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/ultra-graph-wiki/wiki/ ./wiki/
rsync -av ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/ultra-graph-wiki/.obsidian/ ./ .obsidian/

# Commit and push
git add -A wiki/ .obsidian/
git commit -m "chore(wiki): sync from iCloud $(date '+%Y-%m-%d')"
git push origin main
```

Or use the sync script:
```bash
bash .obsidian/sync.sh
```

---

## Option 2: Obsidian Sync (Paid, $10/month)

### Step 1: Subscribe

1. Open Obsidian
2. **Settings** → **Sync**
3. Click **Subscribe** ($10/month or $96/year)

### Step 2: Enable Sync

1. **Settings** → **Sync**
2. Toggle **Enable sync**
3. Wait for initial sync to complete

### Step 3: Access on Other Devices

1. Install Obsidian on iPhone/iPad
2. Sign in with same account
3. Enable sync in settings
4. Vault automatically appears

### Step 4: Keep GitHub in Sync

Same as Option 1, Step 4 — use rsync or sync script.

---

## Option 3: Git-Only Sync (What We Have Now)

### On macOS (Desktop)

1. Open Obsidian
2. **Manage Vaults** → **Open folder as vault**
3. Select: `~/workspace/peterlodri-sec/ultra-graph/`
4. Install **Obsidian Git** plugin (auto-commits every 5 minutes)

### On iPhone/iPad

**Option A: Working Copy (Git client)**
1. Install [Working Copy](https://workingcopyapp.com/) ($20 one-time)
2. Clone `https://github.com/peterlodri-sec/ultra-graph`
3. Install Obsidian
4. Open vault from Working Copy

**Option B: GitHub Desktop + iCloud**
1. Install GitHub Desktop on Mac
2. Clone ultra-graph repo
3. Move `.obsidian` and `wiki/` to iCloud (see Option 1)
4. Use rsync to keep in sync

---

## Recommended Workflow

### Daily Use (macOS)

1. **Morning:** Open Obsidian → ultra-graph vault (in workspace or iCloud)
2. **During day:** Use `/wiki` in Claude Code, take notes
3. **Evening:** Run `bash .obsidian/sync.sh` to commit and push

### Mobile Use (iPhone/iPad)

1. **Setup:** Use Option 1 (iCloud Drive) or Option 2 (Obsidian Sync)
2. **Edit on device:** Notes sync automatically via iCloud/Obsidian Sync
3. **Sync to GitHub:** Run rsync or sync script on Mac after mobile session

### Team Collaboration

1. Everyone clones from GitHub
2. Use `git pull` before editing
3. Use `bash .obsidian/sync.sh` to push changes
4. Resolve conflicts via Git (Obsidian Git plugin helps)

---

## Automation Scripts

### Auto-sync on save (macOS)

Create `~/Library/LaunchAgents/com.ultragraph.obsidian-sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ultragraph.obsidian-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/peter.lodri/workspace/peterlodri-sec/ultra-graph/.obsidian/sync.sh</string>
    </array>
    <key>StartInterval</key>
    <integer>300</integer> <!-- Run every 5 minutes -->
    <key>WorkingDirectory</key>
    <string>/Users/peter.lodri/workspace/peterlodri-sec/ultra-graph</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.ultragraph.obsidian-sync.plist
```

### One-command sync

Add to `~/.zshrc` or `~/.bashrc`:

```bash
alias wiki-sync='cd ~/workspace/peterlodri-sec/ultra-graph && bash .obsidian/sync.sh'
```

Then just run: `wiki-sync`

---

## Troubleshooting

### iCloud sync conflicts

If you see `.sync-conflict` files:
```bash
# Find conflicts
find ~/Library/Mobile\ Documents/com~apple~CloudDocs/obsidian-vaults/ultra-graph-wiki -name "*.sync-conflict*"

# Review and merge manually
# Delete conflict files after resolving
```

### Git conflicts

```bash
# Pull latest
git pull origin main

# If conflicts, open in Obsidian and resolve
# Then:
git add -A
git commit -m "fix: resolve merge conflicts"
git push origin main
```

### Obsidian Git not auto-committing

1. Open Obsidian
2. **Settings** → **Community Plugins** → **Obsidian Git**
3. Set **Auto save interval** to 5 minutes
4. Toggle **Auto commit on change**

---

## Current Setup

- ✅ GitHub repo: `https://github.com/peterlodri-sec/ultra-graph`
- ✅ Pre-commit hook with ruff auto-fix
- ✅ Sync script: `.obsidian/sync.sh`
- ✅ Wiki in: `./wiki/`
- ✅ Obsidian config in: `./.obsidian/`

**Next step:** Choose Option 1, 2, or 3 above and follow the setup steps.

---

**Last updated:** 2026-07-14
