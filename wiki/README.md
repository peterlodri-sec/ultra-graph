# ultra-graph Knowledge Base

Welcome to the **ultra-graph Obsidian vault** — a persistent, compounding knowledge base for the ultra-graph project.

## Quick Start

### 1. Open in Obsidian

1. Open Obsidian
2. **Manage Vaults** → **Open folder as vault**
3. Select: `ultra-graph/`
4. Enable community plugins when prompted

### 2. Install Community Plugins

Go to **Settings** → **Community Plugins** and install:

- **Dataview** — Powers dynamic queries and card views
- **Templater** — Auto-fills frontmatter from templates
- **Obsidian Git** — Auto-commits vault changes every 15 minutes

### 3. Start Using

In Claude Code (in the `ultra-graph/` directory):

```
/wiki
```

Claude will scaffold the knowledge base and guide you through setup.

## Wiki Structure

```
wiki/
├── index.md           # Master index with Dataview queries
├── hot.md             # Session context (hot cache)
├── log.md             # Chronological session log
├── overview.md        # Project overview
├── concepts/          # Abstract ideas (architecture, algorithms)
├── entities/          # Concrete things (classes, modules, functions)
├── sources/           # Ingested documents (papers, URLs, files)
├── meta/              # Session notes, diagrams, milestones
├── references/        # Technical references (APIs, configs)
└── questions/         # FAQ-style pages
```

## Core Commands

| Command | Description |
|---------|-------------|
| `/wiki` | Setup, scaffold, or continue wiki |
| `ingest [file]` | Process a source document (creates 8-15 wiki pages) |
| `ingest all of these` | Batch process multiple sources |
| `what do you know about X?` | Query with citations from wiki pages |
| `/save` | File current conversation as wiki note |
| `/autoresearch [topic]` | Autonomous research loop (search, fetch, synthesize) |
| `lint the wiki` | Health check (orphans, dead links, gaps) |
| `update hot cache` | Refresh session context |

## Pre-configured Features

### Graph View

- **Filtered to:** `wiki/` only
- **Color groups:**
  - Orange: `wiki/entities/`
  - Blue: `wiki/concepts/`
  - Green: `wiki/sources/`
  - Purple: `wiki/meta/`, `wiki/references/`

### CSS Snippets

- **vault-colors:** Color-codes wiki folders in file explorer
- **ITS-Dataview-Cards:** Turns Dataview queries into visual card grids
- **ITS-Image-Adjustments:** Fine-grained image sizing (append `|100` to embeds)

### Pre-installed Plugins

- **Calendar:** Right-sidebar calendar with word count and task dots
- **Thino:** Quick memo capture panel
- **Excalidraw:** Freehand drawing, image annotation
- **Banners:** Header images via `banner:` frontmatter

## Methodology

This wiki follows the **LLM Wiki pattern** by Andrej Karpathy:

1. **Source-first synthesis:** Every claim traces back to code or external sources
2. **Bidirectional cross-references:** `[[Concept]]` ↔ `[[Entity]]`
3. **Contradiction flagging:** `[!contradiction]` callouts with sources
4. **Hot cache:** Session context persists between conversations
5. **Incremental growth:** Knowledge compounds with every ingest

## Tips

### Efficient Querying

```
# Quick fact check
what is the packing ratio for ternary weights?

# Deep dive with citations
explain how autograd works in ultra-graph, cite specific files

# Architecture questions
how does the three-level byte-graph architecture work?
```

### Saving Conversations

```
# Save with auto-generated title
/save

# Save with specific title
/save "GPT KV-cache implementation notes"
```

## Troubleshooting

### Graph colors reset

If graph colors reset after closing Obsidian:
1. Open **Graph view**
2. Go to **Color groups**
3. Re-add them once
4. They persist permanently after that

### Dataview queries not working

Install the **Dataview** community plugin:
1. **Settings** → **Community Plugins**
2. Browse → Search "Dataview"
3. Install and enable

## Related Documentation

- [AGENTS.md](../AGENTS.md) — Project-specific agent instructions
- [README.md](../README.md) — Project overview
- [WIKI.md](../WIKI.md) — Wiki configuration guide

## Credits

- **Pattern:** [Andrej Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- **Plugin:** [claude-obsidian](https://github.com/AgriciDaniel/claude-obsidian) by Daniel Agrici

---

**Vault mode:** GitHub (codebase map, architecture wiki, API reference)
**Last updated:** 2026-07-14
