# ultra-graph Knowledge Index

**Last updated:** 2026-07-14
**Vault mode:** GitHub (codebase map, architecture wiki, API reference)
**Total pages:** 6

## Quick Navigation

| Category | Path | Color |
|----------|------|-------|
| **Concepts** | `wiki/concepts/` | Blue |
| **Entities** | `wiki/entities/` | Orange |
| **Sources** | `wiki/sources/` | Green |
| **Meta** | `wiki/meta/` | Purple |
| **References** | `wiki/references/` | Purple |

## Core Concepts

- [[three_level_byte_graph_architecture]] — Three-level byte-graph architecture
- [[straight_through_estimator]] — Straight-through estimator (STE)

## Key Entities (Components, Classes, Modules)

- [[tree]] — Tree (meso-level storage)
- [[tensor]] — Tensor (custom autograd engine)

## Ingested Sources

*No sources ingested yet. Drop files in `.raw/` and run `ingest [file]`.*

## Recent Session Logs

*No sessions logged yet.*

## Knowledge Gaps

- [ ] Architecture diagram with three-level byte-graph visualization
- [ ] Detailed autograd tape implementation notes
- [ ] BitNet b1.58 quantization mathematical foundation
- [ ] 5-ternary-per-byte packing algorithm explanation
- [ ] KV-cache implementation details

## Cross-Reference Health

- **Orphaned pages:** 0
- **Dead links:** 0
- **Missing cross-refs:** 0
- **Contradictions flagged:** 0

---

## How to Use This Index

1. **Query the wiki:** `what do you know about [topic]?`
2. **Ingest sources:** `ingest [file]` or `ingest [URL]`
3. **Save conversations:** `/save` or `/save [name]`
4. **Run health checks:** `lint the wiki`
5. **Autonomous research:** `/autoresearch [topic]`

## Hot Cache

Current session context is in [[hot]]. Read at session start, update at session end.

---

**Methodology:** Source-first synthesis with bidirectional cross-references. Every claim traces back to code or external sources. Contradictions flagged with `[!contradiction]` callouts.

**Based on:** [Andrej Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
**Plugin:** [claude-obsidian](https://github.com/AgriciDaniel/claude-obsidian)
