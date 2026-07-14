# ultra-graph Wiki Configuration

**Vault Mode:** B (GitHub) — Codebase map, architecture wiki, API reference

This wiki follows the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) to maintain persistent, compounding knowledge about the ultra-graph project.

## Project Context

ultra-graph is a **pure-Python byte-graph that is a 1-bit (ternary) LLM**. MIT-licensed, package name `ultragraph-1bit` on PyPI.

## Wiki Structure

```
wiki/
├── index.md              # Master index of all knowledge
├── hot.md                # Hot cache (session context)
├── log.md                # Session log
├── overview.md           # Project overview
├── concepts/             # Abstract ideas (quantization, autograd, etc.)
├── entities/             # Concrete things (GPT, Tree, Tensor, etc.)
├── sources/              # Ingested documents (papers, articles, URLs)
├── meta/                 # Meta information, diagrams, sessions
└── references/           # Technical references (APIs, configs, etc.)
```

## Commands

| Command | Description |
|---------|-------------|
| `/wiki` | Setup, scaffold, or continue wiki |
| `ingest [file]` | Process a source document |
| `what do you know about X?` | Query with citations |
| `/save` | File current conversation as wiki note |
| `lint the wiki` | Health check (orphans, dead links, gaps) |

## Setup Checklist

- [x] Skills installed in `./skills/`
- [x] Obsidian configuration in `./.obsidian/`
- [x] Graph view filtered to `wiki/` with color groups
- [x] CSS snippets enabled
- [ ] Community plugins to install: Dataview, Templater, Obsidian Git

## Next Steps

1. Open Obsidian → Manage Vaults → Open folder as vault → select `ultra-graph/`
2. Enable community plugins when prompted
3. Install Dataview, Templater, Obsidian Git from Community Plugins
4. Type `/wiki` in Claude Code to scaffold the knowledge base
