# Fetch Corpus

Fetch public-domain text for a target language from Project Gutenberg or custom sources.

## What Success Looks Like

A `.txt` corpus file at `examples/data/{lang}_corpus.txt` with clean, deduplicated text ready for enrichment and training. The pipeline logs what was fetched, how many bytes/paragraphs, and any skipped entries.

## Sources

| Source | How | When to use |
|--------|-----|-------------|
| Project Gutenberg | `examples/fetch_hungarian.py` pattern (via `requests` + HTML scraped books) | Known public-domain literature |
| Wikipedia | `ultragraph.wiki.build_wiki_graph()` (optional `wiki` extra) | Grounded factual content |
| Custom URL/text | Direct download or paste | Any public-domain source |
| Existing corpus | Already in `examples/data/` | Re-training or enrichment only |

## Process

1. Identify available public-domain texts for the target language (Project Gutenberg has texts in ~50 languages)
2. Fetch each text, decode to UTF-8, split into paragraphs
3. Deduplicate: skip paragraphs that already exist in the corpus file (by exact match or fuzzy near-dup)
4. Append new content
5. Report stats: "Corpus now 495,326 bytes, 2460 paragraphs (+180 new from Gutenberg)"

## Existing patterns

- `examples/fetch_hungarian.py` — fetches Hungarian literature from Gutenberg, saves to `examples/data/hungarian_corpus.txt`
- `examples/fetch_gesta.py` — fetches Latin chronicle text
- Both produce plain UTF-8 text files with one paragraph per line (approximately)

## Idempotency

Every fetch must be safe to re-run. Use `urllib`/`requests` with timeouts. Don't overwrite existing corpus without user confirmation. Log skipped duplicates.
