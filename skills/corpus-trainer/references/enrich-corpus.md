# Enrich Corpus

Enrich an existing text corpus with grounded Wikipedia facts — no LLM generation needed.

## What Success Looks Like

New paragraphs appended to the corpus containing grounded facts + Q&A pairs from Wikipedia, deduplicated against existing content, all idempotent and re-runnable.

## How it works

1. Identify seed topics relevant to the target language/culture
2. Use `ultragraph.wiki.MediaWikiClient` (requires `wiki` extra: `pymediawiki>=0.7`) to fetch page summaries
3. Format as definition + `Kérdés:/Válasz:` (Question:/Answer:) lines
4. Dedup against the existing corpus (paragraph-level exact match)
5. Append unique entries — idempotent, safe to re-run

## Usage

```sh
uv run --extra wiki python examples/enrich_corpus.py
```

The enricher caches Wikipedia pages on disk at `examples/data/.wikicache/` so repeated runs never refetch. Failed fetches (KeyError 'query', network errors) are NOT cached — they retry next run.

## Output format

```
Grounded: Honfoglalás — A honfoglalás a magyar törzsek Kárpát-medencébe való letelepedésének folyamata volt a 9-10. század fordulóján.
Kérdés: Mi volt a honfoglalás?
Válasz: A magyar törzsek letelepedése a Kárpát-medencében a 9-10. század fordulóján.
```

## Customization

- Edit the seed topics list in `examples/enrich_corpus.py` to target different cultural contexts
- Change `lang` parameter for other language Wikipedias
- The scraper respects `rate_limit=True` to be a polite API citizen
