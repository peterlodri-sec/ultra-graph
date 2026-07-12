"""Gather + enrich the Hungarian corpus — **no LLM in the loop**.

Deterministic, grounded enrichment: pull factual summaries from Hungarian Wikipedia
(via ``ultragraph.wiki``) and turn them into definition lines + ``Kérdés:/Válasz:``
pairs, optionally add public-domain Gutenberg prose, dedup against what's already in
the corpus, and append only the new unique paragraphs. Unlike the model's self-labelled
answers, everything here is real text — a clean, repeatable data pipeline.

    uv run --extra wiki python examples/enrich_corpus.py            # wiki facts + Q&A
    WITH_BOOKS=1 uv run --extra wiki python examples/enrich_corpus.py   # + Gutenberg prose
"""

import os
import re
import urllib.request

from ultragraph.wiki import MediaWikiClient

CORPUS = os.path.join(os.path.dirname(__file__), "data", "hungarian_corpus.txt")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# real hu.wikipedia article titles — history, geography, culture
TITLES = [
    "Honfoglalás", "I. István magyar király", "Mátyás király", "Mohácsi csata",
    "Az 1848–49-es forradalom és szabadságharc", "Trianoni békeszerződés",
    "Az 1956-os forradalom", "Budapest", "Balaton", "Puszta (tájegység)",
    "Gulyás (étel)", "Magyar nyelv", "Petőfi Sándor", "Arany János", "Duna",
]
# optional public-domain prose (Gutenberg ids) — only fetched with WITH_BOOKS=1
BOOKS = [(64580, "Nagyvárosi képek"), (68877, "A tóparti gyilkosság")]

_PG_START = re.compile(r"\*\*\*\s*START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
_PG_END = re.compile(r"\*\*\*\s*END OF TH[EI]S? PROJECT GUTENBERG EBOOK", re.S)


def first_sentence(text: str, cap: int = 400) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s", s)
    return (parts[0] if parts else s)[:cap].strip()


def existing_paragraphs(text: str) -> set:
    return {p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()}


def wiki_blocks(client: MediaWikiClient):
    """Grounded definition + Q&A blocks from hu.wikipedia summaries."""
    for title in TITLES:
        summ = client.summary(title)
        if not summ:
            print(f"  wiki skip: {title}")
            continue
        sent = first_sentence(summ)
        if len(sent) < 20:
            continue
        yield f"{title}: {sent}"
        yield f"Kérdés: Mit tudunk erről: {title}?\nVálasz: {sent}"


def gutenberg_blocks(ids):
    for pgid, label in ids:
        try:
            req = urllib.request.Request(f"https://www.gutenberg.org/cache/epub/{pgid}/pg{pgid}.txt",
                                         headers={"user-agent": UA})
            raw = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  book skip {pgid}: {e}")
            continue
        m1, m2 = _PG_START.search(raw), _PG_END.search(raw)
        body = raw[m1.end():m2.start()] if (m1 and m2) else raw
        body = re.sub(r"\r\n?", "\n", body)
        for para in re.split(r"\n\s*\n", body):
            para = re.sub(r"[ \t]+", " ", para).strip()
            if len(para) >= 60:
                yield para


def main() -> None:
    text = open(CORPUS, encoding="utf-8").read() if os.path.exists(CORPUS) else ""
    seen = existing_paragraphs(text)
    before = len(seen)

    new: list[str] = []

    def add(block: str):
        b = block.strip()
        if b and b not in seen:
            seen.add(b)
            new.append(b)

    print("gathering hu.wikipedia facts (non-LLM)...")
    client = MediaWikiClient(lang="hu")
    for block in wiki_blocks(client):
        add(block)
    wiki_added = len(new)
    print(f"  +{wiki_added} unique wiki blocks (definitions + Q&A)")

    if os.environ.get("WITH_BOOKS"):
        print("gathering Gutenberg prose...")
        for block in gutenberg_blocks(BOOKS):
            add(block)
        print(f"  +{len(new) - wiki_added} unique prose paragraphs")

    if not new:
        print("nothing new to add — corpus already enriched.")
        return
    header = "\n\n# Enriched (non-LLM): grounded facts from hu.wikipedia\n"
    with open(CORPUS, "a", encoding="utf-8") as f:
        f.write(header + "\n\n".join(new) + "\n")
    added_bytes = len((header + "\n\n".join(new)).encode("utf-8"))
    print(f"\nappended {len(new)} unique paragraphs (+{added_bytes:,} bytes)")
    print(f"corpus paragraphs: {before} -> {before + len(new)}  |  {os.path.getsize(CORPUS):,} bytes")


if __name__ == "__main__":
    main()
