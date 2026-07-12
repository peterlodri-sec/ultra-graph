"""Fetch a public-domain **Hungarian-language** corpus from Project Gutenberg.

A handful of canonical Hungarian works (Arany János's epic *Buda halála* plus some
19th–early-20th-c. prose), Gutenberg boilerplate stripped, concatenated and capped —
the training set for a genuinely Hungarian (not Latin) byte-level ternary GPT.

    uv run python examples/fetch_hungarian.py
"""

import os
import re
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "data", "hungarian_corpus.txt")
CAP = 450_000  # bytes — bound the corpus (and training time)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# (gutenberg id, short label) — all public domain
WORKS = [
    (41699, "Arany János — Buda halála"),
    (62239, "Furcsa emberek — elbeszélések"),
    (64580, "Nagyvárosi képek"),
    (68877, "A tóparti gyilkosság és egyéb elbeszélések"),
    (34759, "A vörös regina — regény"),
]

START = re.compile(r"\*\*\*\s*START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\*\s*END OF TH[EI]S? PROJECT GUTENBERG EBOOK", re.S)


def fetch(pgid: int) -> str | None:
    for url in (f"https://www.gutenberg.org/cache/epub/{pgid}/pg{pgid}.txt",
                f"https://www.gutenberg.org/ebooks/{pgid}.txt.utf-8"):
        try:
            req = urllib.request.Request(url, headers={"user-agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def strip_pg(raw: str) -> str:
    m1, m2 = START.search(raw), END.search(raw)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    body = re.sub(r"\r\n?", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def main() -> None:
    parts, total = [], 0
    for pgid, label in WORKS:
        raw = fetch(pgid)
        if not raw:
            print(f"  skip {pgid} ({label}) — fetch failed")
            continue
        body = strip_pg(raw)
        parts.append(body)
        total += len(body.encode("utf-8"))
        print(f"  + {label}: {len(body):,} chars")
        if total >= CAP:
            break
    text = ("\n\n".join(parts))[:CAP] + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nwrote {OUT}  ({len(text):,} chars, {len(text.encode('utf-8')):,} bytes)")
    print("--- head ---")
    print(text[:400])


if __name__ == "__main__":
    main()
