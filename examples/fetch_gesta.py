"""Fetch + clean the corpus: Anonymus, *Gesta Hungarorum* (c. 1200).

The founding chronicle of the Hungarian nation, written by "P. dictus magister",
notary of King Béla. The Latin original is public domain; we pull the raw wikitext
from Latin Wikisource, strip the markup, and write a plain-text corpus for training
the byte-level ternary GPT in `anonymus_lm.py`.

    uv run python examples/fetch_gesta.py
"""
from __future__ import annotations

import os
import re
import urllib.request

URL = "https://la.wikisource.org/wiki/Gesta_Hungarorum?action=raw"
OUT = os.path.join(os.path.dirname(__file__), "data", "gesta_hungarorum.txt")


def strip_wikitext(raw: str) -> str:
    s = raw
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)         # HTML comments
    s = re.sub(r"<ref[^>]*?/>", "", s)                    # self-closing refs
    s = re.sub(r"<ref.*?</ref>", "", s, flags=re.S)       # ref bodies
    s = re.sub(r"\{\{.*?\}\}", "", s, flags=re.S)         # templates
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)  # [[a|b]] -> b, [[a]] -> a
    s = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", s)   # [url label] -> label
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)               # stray html tags
    s = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1.", s, flags=re.M)  # ==heading== -> "heading."
    s = s.replace("'''", "").replace("''", "")            # bold / italic
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip() + "\n"


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "ultragraph-anonymus/1.0 (educational)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8")
    text = strip_wikitext(raw)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT}  ({len(text):,} chars, {len(text.encode('utf-8')):,} bytes)")
    print("--- head ---")
    print(text[:400])


if __name__ == "__main__":
    main()
