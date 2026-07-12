"""Live MediaWiki -> ultra-graph.

A tiny client over ``pymediawiki`` (import name ``mediawiki``) plus a BFS graph
builder: fetch pages from a Wikipedia (default hu.wikipedia.org), turn each page
into a node of a single sparse ``Tree`` (title + short summary kept in
``tree.adhoc["labels"]``), and wire a micro-edge wherever one fetched page links
to another already in the set. Page records are cached on disk as JSON so
repeated runs never refetch.

    from ultragraph.wiki import MediaWikiClient, build_wiki_graph
    ug = build_wiki_graph(["Honfoglalás"], lang="hu")

Requires the optional ``wiki`` extra:  ``uv sync --extra wiki``.
"""

import hashlib
import json
from pathlib import Path

from .core import Tree, UltraGraph

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "examples" / "data" / ".wikicache"


class MediaWikiClient:
    """Thin, on-disk-cached wrapper over ``mediawiki.MediaWiki``."""

    def __init__(self, lang="hu", user_agent="ultragraph-wiki/1.0", cache_dir=None):
        from mediawiki import MediaWiki  # lazy: only needed with the `wiki` extra

        self.lang = lang
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._wiki = MediaWiki(
            url=f"https://{lang}.wikipedia.org/w/api.php",
            user_agent=user_agent,
            rate_limit=True,  # be a polite API citizen; avoids intermittent throttling
        )

    # -- on-disk JSON cache --------------------------------------------------
    def _cache_path(self, kind, key):
        h = hashlib.sha1(f"{self.lang}:{kind}:{key}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{kind}_{h}.json"

    def _cached(self, kind, key, produce):
        path = self._cache_path(kind, key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        value = produce()
        if value is not None:  # never cache a failed fetch — let it retry next run
            try:
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        return value

    # -- fetch ---------------------------------------------------------------
    def _fetch_page(self, title):
        from mediawiki.exceptions import DisambiguationError

        try:
            try:
                p = self._wiki.page(title, auto_suggest=False)
            except DisambiguationError as e:
                options = list(getattr(e, "options", None) or [])
                if not options:
                    return None
                p = self._wiki.page(options[0], auto_suggest=False)
            return {"title": p.title, "summary": p.summary or "", "links": list(p.links), "url": p.url}
        except Exception:
            # PageError, intermittent API glitches (e.g. KeyError 'query'), or network — skip.
            return None

    def page(self, title):
        """Cached page record ``{title, summary, links, url}`` (``None`` if missing)."""
        return self._cached("page", title, lambda: self._fetch_page(title))

    def summary(self, title):
        """Cached page summary string, or ``None`` if the page is missing."""
        rec = self.page(title)
        return rec["summary"] if rec else None

    def links(self, title):
        """Cached list of outbound page titles (``[]`` if the page is missing)."""
        rec = self.page(title)
        return rec["links"] if rec else []


def build_wiki_graph(seed_titles, lang="hu", max_pages=40, client=None) -> UltraGraph:
    """BFS from ``seed_titles`` into one sparse ``Tree`` wrapped in an ``UltraGraph``.

    Every fetched page becomes a node; a micro-edge is added wherever one fetched
    page links to another fetched page. The node set is capped at ``max_pages`` and
    candidate links are visited in sorted order, so runs are deterministic given a
    stable cache. Missing pages and fetch errors are skipped with a printed note.
    """
    client = client or MediaWikiClient(lang=lang)
    records: dict[str, dict] = {}
    order: list[str] = []
    queue: list[str] = list(seed_titles)
    seen: set[str] = set()

    while queue and len(records) < max_pages:
        title = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        try:
            rec = client.page(title)
        except Exception as e:  # network blocked/unreachable — skip, keep going
            print(f"  skip (fetch error): {title} -- {type(e).__name__}: {e}")
            continue
        if not rec:
            print(f"  skip (missing/404): {title}")
            continue
        ctitle = rec["title"]
        if ctitle in records:
            continue
        records[ctitle] = rec
        order.append(ctitle)
        queue.extend(sorted(rec["links"]))

    idx = {title: i for i, title in enumerate(order)}
    tree = Tree(len(order), name="wiki")
    tree.adhoc["title"] = "wiki"
    tree.adhoc["lang"] = lang

    # micro-edges: each fetched page -> every fetched page it links to
    degree = [0] * len(order)
    seen_edges: set[tuple[int, int]] = set()
    for title in order:
        a = idx[title]
        for link in records[title]["links"]:
            b = idx.get(link)
            if b is None or b == a or (a, b) in seen_edges:
                continue
            seen_edges.add((a, b))
            tree.add_edge(a, b, 1)
            degree[a] += 1

    labels = []
    for i, title in enumerate(order):
        summary = " ".join(records[title]["summary"].split())[:160]
        labels.append((title, summary))
        tree.nodes[i] = min(127, degree[i])  # a byte per node: intra-set out-degree
    tree.adhoc["labels"] = labels

    ug = UltraGraph("wiki")
    ug.add(tree)
    return ug
