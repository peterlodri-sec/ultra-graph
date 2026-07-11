# References — a graph theory reading list

`ultragraph` is a byte-graph, so it owes a debt to graph theory. This is a short,
opinionated reading list of Paul Erdős's most important graph-theory (and
graph-theorist-adjacent combinatorics) publications — the classics every graph
person eventually meets.

Citation details were checked against the Wikipedia articles for the named
theorems, the journals (Oxford Academic / Cambridge Core / Springer / AMS), and
the Rényi Institute's Erdős archive. A few conjecture citations are flagged where
a canonical volume/page could not be confirmed.

### Foundational papers

- **Erdős, P. & Szekeres, G. (1935).** "A combinatorial problem in geometry." *Compositio Mathematica*, 2, 463–470. [numdam](http://www.numdam.org/item/CM_1935__2__463_0/) — The Erdős–Szekeres theorem: any sequence longer than $(r-1)(s-1)$ has a monotone subsequence of length $r$ or $s$. A founding result of Ramsey theory (the "happy ending problem").
- **Erdős, P. & Stone, A. H. (1946).** "On the structure of linear graphs." *Bulletin of the AMS*, 52(12), 1087–1091. [doi:10.1090/S0002-9904-1946-08715-7](https://doi.org/10.1090/S0002-9904-1946-08715-7) — The Erdős–Stone theorem fixes the asymptotic maximum edge count of graphs forbidding a fixed subgraph — the "fundamental theorem of extremal graph theory," generalizing Turán.
- **Erdős, P. (1947).** "Some remarks on the theory of graphs." *Bulletin of the AMS*, 53(4), 292–294. [doi:10.1090/S0002-9904-1947-08785-1](https://doi.org/10.1090/S0002-9904-1947-08785-1) — The probabilistic lower bound $R(k,k) > 2^{k/2}$ for Ramsey numbers; launched the probabilistic method.
- **Erdős, P. & Rényi, A. (1959).** "On random graphs. I." *Publicationes Mathematicae Debrecen*, 6, 290–297. [doi:10.5486/PMD.1959.6.3-4.12](https://doi.org/10.5486/PMD.1959.6.3-4.12) — Introduced the $G(n,m)$ random-graph model; founded the theory of random graphs.
- **Erdős, P. & Rényi, A. (1960).** "On the evolution of random graphs." *Publ. Math. Inst. Hungar. Acad. Sci.*, 5, 17–61. [PDF](https://www.renyi.hu/~p_erdos/1960-10.pdf) (no DOI) — The phase-transition / giant-component paper; foundational for modern network science.
- **Erdős, P. & Gallai, T. (1960).** "Gráfok előírt fokszámú pontokkal" [Graphs with vertices of prescribed degrees]. *Matematikai Lapok*, 11, 264–274. [PDF](https://www.renyi.hu/~p_erdos/1961-05.pdf) (no DOI) — The Erdős–Gallai theorem: the inequalities that decide when an integer sequence is *graphic* (a degree sequence of a simple graph).
- **Erdős, P. (1959).** "Graph theory and probability." *Canadian Journal of Mathematics*, 11, 34–38. [doi:10.4153/CJM-1959-003-9](https://doi.org/10.4153/CJM-1959-003-9) — Graphs with simultaneously arbitrarily high girth *and* high chromatic number exist — a celebrated nonconstructive result.
- **Erdős, P., Ko, C. & Rado, R. (1961).** "Intersection theorems for systems of finite sets." *Quarterly J. of Mathematics* (Oxford, 2nd Ser.), 12, 313–320. [doi:10.1093/qmath/12.1.313](https://doi.org/10.1093/qmath/12.1.313) — The Erdős–Ko–Rado theorem bounds the largest intersecting family of $k$-subsets; a cornerstone of extremal set theory.
- **Erdős, P., Rényi, A. & Sós, V. T. (1966).** "On a problem of graph theory." *Studia Sci. Math. Hungar.*, 1, 215–235. [PDF](https://www.renyi.hu/~p_erdos/1966-06.pdf) (no DOI) — The friendship theorem: if every two vertices share exactly one common neighbour, the graph is a windmill.

### Conjectures

- **Erdős–Faber–Lovász** (1972; recorded in Erdős, P. (1981), "On the combinatorial problems which I would most like to see solved," *Combinatorica*, 1(1), 25–42, [doi:10.1007/BF02579174](https://doi.org/10.1007/BF02579174)). [wiki](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Faber%E2%80%93Lov%C3%A1sz_conjecture) — The union of $k$ cliques of size $k$ pairwise sharing ≤1 vertex is $k$-colourable. Proved for all large $k$ in 2023 (Kang–Kelly–Kühn–Methuku–Osthus).
- **Erdős–Sós** (posed ~1962; stated in Erdős, P. (1964), "Extremal problems in graph theory," in *Theory of Graphs and Its Applications* (Proc. Sympos. Smolenice 1963), Czechoslovak Acad. Sci., pp. 29–36; no DOI). [Erdős list](https://www.emis.de/classics/Erdos/extrram.htm) — Any graph with more than $(k-1)n/2$ edges contains every tree with $k$ edges. *(Exact page range as commonly cited; volume not viewed directly.)*
- **Erdős–Hajnal** (Erdős, P. & Hajnal, A. (1977), "On spanned subgraphs of graphs," in *Contributions to Graph Theory and Its Applications*, Ilmenau, pp. 80–96; often cited via their (1989) "Ramsey-type theorems," *Discrete Applied Mathematics*, 25(1–2), 37–52). [wiki](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Hajnal_conjecture) — Graphs omitting a fixed induced subgraph contain a clique or independent set of polynomial size $n^{c}$. Major open problem.
- **Erdős–Gyárfás** (Erdős & Gyárfás, 1995). [wiki](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Gy%C3%A1rf%C3%A1s_conjecture) · [West](http://dwest.web.illinois.edu/openp/2powcyc.html) — Every graph with minimum degree ≥ 3 has a cycle whose length is a power of two. *(Posed at a conference; no single origin paper — citation details unverified.)*

---

Compiled 2026-07-11; verify anything you plan to cite. `ultragraph` genesis `251e6ea`.
