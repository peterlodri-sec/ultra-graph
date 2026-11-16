import math

from ultragraph import Tree, UltraGraph
from ultragraph.zerocpu import (
    FRAC_BITS,
    INT8_MAX,
    INT8_MIN,
    ONE,
    PqError,
    SpacetimeGraph,
    apply_pq,
    evolve,
    fx_dist,
    fx_div,
    fx_dot,
    fx_matvec,
    fx_mul,
    fx_sqrt,
    fx_sub,
    parse_pq,
    saturate,
    to_float,
    to_fx,
    to_int,
)


def _raises(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def _world(n=4):
    g = SpacetimeGraph("w")
    for _ in range(n):
        g.add_node(state=0)
    return g


# -- fixed point --------------------------------------------------------------
def test_fixed_exact_conversions():
    assert FRAC_BITS == 16 and ONE == 65536
    assert to_fx(2) == 131072
    assert to_fx(0.5) == 32768
    assert to_fx("3/4") == 49152
    assert to_fx(-1.25) == -81920
    assert to_float(to_fx(0.25)) == 0.25
    assert to_int(to_fx(2.5)) == 3 and to_int(to_fx(-2.5)) == -3


def test_fixed_rejects_bad_input():
    assert _raises(ValueError, to_fx, float("nan"))
    assert _raises(ValueError, to_fx, float("inf"))
    assert _raises(TypeError, to_fx, True)


def test_fixed_mul_div_are_symmetric_about_zero():
    assert fx_mul(to_fx(1.5), to_fx(-2.0)) == to_fx(-3.0)
    assert fx_mul(to_fx(-1.5), to_fx(-2.0)) == to_fx(3.0)
    a, b = fx_div(to_fx(-1.0), to_fx(3.0)), fx_div(to_fx(1.0), to_fx(3.0))
    assert a == -b and a == -to_fx("1/3")
    assert _raises(ZeroDivisionError, fx_div, ONE, 0)


def test_fixed_dot_is_exact_and_single_rounding():
    a = [to_fx(1), to_fx(2), to_fx(3)]
    b = [to_fx(4), to_fx(-1), to_fx(0.5)]
    assert fx_dot(a, b) == to_fx(3.5)
    assert fx_matvec([a, b], a) == [to_fx(14), to_fx(3.5)]
    assert _raises(ValueError, fx_dot, a, b[:2])


def test_fixed_sqrt_and_distance_exact():
    assert fx_sqrt(to_fx(2.25)) == to_fx(1.5)
    assert fx_sqrt(0) == 0
    assert _raises(ValueError, fx_sqrt, -1)
    origin, corner = (0, 0, 0), (to_fx(3), to_fx(4), 0)
    assert fx_dist(origin, corner) == to_fx(5)
    assert _raises(ValueError, fx_dist, origin, corner[:2])


def test_saturate_clamps_and_is_idempotent():
    assert saturate(200, INT8_MIN, INT8_MAX) == INT8_MAX
    assert saturate(-300, INT8_MIN, INT8_MAX) == INT8_MIN
    assert saturate(7, INT8_MIN, INT8_MAX) == 7
    assert saturate(saturate(9999, INT8_MIN, INT8_MAX), INT8_MIN, INT8_MAX) == INT8_MAX
    assert _raises(ValueError, saturate, 0, 5, 1)


def test_fixed_add_sub_are_exact():
    assert fx_sub(to_fx(2.5), to_fx(0.25)) == to_fx(2.25)
    assert fx_mul(to_fx(0.1), to_fx(10.0)) - ONE in range(-8, 9)


def test_no_floats_in_tick_path():
    """The tick arithmetic must stay on Python ints (bit-exact, portable)."""
    g = SpacetimeGraph()
    a, b = g.add_node(state=3), g.add_node(state=-4)
    g.add_edge(a, b, 1)
    state = g.tick(3)
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in state)
    assert all(isinstance(c, int) for p in (g.pos_of(a), g.pos_of(b)) for c in p)


# -- space-time world model ---------------------------------------------------
def test_spacetime_construction_and_validation():
    g = SpacetimeGraph("v")
    assert (g.n_nodes, g.n_edges, g.tick_count) == (0, 0, 0)
    a = g.add_node(state=5)
    b = g.add_node(state=-5, pos=(to_fx(3), to_fx(4), 0))
    edge = g.add_edge(a, b, 1, dt=2)
    assert (edge.src, edge.dst, edge.weight, edge.dt) == (a, b, 1, 2)
    assert g.distance(a, b) == to_fx(5)
    assert repr(g) == "SpacetimeGraph('v', nodes=2, edges=1, t=0)"

    assert _raises(ValueError, g.add_edge, a, b, 2)
    assert _raises(ValueError, g.add_edge, a, b, 1, -1)
    assert _raises(IndexError, g.add_edge, a, 99)
    assert _raises(IndexError, g.set_state, 99, 0)
    assert _raises(IndexError, g.pos_of, 99)
    assert g.set_state(a, 500) is None and g.state()[a] == INT8_MAX


def test_spacetime_tick_is_deterministic_and_order_independent():
    def build(order):
        g = SpacetimeGraph("d")
        for _ in range(3):
            g.add_node(state=1)
        for src, dst, w in order:
            g.add_edge(src, dst, w)
        return g

    edges = [(0, 1, 1), (1, 2, -1), (0, 2, 1)]
    g1, g2 = build(edges), build(list(reversed(edges)))
    g1.tick(25)
    g2.tick(25)
    assert g1.state() == g2.state()
    again = build(edges)
    again.tick(25)
    assert again.fingerprint() == g1.fingerprint()


def test_spacetime_saturation_then_exact_untick():
    g = SpacetimeGraph()
    driver = g.add_node(state=1)
    sink = g.add_node(state=0)
    g.add_edge(driver, sink, 1)
    g.tick()
    assert g.state() == (1, 1)
    g.tick(200)
    assert g.state() == (1, INT8_MAX)
    g.untick(201)
    assert g.state() == (1, 0) and g.tick_count == 0
    assert _raises(IndexError, g.untick, 1)
    assert _raises(ValueError, g.tick, -1)


def test_spacetime_lightcone_respects_delays():
    g = SpacetimeGraph()
    n = [g.add_node() for _ in range(4)]
    g.add_edge(n[0], n[1], dt=2)
    g.add_edge(n[1], n[2], dt=2)
    g.add_edge(n[0], n[2], dt=10)
    g.add_edge(n[2], n[3], dt=1)
    assert g.lightcone(n[0]) == {0: 0, 1: 2, 2: 4, 3: 5}
    assert g.lightcone(n[0], max_dt=3) == {0: 0, 1: 2}
    assert g.lightcone(n[2]) == {2: 0, 3: 1}
    assert g.lightcone(n[3]) == {3: 0}


def test_spacetime_edge_rewrite_and_prune():
    g = SpacetimeGraph()
    a, b = g.add_node(), g.add_node()
    g.add_edge(a, b, 1)
    assert g.find_edge(a, b).weight == 1
    assert g.set_edge_weight(a, b, -1) is True
    assert g.find_edge(a, b).weight == -1
    assert g.set_edge_weight(a, 99, 0) is False
    assert _raises(ValueError, g.set_edge_weight, a, b, 5)
    assert g.remove_edge(a, b) == 1 and g.remove_edge(a, b) == 0
    assert g.n_edges == 0


# -- p/q prompt parsing -------------------------------------------------------
def test_pq_parse_and_canonical_roundtrip():
    source = """
    # program the world: two ticks, then mutate
    step 2                 # advance time
    edge 0 1 +1
    set 3 -7 q 1/2
    wire a b residual q 3/4

    flip 0 1 q 2/3
    """
    p = parse_pq(source)
    assert len(p) == 5
    assert p[0].kind == "STEP" and p[0].args == (2,) and p[0].line == 3
    assert p[2].args == (3, -7) and (p[2].p, p[2].q) == (1, 2)
    assert p[3].args == ("a", "b", "residual")
    assert p[2].text() == "set 3 -7 q 1/2"
    assert p[0].text() == "step 2"
    assert parse_pq(p.text()) == p


def test_pq_parse_empty_and_comments_only():
    p = parse_pq("\n# nothing\n   \n")
    assert len(p) == 0 and p.text() == ""


def test_pq_parse_rejects_malformed_programs():
    cases = {
        "blorp 1 2": 1,
        "edge 0 1": 1,
        "edge 0 1 2": 1,
        "edge -1 1 +1": 1,
        "set 0 1 q 3/2": 1,
        "set 0 1 q 1/0": 1,
        "set 0 1 q": 1,
        "step -1": 1,
        "wire a b sideways": 1,
        "edge 0 1 +1\nset 9 x": 2,
    }
    for source, line in cases.items():
        err = None
        try:
            parse_pq(source)
        except PqError as exc:
            err = exc
        assert err is not None, source
        assert err.line == line, (source, err.line, line)
    assert _raises(TypeError, parse_pq, 42)


# -- p/q prompt execution -----------------------------------------------------
def test_pq_acceptance_ratios_are_deterministic():
    p = parse_pq("set 0 1 q 1/2\nset 1 2 q 0/1\nset 2 3 q 1/1")
    assert p[1].accepts(None) is False
    assert p[2].accepts(None) is True  # p >= q short-circuits, no draw
    import random as _random

    accepted = sum(p[0].accepts(_random.Random(s)) for s in range(200))
    assert 60 <= accepted <= 140


def test_pq_apply_is_reproducible_from_seed():
    source = "\n".join(f"set {i % 4} {i + 1} q 1/2" for i in range(40))
    g1, g2, g3 = _world(), _world(), _world()
    r1 = apply_pq(g1, source, seed=7)
    r2 = apply_pq(g2, source, seed=7)
    apply_pq(g3, source, seed=8)
    assert g1.fingerprint() == g2.fingerprint()
    assert r1.as_dict() == r2.as_dict()
    assert 0 < r1.applied < 40 and r1.skipped == 40 - r1.applied
    assert g3.state() != g1.state()


def test_pq_budget_caps_applied_ops():
    g = _world()
    report = apply_pq(g, "set 0 1\nset 1 2\nset 2 3", budget=2)
    assert (report.applied, report.skipped) == (2, 1)
    assert g.state()[:3] == (1, 2, 0) and g.state()[3] == 0
    assert "budget exhausted" in report.log[-1]
    assert _raises(ValueError, apply_pq, g, "step 1", 0, -1)


def test_pq_on_spacetime_ops_and_errors():
    g = _world()
    report = apply_pq(g, "edge 0 1 +1\nflip 0 1\nstep 3\nset 2 9\nprune 0 1")
    assert report.applied == 5
    assert g.n_edges == 0 and g.state()[2] == 9 and g.tick_count == 3
    assert _raises(PqError, apply_pq, g, "flip 0 1")
    assert _raises(PqError, apply_pq, g, "prune 0 1")
    assert _raises(PqError, apply_pq, g, "wire 0 1")


def test_pq_on_tree_programs_micro_edges_and_bytes():
    t = Tree(4, "t")
    report = apply_pq(t, "edge 0 1 +1\nset 2 5\nflip 0 1\nprune 0 1 q 1/1")
    assert report.applied == 4
    assert t.nodes[2] == 5
    assert len(t._esrc) == 0
    assert _raises(PqError, apply_pq, t, "step 1")
    assert _raises(PqError, apply_pq, t, "set 9 1")
    assert _raises(PqError, apply_pq, t, "wire a b")


def test_pq_on_ultragraph_wires_named_trees():
    ug = UltraGraph()
    ug.add(Tree(2, "a"))
    ug.add(Tree(2, "b"))
    report = apply_pq(ug, "wire a b residual")
    assert report.applied == 1
    assert len(ug.ultra_edges) == 1 and ug.ultra_edges[0].kind == "residual"
    assert _raises(PqError, apply_pq, ug, "wire a nope")
    assert _raises(PqError, apply_pq, ug, "edge 0 1 +1")


def test_pq_rejects_unknown_target():
    assert _raises(TypeError, apply_pq, object(), "step 1")
    assert _raises(TypeError, apply_pq, _world(), 42)


def test_evolve_runs_deterministic_rounds():
    g = _world(1)
    reports = evolve(g, "set 0 1 q 1/2", seed=3, rounds=4)
    again = _world(1)
    reports2 = evolve(again, "set 0 1 q 1/2", seed=3, rounds=4)
    assert len(reports) == 4 and reports == reports2
    assert g.state() == again.state()
    assert all(r.applied + r.skipped == 1 for r in reports)
    assert _raises(ValueError, evolve, g, "step 1", 0, -1)


def test_fixed_helpers_agree_with_float():
    for value in (0.0, 0.5, -0.25, 1.0, -2.75, 1 / 3, 123.456):
        assert math.isclose(to_float(to_fx(value)), value, rel_tol=0, abs_tol=1e-4)


# -- UGMZ world artifact (the compiled form the Rust runtime ticks) -----------

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from ultragraph.zerocpu import emit as emit_mod  # noqa: E402
from ultragraph.zerocpu import emit_ugm, from_bytes, load_ugm, to_bytes  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "rust" / "zerocpu" / "tests" / "fixtures"


def _fixture_world():
    g = SpacetimeGraph("fixture")
    states = [1, -1, 2, 0, 3]
    pos = [(0, 0, 0), (65536, 0, 0), (0, 65536, 0), (1, 2, 3), (-65536, 65536, -1)]
    for s, p in zip(states, pos):
        g.add_node(s, p)
    for src, dst, w, dt in [(0, 1, 1, 1), (1, 2, -1, 2), (2, 3, 1, 1),
                            (3, 4, 1, 0), (4, 0, -1, 3), (2, 2, 1, 1)]:
        g.add_edge(src, dst, w, dt)
    return g


def test_ugmz_roundtrip_preserves_world():
    g = _fixture_world()
    g.tick(2, remember=False)
    blob = to_bytes(g)
    back = from_bytes(blob)
    assert back.state() == g.state()
    assert [back.pos_of(i) for i in range(g.n_nodes)] == [g.pos_of(i) for i in range(g.n_nodes)]
    assert back.edges == g.edges
    assert back.tick_count == g.tick_count
    assert to_bytes(back) == blob


def test_ugmz_rejects_bad_input():
    blob = to_bytes(_fixture_world())
    assert _raises(ValueError, from_bytes, b"")
    assert _raises(ValueError, from_bytes, b"XXXX" + blob[4:])
    assert _raises(ValueError, from_bytes, blob[:4] + b"\x09\x00" + blob[6:])
    assert _raises(ValueError, from_bytes, blob[:-1])
    assert _raises(ValueError, from_bytes, blob + b"\x00")
    g = SpacetimeGraph("ragged")
    g.add_node(1, (0, 0))
    g.add_node(2, (0, 0, 0))
    assert _raises(ValueError, to_bytes, g)
    wide = SpacetimeGraph("wide")
    wide.add_node(1, (0, 0, 0, 0))
    assert _raises(ValueError, to_bytes, wide)


def test_ugmz_rejects_edges_to_missing_nodes():
    blob = bytearray(to_bytes(_fixture_world()))
    off = emit_mod.HEADER_SIZE + 5 * emit_mod.NODE_RECORD  # first edge record
    blob[off:off + 4] = (99).to_bytes(4, "little")
    assert _raises(ValueError, from_bytes, bytes(blob))


def test_ugmz_fixture_is_stable():
    """The committed artifact is what `to_bytes` produces today, byte for byte."""
    path = _FIXTURES / "world.ugmz"
    if not path.exists():  # installed package without the rust tree
        return
    g = _fixture_world()
    assert path.read_bytes() == to_bytes(g)
    g.tick(3)
    expected = (_FIXTURES / "world.expected.txt").read_text().splitlines()
    assert expected == [
        "ok magic=UGMZ version=1 nodes=5 edges=6 pos_dim=3 start_tick=0 ticks=3 tick=3",
        "state=" + emit_mod.state_hex(g),
        "pos=" + emit_mod.pos_hex(g),
    ]


def test_ugmz_ticked_world_matches_rust_expectation():
    """The Python side of the cross-language check: load, tick 3, same bytes."""
    path = _FIXTURES / "world.ugmz"
    if not path.exists():
        return
    g = load_ugm(path)
    assert g.tick_count == 0
    g.tick(3)
    expected = dict(
        line.split("=", 1)
        for line in (_FIXTURES / "world.expected.txt").read_text().splitlines()
        if "=" in line and line.startswith(("state=", "pos="))
    )
    assert emit_mod.state_hex(g) == expected["state"]
    assert emit_mod.pos_hex(g) == expected["pos"]


def test_ugmz_emit_ugm_writes_and_reloads():
    g = _fixture_world()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "world.ugmz"
        assert emit_ugm(g, out) == str(out)
        assert out.read_bytes() == to_bytes(g)
        assert load_ugm(out).state() == g.state()


# -- LOD: quasi-regular partitioning ------------------------------------------

from ultragraph.zerocpu import lod as lod_mod  # noqa: E402
from ultragraph.zerocpu.lod import PartitionError, coarsen, density_stats, partition  # noqa: E402


def _clusters():
    n = 12
    edges = [
        (u, v)
        for cl in (range(0, 4), range(4, 8), range(8, 12))
        for u in cl
        for v in cl
        if u != v
    ]
    return n, edges + [(0, 4), (4, 8), (8, 0)]


def test_lod_partition_is_balanced_and_recovers_a_cluster():
    n, edges = _clusters()
    p = partition(n, edges, 3)
    assert p.blocks == 3 and len(p.block_of) == n
    assert max(p.sizes) - min(p.sizes) <= 1
    first = p.block_of[0]
    assert all(p.block_of[v] == first for v in range(4)), p.block_of
    assert p.concentration > 0
    assert p.iterations == 1


def test_lod_is_deterministic_and_reports_both_measures():
    n, edges = _clusters()
    a, b = partition(n, edges, 3), partition(n, edges, 3)
    assert a == b
    seed = lod_mod._balanced_order(n, 3, lod_mod._seed_order(n, edges))
    seed_stats = density_stats(n, edges, seed)
    assert a.concentration >= seed_stats["concentration"]
    assert 0 <= a.defect <= (1 << 16)
    assert a.global_density == seed_stats["global_density"]


def test_lod_epsilon_stops_before_any_move():
    n, edges = _clusters()
    p = partition(n, edges, 3, epsilon=1 << 16)
    assert p.iterations == 0


def test_lod_single_block_is_trivial_and_coarsen_preserves_mass():
    n, edges = _clusters()
    p = partition(n, edges, 1)
    assert p.block_of == (0,) * n and p.sizes == (n,) and p.defect == 0
    assert coarsen(n, edges, p.block_of) == ((0, 0, len(edges)),)
    q = partition(n, edges, 3)
    sup = coarsen(n, edges, q.block_of)
    assert sum(w for _, _, w in sup) == len(edges)
    assert sup == tuple(sorted(sup)) and len({(a, b) for a, b, _ in sup}) == len(sup)


def test_lod_rejects_bad_input():
    assert _raises(PartitionError, partition, 0, [], 1)
    assert _raises(PartitionError, partition, 4, [], 0)
    assert _raises(PartitionError, partition, 4, [], 5)
    assert _raises(PartitionError, partition, 4, [(0, 9)], 2)
    assert _raises(PartitionError, partition, 4, [(0, 1)], 2, max_iters=-1)
    assert _raises(PartitionError, partition, 4, [(0, 1)], 2, epsilon=-1)
    assert _raises(PartitionError, density_stats, 4, [], (0, 1))
    assert _raises(PartitionError, coarsen, 4, [], (0, 1))
    assert _raises(PartitionError, coarsen, 4, [(0, 1)], (0, 1))


def test_lod_fixture_is_stable():
    """The committed LOD fixture is what Python produces today, byte for byte."""
    edges_file = _FIXTURES / "lod_edges.txt"
    if not edges_file.exists():
        return
    n, edges = _clusters()
    assert edges_file.read_text() == "\n".join(f"{u} {v}" for u, v in edges) + "\n"
    assert (_FIXTURES / "lod_params.txt").read_text() == "n=12\nk=3\nepsilon=0\nmax_iters=8\n"
    p = partition(n, edges, 3)
    want = (
        "block_of=" + ",".join(map(str, p.block_of)) + "\n"
        "sizes=" + ",".join(map(str, p.sizes)) + "\n"
        "densities=" + ",".join(map(str, p.densities)) + "\n"
        f"global={p.global_density}\ndefect={p.defect}\n"
        f"concentration={p.concentration}\niterations={p.iterations}\n"
        f"super_edges={len(coarsen(n, edges, p.block_of))}\n"
    )
    assert (_FIXTURES / "lod_expected.txt").read_text() == want


# -- Vaked bridge: lowered workflow -> world -> artifact ----------------------

from ultragraph.zerocpu import WorkflowError, lower_vaked, workflow_meta, workflow_to_graph  # noqa: E402


def _workflow():
    return {
        "on": "e:created",
        "maxDepth": 3,
        "depth": 3,
        "log": "var/lib/demo/eventd/log.jsonl",
        "steps": [
            {"name": "A", "agent": "field.a"},
            {"name": "B", "agent": "field.a"},
            {"name": "C", "agent": "field.a"},
            {"name": "D", "agent": "field.solo"},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "A", "to": "C"},
            {"from": "B", "to": "D"},
            {"from": "C", "to": "D", "weight": -1, "dt": 2},
        ],
    }


def test_workflow_maps_to_a_world():
    g = workflow_to_graph(_workflow())
    assert g.n_nodes == 4 and g.n_edges == 4
    assert [g.pos_of(i) for i in range(4)] == [(0, 0, 0), (ONE, 0, 0), (2 * ONE, 0, 0), (3 * ONE, 0, 0)]
    assert g.state()[0] == g.state()[1] == g.state()[2]
    assert g.state()[3] != g.state()[0]
    assert g.edges[-1].weight == -1 and g.edges[-1].dt == 2
    assert workflow_meta(_workflow())["maxDepth"] == 3


def test_workflow_explicit_state_wins_and_is_clipped():
    wf = {
        "steps": [{"name": "A", "state": 300}, {"name": "B", "state": -5}],
        "edges": [{"from": "A", "to": "B"}],
    }
    g = workflow_to_graph(wf)
    assert g.state() == (INT8_MAX, -5)


def test_workflow_rejects_malformed_graphs():
    cases = [
        {},
        {"steps": [], "edges": []},
        {"steps": [{"agent": "x"}], "edges": []},
        {"steps": [{"name": "A"}, {"name": "A"}], "edges": []},
        {"steps": [{"name": "A"}], "edges": [{"from": "A", "to": "Z"}]},
        {"steps": [{"name": "A"}], "edges": [{"from": "A", "to": "A", "weight": 2}]},
        {"steps": [{"name": "A"}], "edges": [{"from": "A", "to": "A", "dt": -1}]},
        {"steps": "nope", "edges": []},
        {"steps": [{"name": "A"}], "edges": "nope"},
    ]
    for wf in cases:
        assert _raises(WorkflowError, workflow_to_graph, wf), wf


def test_lower_vaked_without_a_compiler_explains_itself():
    import os

    saved = os.environ.pop("VAKED_ROOT", None)
    try:
        assert _raises(WorkflowError, lower_vaked, "whatever.vaked")
    finally:
        if saved is not None:
            os.environ["VAKED_ROOT"] = saved


def test_vaked_diamond_fixture_ticks_as_expected():
    """The `.vaked` -> `.ugmz` fixture: load, tick, match the recorded bytes."""
    path = _FIXTURES / "vaked_diamond.ugmz"
    if not path.exists():
        return
    g = load_ugm(path)
    assert (g.n_nodes, g.n_edges) == (4, 4)
    assert g.tick(2) == (14, 42, 42, 98)
    want = dict(
        line.split("=", 1)
        for line in (_FIXTURES / "vaked_diamond.expected.txt").read_text().splitlines()
        if line.startswith(("state=", "pos="))
    )
    assert emit_mod.state_hex(g) == want["state"]
    assert emit_mod.pos_hex(g) == want["pos"]


# -- Compiled p/q programs: .pqq ---------------------------------------------

from ultragraph.zerocpu import (  # noqa: E402
    PQQ_HEADER,
    Resolved,
    dump_pqq,
    load_pqq,
    resolve,
    run_pqq,
    write_bundle,
)

_PROG = """# vaked diamond: push time, then program the wiring
step 2
edge 1 3 +1
flip 0 1
set 2 -9 q 1/2
prune 3 0 q 1/4
step 1
"""


def _diamond():
    from ultragraph.zerocpu import load_ugm

    return load_ugm(_FIXTURES / "vaked_diamond.ugmz")


def test_resolve_settles_the_same_draws_as_apply_pq():
    direct, replayed = _diamond(), _diamond()
    apply_pq(direct, _PROG, seed=7)
    r = resolve(_PROG, seed=7)
    run_pqq(replayed, dump_pqq(r))
    assert (direct.state(), direct.tick_count, direct.n_edges) == (
        replayed.state(),
        replayed.tick_count,
        replayed.n_edges,
    )
    assert len(r.accepted) + len(r.skipped) == 6
    assert r.seed == 7
    assert all(op.p == 1 and op.q == 1 for op in r.accepted)


def test_resolve_is_reproducible_and_budget_is_respected():
    a, b = resolve(_PROG, seed=7), resolve(_PROG, seed=7)
    assert a == b
    capped = resolve(_PROG, seed=7, budget=1)
    assert len(capped.accepted) == 1
    assert any("budget" in why for _, why in capped.skipped)
    assert _raises(ValueError, resolve, _PROG, 0, -1)


def test_dump_and_load_roundtrip():
    r = resolve(_PROG, seed=7)
    text = dump_pqq(r, nodes=4, edges=4, tick=0)
    assert text.startswith(PQQ_HEADER + "\n")
    assert [op.text() for op in load_pqq(text)] == [op.text() for op in r.accepted]
    assert "# skip set 2 -9 q 1/2: rejected (1/2)" in text
    assert Resolved() == Resolved()
    assert load_pqq(dump_pqq(Resolved())) == load_pqq(PQQ_HEADER + "\n")


def test_load_pqq_rejects_junk():
    assert _raises(PqError, load_pqq, "")
    assert _raises(PqError, load_pqq, "step 1")
    assert _raises(PqError, load_pqq, "pqq 2\nstep 1")
    assert _raises(PqError, load_pqq, "pqq 1\nset 1 2 q 1/2")
    # the parser is syntax-only; target validity is an apply-time property
    assert [op.text() for op in load_pqq("pqq 1\nedge 0 99 1")] == ["edge 0 99 1"]
    # an uncompiled prompt has no header, so replaying it is a mistake, not a guess
    assert _raises(PqError, run_pqq, _diamond(), _PROG)
    assert _raises(PqError, run_pqq, _diamond(), "pqq 1\nedge 0 99 1")
    assert _raises(PqError, run_pqq, _diamond(), "pqq 1\nflip 3 0")
    assert _raises(PqError, run_pqq, _diamond(), "pqq 1\nwire a b")


def test_write_bundle_emits_a_replayable_pair(tmp_path=None):
    import tempfile
    from pathlib import Path

    from ultragraph.zerocpu import load_ugm

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "diamond"
        info = write_bundle(_diamond(), base, _PROG, seed=7)
        assert info["applied"] == 4 and info["skipped"] == 2
        ugmz, pqq = Path(info["ugmz"]), Path(info["pqq"])
        assert ugmz.exists() and pqq.exists()

        emitted = load_ugm(ugmz)
        replayed = _diamond()
        run_pqq(replayed, pqq.read_text())
        assert emitted.state() == replayed.state()
        assert emitted.tick_count == replayed.tick_count == 3


def test_replay_fixture_is_stable():
    """The committed .pqq and its expected state are what Python produces today."""
    pqq = _FIXTURES / "replay.pqq"
    if not pqq.exists():
        return
    expected = _FIXTURES / "replay.expected.txt"
    g = _diamond()
    apply_pq(g, _PROG, seed=7)
    assert pqq.read_text() == dump_pqq(
        resolve(_PROG, seed=7), nodes=4, edges=4, tick=0
    )
    assert expected.read_text() == (
        f"state={emit_mod.state_hex(g)}\n"
        f"pos={emit_mod.pos_hex(g)}\n"
        f"tick={g.tick_count}\nedges={g.n_edges}\n"
    )
    again = _diamond()
    run_pqq(again, pqq.read_text())
    assert emit_mod.state_hex(again) == expected.read_text().split("state=")[1][:8]


# -- Hope cascade: a text becomes integer heat -------------------------------

from ultragraph.zerocpu import (  # noqa: E402
    SIGNAL_STEP,
    Observer,
    Resonance,
    resonate,
    seed_signal,
    signal,
)

_HOPE_TEXT = (_FIXTURES / "hope_signal.txt").read_text()


def test_hope_signal_is_integer_and_deterministic():
    sig = signal("Hope is all around.", 8)
    assert sig == signal("Hope is all around.", 8)
    assert len(sig) == 8
    assert all(type(h) is int for h in sig.heat)
    assert all(t in (-1, 0, 1) for t in sig.trits)
    assert sig.total_heat() % SIGNAL_STEP == 0
    assert _raises(ValueError, signal, "x", 0)
    assert _raises(TypeError, signal, b"bytes", 4)


def test_hope_signal_carries_shape_not_just_heat():
    """Centering keeps long texts from drifting to all-+1."""
    sig = signal(_HOPE_TEXT, 4)
    assert 1 in sig.trits and -1 in sig.trits
    assert abs(sig.balance()) <= len(sig)
    # position is part of the signal: a permutation is a different projection
    assert signal(_HOPE_TEXT[::-1], 4).heat != sig.heat


def test_hope_seed_requires_matching_lanes():
    g = _world(4)
    sig = signal("abc", 4)
    seed_signal(g, sig)
    assert g.state() == tuple(sig.trits)
    assert _raises(ValueError, seed_signal, _world(4), signal("abc", 5))


def test_hope_resonate_ticks_and_collapses():
    g = _world(4)
    g.add_edge(0, 1, +1)
    g.add_edge(1, 2, +1)
    g.add_edge(2, 3, +1)
    r = resonate(g, _HOPE_TEXT, rounds=2)
    assert isinstance(r, Resonance)
    assert r.rounds == 2
    assert len(r.trace) == 3
    assert r.trace[0] == emit_mod.state_hex(g) or r.settled() == emit_mod.state_hex(g)
    assert r.settled() == r.trace[-1]
    assert r.collapse in (-1, 0, 1)
    assert r.collapse == Observer("radio").of_world(g)
    assert _raises(ValueError, resonate, g, "abc", rounds=-1)


def test_hope_observer_is_silent_on_a_flat_world():
    g = _world(3)
    assert Observer().of_world(g) == 0
    assert Observer().of_node(g, 0) == 0
    g.set_state(0, -5)
    assert Observer().of_node(g, 0) == -1
    assert Observer().of_world(g) == -1


def test_hope_fixture_is_stable():
    """The committed projection and trace are what Python produces today."""
    g = load_ugm(_FIXTURES / "vaked_diamond.ugmz")
    r = resonate(g, _HOPE_TEXT, rounds=2)
    heat_hex = b"".join(h.to_bytes(8, "little", signed=True) for h in r.signal.heat).hex()
    expected = "\n".join(
        [
            f"heat={heat_hex}",
            "trits=" + "".join(str(t) for t in r.signal.trits),
            "trace=" + ",".join(r.trace),
            f"collapse={r.collapse}",
            f"balance={r.signal.balance()}",
            f"total_heat={r.signal.total_heat()}",
        ]
    )
    assert (_FIXTURES / "hope.expected.txt").read_text().strip() == expected
