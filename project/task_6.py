from __future__ import annotations

from collections import defaultdict, deque

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Production, Variable


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    nf_cfg = cfg.to_normal_form()
    productions = set(nf_cfg.productions)

    for var in cfg.get_nullable_symbols():
        v = Variable(var.value)
        productions.add(Production(v, []))
        productions.add(Production(v, [Epsilon()]))

    return CFG(
        start_symbol=cfg.start_symbol, productions=productions
    ).remove_useless_symbols()


def _extract_productions(cfg: CFG):
    unary_by_label = defaultdict(set)
    binary_by_pair = defaultdict(set)
    epsilon_vars: set[Variable] = set()

    for p in cfg.productions:
        k = len(p.body)
        if k == 0:
            epsilon_vars.add(p.head)
        elif k == 1:
            sym = p.body[0]
            if hasattr(sym, "value"):
                unary_by_label[str(sym.value)].add(p.head)
        elif k == 2:
            left, right = p.body
            if isinstance(left, Variable) and isinstance(right, Variable):
                binary_by_pair[(left, right)].add(p.head)

    return unary_by_label, binary_by_pair, epsilon_vars


def _push(
    triple: tuple[Variable, int, int],
    r: set[tuple[Variable, int, int]],
    new: deque[tuple[Variable, int, int]],
    left_index: dict[int, set[tuple[Variable, int, int]]],
    right_index: dict[int, set[tuple[Variable, int, int]]],
) -> None:
    if triple in r:
        return
    r.add(triple)
    A, u, v = triple
    left_index[v].add(triple)
    right_index[u].add(triple)
    new.append(triple)


def _process_initial_triples(
    graph: nx.DiGraph,
    unary_by_label,
    epsilon_vars,
    r,
    new,
    left_index,
    right_index,
) -> None:
    for u, v, label in graph.edges(data="label"):
        for A in unary_by_label.get(str(label), ()):
            _push((A, u, v), r, new, left_index, right_index)

    if epsilon_vars:
        for x in graph.nodes:
            for A in epsilon_vars:
                _push((A, x, x), r, new, left_index, right_index)


def _process_queue(
    new: deque[tuple[Variable, int, int]],
    r: set[tuple[Variable, int, int]],
    left_index,
    right_index,
    binary_by_pair,
) -> None:
    while new:
        N, n, m = new.popleft()

        for M, n_prime, _m_prime in list(left_index.get(n, ())):
            for A in binary_by_pair.get((M, N), ()):
                _push((A, n_prime, m), r, new, left_index, right_index)

        for M, _n_prime, m_prime in list(right_index.get(m, ())):
            for A in binary_by_pair.get((N, M), ()):
                _push((A, n, m_prime), r, new, left_index, right_index)


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcfg = cfg_to_weak_normal_form(cfg)
    unary_by_label, binary_by_pair, epsilon_vars = _extract_productions(wcfg)

    r: set[tuple[Variable, int, int]] = set()
    new: deque[tuple[Variable, int, int]] = deque()
    left_index = defaultdict(set)
    right_index = defaultdict(set)

    _process_initial_triples(
        graph, unary_by_label, epsilon_vars, r, new, left_index, right_index
    )
    _process_queue(new, r, left_index, right_index, binary_by_pair)

    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    S = cfg.start_symbol
    return {
        (u, v)
        for (A, u, v) in r
        if A == S and u in start_nodes and v in final_nodes
    }
