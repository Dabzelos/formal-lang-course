from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Variable
from scipy.sparse import csr_matrix

from project.task_6 import cfg_to_weak_normal_form


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


def _index_nodes(graph: nx.DiGraph):
    nodes: list[int] = list(graph.nodes)
    node_to_idx: dict[int, int] = {node: i for i, node in enumerate(nodes)}
    idx_to_node: dict[int, int] = {i: node for i, node in enumerate(nodes)}
    return nodes, node_to_idx, idx_to_node, len(nodes)


def _init_decomposition(
    variables: set[Variable], n: int
) -> dict[Variable, csr_matrix]:
    return {A: csr_matrix((n, n), dtype=bool) for A in variables}


def _seed_unary_and_epsilon(
    graph: nx.DiGraph,
    unary_by_label,
    epsilon_vars: set[Variable],
    node_to_idx: dict[int, int],
    decomp: dict[Variable, csr_matrix],
) -> None:
    for u, v, label in graph.edges(data="label"):
        for A in unary_by_label.get(str(label), ()):
            decomp[A][node_to_idx[u], node_to_idx[v]] = True
    for A in epsilon_vars:
        decomp[A].setdiag(True)


def _fixed_point(binary_by_pair, decomp: dict[Variable, csr_matrix]) -> None:
    changed = True
    while changed:
        changed = False
        for (B, C), heads in binary_by_pair.items():
            prod = decomp[B] @ decomp[C]
            if prod.nnz == 0:
                continue
            for A in heads:
                new_head = decomp[A].maximum(prod)
                if (new_head != decomp[A]).nnz != 0:
                    decomp[A] = new_head
                    changed = True


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    wcfg = cfg_to_weak_normal_form(cfg)
    unary_by_label, binary_by_pair, epsilon_vars = _extract_productions(wcfg)
    nodes, node_to_idx, idx_to_node, n = _index_nodes(graph)
    decomp = _init_decomposition(wcfg.variables, n)
    _seed_unary_and_epsilon(
        graph, unary_by_label, epsilon_vars, node_to_idx, decomp
    )
    _fixed_point(binary_by_pair, decomp)

    if start_nodes is None:
        start_nodes = set(nodes)
    if final_nodes is None:
        final_nodes = set(nodes)

    S = wcfg.start_symbol
    rows, cols = decomp[S].nonzero()
    result: set[tuple[int, int]] = set()
    for i, j in zip(rows, cols):
        u, v = idx_to_node[i], idx_to_node[j]
        if u in start_nodes and v in final_nodes:
            result.add((u, v))
    return result
