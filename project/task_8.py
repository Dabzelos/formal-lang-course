import networkx as nx
import scipy.sparse as sp
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from pyformlang.rsa import Box, RecursiveAutomaton

from project.task_2 import graph_to_nfa
from project.task_3 import AdjacencyMatrixFA, intersect_automata


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def _build_rsm_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    def _val(x):
        return x.value if hasattr(x, "value") else x

    for sym in rsm.boxes:
        box: Box = rsm.get_box(sym)

        for s, t, lbl in box.dfa.to_networkx().edges(data="label"):
            nfa.add_transition(
                State((sym, _val(s))), lbl, State((sym, _val(t)))
            )
        for st in box.start_state:
            nfa.add_start_state(State((sym, _val(st))))
        for st in box.final_states:
            nfa.add_final_state(State((sym, _val(st))))

    return nfa


def _product_start_indices(
    inter: AdjacencyMatrixFA, rsm_adj: AdjacencyMatrixFA
) -> set[int]:
    rsm_start_objs = {
        st
        for st, i in rsm_adj.states.items()
        if i in rsm_adj.start_state_indices
    }
    return {
        idx
        for (g_st, r_st), idx in inter.states.items()
        if r_st in rsm_start_objs
    }


def _ms_bfs_on_product(
    inter: AdjacencyMatrixFA, rsm_adj: AdjacencyMatrixFA
) -> sp.csr_matrix:
    n = inter.state_count
    reach = sp.csr_matrix((n, n), dtype=bool)

    for i in _product_start_indices(inter, rsm_adj):
        reach[i, i] = True

    mats = list(inter.boolean_decomposition.values())
    changed = True
    while changed:
        changed = False
        before = reach.count_nonzero()
        for m in mats:
            reach = reach.maximum(reach @ m)
        changed = reach.count_nonzero() > before

    return reach


def _same_box_if_start_to_final(
    rsm_adj: AdjacencyMatrixFA, r_s: State, r_f: State
) -> object | None:
    rsm_start_objs = {
        st
        for st, i in rsm_adj.states.items()
        if i in rsm_adj.start_state_indices
    }
    rsm_final_objs = {
        st
        for st, i in rsm_adj.states.items()
        if i in rsm_adj.final_state_indices
    }
    if r_s not in rsm_start_objs or r_f not in rsm_final_objs:
        return None
    try:
        box_s, _ = r_s.value
        box_f, _ = r_f.value
    except Exception:
        return None
    return box_s if box_s == box_f else None


def _augment_graph_with_nonterminals(
    reach: sp.csr_matrix,
    g_adj: AdjacencyMatrixFA,
    rsm_adj: AdjacencyMatrixFA,
    inter: AdjacencyMatrixFA,
) -> bool:
    inter_idx_to_state: dict[int, tuple[State, State]] = {
        idx: pair for pair, idx in inter.states.items()
    }
    changed = False

    rows, cols = reach.nonzero()
    for i, j in zip(rows, cols):
        (g_s, r_s) = inter_idx_to_state[i]
        (g_f, r_f) = inter_idx_to_state[j]

        box_label = _same_box_if_start_to_final(rsm_adj, r_s, r_f)
        if box_label is None:
            continue

        if box_label not in g_adj.boolean_decomposition:
            g_adj.boolean_decomposition[box_label] = sp.csr_matrix(
                (g_adj.state_count, g_adj.state_count), dtype=bool
            )

        g_s_idx = g_adj.states[g_s]
        g_f_idx = g_adj.states[g_f]
        mat = g_adj.boolean_decomposition[box_label]

        if not mat[g_s_idx, g_f_idx]:
            mat[g_s_idx, g_f_idx] = True
            g_adj.boolean_decomposition[box_label] = mat
            changed = True

    return changed


def _fixpoint_tensor_pass(
    g_adj: AdjacencyMatrixFA, rsm_adj: AdjacencyMatrixFA
) -> bool:
    inter = intersect_automata(g_adj, rsm_adj)
    reach = _ms_bfs_on_product(inter, rsm_adj)
    return _augment_graph_with_nonterminals(reach, g_adj, rsm_adj, inter)


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    s_nodes = set(graph.nodes) if start_nodes is None else set(start_nodes)
    f_nodes = set(graph.nodes) if final_nodes is None else set(final_nodes)

    fa_graph = graph_to_nfa(graph, s_nodes, f_nodes)
    g_adj = AdjacencyMatrixFA(fa_graph)

    fa_rsm = _build_rsm_nfa(rsm)
    rsm_adj = AdjacencyMatrixFA(fa_rsm)

    updated = True
    while updated:
        updated = _fixpoint_tensor_pass(g_adj, rsm_adj)

    result: set[tuple[int, int]] = set()
    init_label = rsm.initial_label
    if init_label in g_adj.boolean_decomposition:
        mat = g_adj.boolean_decomposition[init_label]
        rows, cols = mat.nonzero()

        idx_to_state: dict[int, object] = {
            idx: st for st, idx in g_adj.states.items()
        }

        def _val(x):
            return x.value if hasattr(x, "value") else x

        for i, j in zip(rows, cols):
            u, v = _val(idx_to_state[i]), _val(idx_to_state[j])
            if u in s_nodes and v in f_nodes:
                result.add((u, v))

    return result
