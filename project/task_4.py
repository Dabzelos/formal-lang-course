import numpy as np
from networkx import MultiDiGraph
from scipy import sparse

from project.task_2 import graph_to_nfa, regex_to_dfa
from project.task_3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    dfa_m = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa_m = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    nfa_start_states_list = list(nfa_m.start_state_indices)
    nfa_start_states_count = len(nfa_start_states_list)

    def init_front() -> sparse.csr_matrix:
        dfa_start = next(iter(dfa_m.start_state_indices))
        rows = [
            dfa_start + dfa_m.state_count * i
            for i in range(nfa_start_states_count)
        ]
        cols = nfa_start_states_list
        data = np.ones(nfa_start_states_count, dtype=bool)
        return sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(
                dfa_m.state_count * nfa_start_states_count,
                nfa_m.state_count,
            ),
            dtype=bool,
        )

    def update_front(front: sparse.csr_matrix) -> sparse.csr_matrix:
        dfa_tr = {
            lbl: m.transpose()
            for lbl, m in dfa_m.boolean_decomposition.items()
        }
        labels = (
            dfa_m.boolean_decomposition.keys()
            & nfa_m.boolean_decomposition.keys()
        )

        front_new = sparse.csr_matrix(
            (dfa_m.state_count * nfa_start_states_count, nfa_m.state_count),
            dtype=bool,
        )

        for label in labels:
            tmp = front @ nfa_m.boolean_decomposition[label]
            for i in range(nfa_start_states_count):
                r0 = i * dfa_m.state_count
                r1 = r0 + dfa_m.state_count
                tmp[r0:r1] = dfa_tr[label] @ tmp[r0:r1]

            front_new += tmp

        return front_new

    visited = sparse.csr_matrix(
        (dfa_m.state_count * nfa_start_states_count, nfa_m.state_count),
        dtype=bool,
    )
    front = init_front()

    while front.count_nonzero() > 0:
        visited += front
        front = update_front(front)
        front = front > visited

    pairs: set[tuple[int, int]] = set()

    dfa_final_states = list(dfa_m.final_state_indices)
    nfa_idx_to_state = {idx: st for st, idx in nfa_m.states.items()}
    nfa_final_mask = np.array(
        [i in nfa_m.final_state_indices for i in range(nfa_m.state_count)],
        dtype=bool,
    )

    for i, nfa_start_idx in enumerate(nfa_start_states_list):
        base = i * dfa_m.state_count
        for dfa_fin_idx in dfa_final_states:
            row = visited[base + dfa_fin_idx]
            if row.nnz == 0:
                continue
            reached = np.zeros(nfa_m.state_count, dtype=bool)
            reached[row.indices] = True
            reached &= nfa_final_mask

            for nfa_fin_idx in np.nonzero(reached)[0]:
                pairs.add(
                    (
                        nfa_idx_to_state[nfa_start_idx],
                        nfa_idx_to_state[nfa_fin_idx],
                    )
                )

    return pairs
