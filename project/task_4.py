import numpy as np
from networkx import MultiDiGraph
from scipy import sparse

from project.task_2 import graph_to_nfa, regex_to_dfa
from project.task_3 import AdjacencyMatrixFA


def init_front(
    dfa_initial: AdjacencyMatrixFA, nfa_initial: AdjacencyMatrixFA
) -> sparse.csr_matrix:
    dfa_start = list(dfa_initial.start_states_indices)[0]
    nfa_start_states_count = len(nfa_initial.start_states_indices)
    rows = [
        dfa_start + dfa_initial.states_count * i
        for i in range(nfa_start_states_count)
    ]
    cols = [
        start_state_ind for start_state_ind in nfa_initial.start_states_indices
    ]
    data = np.ones(nfa_start_states_count, dtype=bool)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(
            dfa_initial.states_count * nfa_start_states_count,
            nfa_initial.states_count,
        ),
        dtype=bool,
    )


def update_front(
    front: sparse.csr_matrix,
    dfa_initial: AdjacencyMatrixFA,
    nfa_initial: AdjacencyMatrixFA,
):
    fronts_decomposed = {}
    nfa_start_states_count = len(nfa_initial.start_states_indices)
    dfa_matrices_tr = {
        key: m.transpose()
        for key, m in dfa_initial.boolean_decomposition.items()
    }
    labels = (
        dfa_initial.boolean_decomposition.keys()
        & nfa_initial.boolean_decomposition.keys()
    )
    for label in labels:
        fronts_decomposed[label] = (
            front @ nfa_initial.boolean_decomposition[label]
        )
        for ind in range(nfa_start_states_count):
            fronts_decomposed[label][
                ind * dfa_initial.states_count : (ind + 1)
                * dfa_initial.states_count
            ] = (
                dfa_matrices_tr[label]
                @ fronts_decomposed[label][
                    ind * dfa_initial.states_count : (ind + 1)
                    * dfa_initial.states_count
                ]
            )

    front_new = sparse.csr_matrix(
        (
            dfa_initial.states_count * nfa_start_states_count,
            nfa_initial.states_count,
        ),
        dtype=bool,
    )
    for front_matrix in fronts_decomposed.values():
        front_new += front_matrix

    return front_new


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    dfa_initial = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa_initial = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )
    nfa_start_states_count = len(nfa_initial.start_states_indices)

    visited = sparse.csr_matrix(
        (
            dfa_initial.states_count * nfa_start_states_count,
            nfa_initial.states_count,
        ),
        dtype=bool,
    )
    front = init_front(dfa_initial, nfa_initial)
    while front.count_nonzero() > 0:
        visited += front
        front = update_front(front, dfa_initial, nfa_initial)
        front = front - visited

    dfa_final_states_index = dfa_initial.final_states_indices
    nfa_idx_to_st = {
        index: state for state, index in nfa_initial.states.items()
    }
    nfa_final_states = np.array(
        [
            i in nfa_initial.final_states_indices
            for i in range(nfa_initial.states_count)
        ],
        dtype=bool,
    )
    pairs = set()

    for i, nfa_start_state_id in enumerate(
        nfa_initial.start_states_indices, 0
    ):
        for dfa_final_state_id in dfa_final_states_index:
            row = visited[i * dfa_initial.states_count + dfa_final_state_id]
            row_vector = np.array(
                [i in row.indices for i in range(nfa_initial.states_count)],
                dtype=bool,
            )
            vector = row_vector & nfa_final_states
            nfa_final_states_reached = np.nonzero(vector)[0]
            for reached_nfa_final_state_ind in nfa_final_states_reached:
                pairs.add(
                    (
                        nfa_idx_to_st[nfa_start_state_id],
                        nfa_idx_to_st[reached_nfa_final_state_ind],
                    )
                )

    return pairs
