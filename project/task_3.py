import itertools
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)

from project.task2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, fa: NondeterministicFiniteAutomaton = None):
        if fa is None:
            self.states = {}
            self.state_count = 0
            self.start_state_indices = set()
            self.final_state_indices = set()
            self.boolean_decomposition = {}
            return

        self.states = {state: idx for idx, state in enumerate(fa.states)}
        self.state_count = len(fa.states)
        start_states = {self.states[state] for state in fa.start_states}
        self.start_state_indices = start_states
        final_states = {self.states[state] for state in fa.final_states}
        self.final_state_indices = final_states
        self.boolean_decomposition = self._build_boolean_decomposition(fa)

    def _build_boolean_decomposition(
        self, fa: NondeterministicFiniteAutomaton
    ):
        decomposition = {}

        for source_state, transitions in fa.to_dict().items():
            source_idx = self.states[source_state]

            for symbol, target_states in transitions.items():
                if not isinstance(target_states, set):
                    target_states = {target_states}

                if symbol not in decomposition:
                    size = (self.state_count, self.state_count)
                    decomposition[symbol] = sp.csr_matrix(size, dtype=bool)

                for target_state in target_states:
                    target_idx = self.states[target_state]
                    decomposition[symbol][source_idx, target_idx] = True

        return decomposition

    def transitive_closure(self):
        size = (self.state_count, self.state_count)
        closure = sp.csr_matrix(size, dtype=bool)
        closure.setdiag(True)

        if not self.boolean_decomposition:
            return closure

        all_transitions = None
        for symbol_matrix in self.boolean_decomposition.values():
            if all_transitions is None:
                all_transitions = symbol_matrix.copy()
            else:
                all_transitions += symbol_matrix

        closure += all_transitions

        dense_closure = closure.toarray()
        result_matrix = np.linalg.matrix_power(dense_closure, self.state_count)

        return sp.csr_matrix(result_matrix)

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        for source in self.start_state_indices:
            for target in self.final_state_indices:
                if transitive_closure[source, target]:
                    return False
        return True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        if not self.boolean_decomposition:
            return False

        current_states = sp.csr_matrix((1, self.state_count), dtype=bool)
        for start_idx in self.start_state_indices:
            current_states[0, start_idx] = True

        for symbol in word:
            if symbol not in self.boolean_decomposition:
                return False

            transition_matrix = self.boolean_decomposition[symbol]
            next_states = current_states.dot(transition_matrix)

            if next_states.nnz == 0:
                return False

            current_states = next_states

        for final_idx in self.final_state_indices:
            if current_states[0, final_idx]:
                return True

        return False


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    result = AdjacencyMatrixFA()

    state_mapping = {}
    result.state_count = automaton1.state_count * automaton2.state_count

    idx = 0
    for i in range(automaton1.state_count):
        for j in range(automaton2.state_count):
            state_mapping[(i, j)] = idx
            idx += 1

    symbols1 = set(automaton1.boolean_decomposition.keys())
    symbols2 = set(automaton2.boolean_decomposition.keys())
    common_symbols = symbols1 & symbols2

    result.boolean_decomposition = {}

    for symbol in common_symbols:
        matrix1 = automaton1.boolean_decomposition[symbol]
        matrix2 = automaton2.boolean_decomposition[symbol]
        product_matrix = sp.kron(matrix1, matrix2, format="csr")
        result.boolean_decomposition[symbol] = product_matrix

    result.start_state_indices = set()
    for start1 in automaton1.start_state_indices:
        for start2 in automaton2.start_state_indices:
            state_idx = state_mapping[(start1, start2)]
            result.start_state_indices.add(state_idx)

    result.final_state_indices = set()
    for final1 in automaton1.final_state_indices:
        for final2 in automaton2.final_state_indices:
            state_idx = state_mapping[(final1, final2)]
            result.final_state_indices.add(state_idx)

    states_dict = {idx: f"({i},{j})" for (i, j), idx in state_mapping.items()}
    result.states = states_dict

    return result


def _get_state_indices(automaton: AdjacencyMatrixFA, state_set: set) -> list:
    return [
        state for state, idx in automaton.states.items() if idx in state_set
    ]


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    dfa = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersection = intersect_automata(dfa, nfa)
    tc = intersection.transitive_closure()

    regex_start_states = _get_state_indices(dfa, dfa.start_state_indices)
    regex_final_states = _get_state_indices(dfa, dfa.final_state_indices)

    return {
        (start, final)
        for (start, final) in itertools.product(start_nodes, final_nodes)
        for (regex_start, regex_final) in itertools.product(
            regex_start_states, regex_final_states
        )
        if tc[
            intersection.states[(regex_start, start)],
            intersection.states[(regex_final, final)],
        ]
    }
