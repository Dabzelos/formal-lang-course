import itertools
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import csr_matrix

from project.task_2 import graph_to_nfa, regex_to_dfa


class AdjacencyMatrixFA:
    def __init__(self, fa: NondeterministicFiniteAutomaton = None):
        if fa is None:
            self.states = {}
            self.state_count = 0
            self.start_state_indices = set()
            self.final_state_indices = set()
            self.boolean_decomposition = {}
            return

        self.states = {state: index for (index, state) in enumerate(fa.states)}
        self.state_count = len(fa.states)
        self.start_state_indices = set(
            self.states[state] for state in fa.start_states
        )
        self.final_state_indices = set(
            self.states[state] for state in fa.final_states
        )
        self.boolean_decomposition = self.boolean_decomposition(fa)

    def boolean_decomposition(self, fa: NondeterministicFiniteAutomaton):
        decomposition = {}
        for first_state, trans in fa.to_dict().items():
            for symbol, next_states in trans.items():
                next_states = (
                    {next_states}
                    if not isinstance(next_states, set)
                    else next_states
                )
                if symbol not in decomposition:
                    decomposition[symbol] = csr_matrix(
                        (self.state_count, self.state_count), dtype=bool
                    )
                for next_state in next_states:
                    first_state_index = self.states[first_state]
                    next_state_index = self.states[next_state]
                    decomposition[symbol][
                        first_state_index, next_state_index
                    ] = True
        return decomposition

    def transitive_closure(self):
        closure = sp.csr_matrix(
            (self.state_count, self.state_count), dtype=bool
        )
        closure.setdiag(True)

        if not self.boolean_decomposition:
            return closure

        closure = closure + sum(self.boolean_decomposition.values())
        res = np.linalg.matrix_power(closure.toarray(), self.state_count)
        return sp.csr_matrix(res)

    def accepts(self, word: Iterable[Symbol]) -> bool:
        symbols = list(word)
        configs = [(symbols, state) for state in self.start_state_indices]

        while len(configs) > 0:
            tape, state = configs.pop()
            if len(tape) == 0 and state in self.final_state_indices:
                return True
            for next_state in self.states.values():
                if self.boolean_decomposition[tape[0]][state, next_state]:
                    configs.append((tape[1:], next_state))

        return False

    def is_empty(self) -> bool:
        transitive_closure = self.transitive_closure()
        for start_state_id in self.start_state_indices:
            for final_state_id in self.final_state_indices:
                if transitive_closure[start_state_id, final_state_id]:
                    return False
        return True


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersection = AdjacencyMatrixFA()
    intersection.state_count = automaton1.state_count * automaton2.state_count

    intersection.states = {
        (i1, i2): (
            automaton1.states[i1] * automaton2.state_count
            + automaton2.states[i2]
        )
        for i1, i2 in itertools.product(
            automaton1.states.keys(), automaton2.states.keys()
        )
    }
    intersection.start_state_indices = [
        (s1 * automaton2.state_count + s2)
        for s1, s2 in itertools.product(
            automaton1.start_state_indices, automaton2.start_state_indices
        )
    ]
    intersection.final_state_indices = [
        (f1 * automaton2.state_count + f2)
        for f1, f2 in itertools.product(
            automaton1.final_state_indices, automaton2.final_state_indices
        )
    ]

    intersection_symbols = (
        automaton1.boolean_decomposition.keys()
        & automaton2.boolean_decomposition.keys()
    )
    for symbol in intersection_symbols:
        intersection.boolean_decomposition[symbol] = sp.kron(
            automaton1.boolean_decomposition[symbol],
            automaton2.boolean_decomposition[symbol],
            format="csr",
        )

    return intersection


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    dfa_m = AdjacencyMatrixFA(regex_to_dfa(regex))
    nfa_m = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersection = intersect_automata(dfa_m, nfa_m)
    tc = intersection.transitive_closure()

    regex_init_start_states = [
        key
        for key in dfa_m.states
        if dfa_m.states[key] in dfa_m.start_state_indices
    ]
    reg_init_final_states = [
        key
        for key in dfa_m.states
        if dfa_m.states[key] in dfa_m.final_state_indices
    ]

    return {
        (start, final)
        for (start, final) in itertools.product(start_nodes, final_nodes)
        for (regex_start, regex_final) in itertools.product(
            regex_init_start_states, reg_init_final_states
        )
        if tc[
            intersection.states[(regex_start, start)],
            intersection.states[(regex_final, final)],
        ]
    }
