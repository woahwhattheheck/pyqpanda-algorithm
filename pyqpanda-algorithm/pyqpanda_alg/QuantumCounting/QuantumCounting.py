# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum counting by phase estimation of the Grover operator."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from pyqpanda3.core import CPUQVM, H, QCircuit, QProg

from ..Grover import amp_operator, mark_data_reflection
from ..plugin import QFT, apply_QGate


@dataclass(frozen=True)
class QuantumCountingResult:
    """Structured result returned by QuantumCounting."""

    estimated_count: float
    rounded_count: int
    phase: float
    phase_state: str
    phase_probability: float
    search_space_size: int


class QuantumCounting:
    """Estimate the number of marked states in a Grover search space.

    Quantum counting combines Grover amplitude amplification with quantum phase
    estimation. If the search space has size N and contains M marked states,
    the Grover eigenphase theta obeys M / N = sin(theta) ** 2.

    The implementation follows the QFT and phase-register convention already
    used by pyqpanda_alg.QAE so callers can compose it with this package.

    Args:
        qnumber: Number of search qubits. Search-space size is 2 ** qnumber.
        phase_bits: Number of phase-estimation qubits.
        flip_operator: Callable producing the Grover phase-oracle circuit.
        mark_data: Convenience alternative to flip_operator. A bit string or
            sequence of bit strings marked through Grover.mark_data_reflection.
        in_operator: Optional state-preparation callable. When omitted, a
            uniform Hadamard superposition is used.
        shots: Executions used to identify the dominant phase state.

    Exactly one of flip_operator and mark_data must be supplied.
    """

    def __init__(
        self,
        qnumber: int,
        phase_bits: int = 6,
        flip_operator: Optional[Callable[[Sequence[int]], QCircuit]] = None,
        mark_data: Optional[Union[str, Sequence[str]]] = None,
        in_operator: Optional[Callable[[Sequence[int]], QCircuit]] = None,
        shots: int = 2048,
    ) -> None:
        if not isinstance(qnumber, int) or qnumber <= 0:
            raise ValueError("qnumber must be a positive integer")
        if not isinstance(phase_bits, int) or phase_bits <= 0:
            raise ValueError("phase_bits must be a positive integer")
        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if (flip_operator is None) == (mark_data is None):
            raise ValueError("provide exactly one of flip_operator or mark_data")

        self.qnumber = qnumber
        self.phase_bits = phase_bits
        self.in_operator = in_operator
        self.shots = shots
        self.search_space_size = 1 << qnumber

        if mark_data is not None:
            states = [mark_data] if isinstance(mark_data, str) else list(mark_data)
            if not states:
                raise ValueError("mark_data must contain at least one state")
            if len(states) > self.search_space_size:
                raise ValueError("mark_data cannot exceed the search-space size")
            if len(set(states)) != len(states):
                raise ValueError("mark_data contains duplicate states")
            for state in states:
                if not isinstance(state, str):
                    raise TypeError("each marked state must be a bit string")
                if len(state) != qnumber or set(state) - {"0", "1"}:
                    raise ValueError(
                        "each marked state must be a binary string of length qnumber"
                    )

            def marked_state_oracle(qubits):
                return mark_data_reflection(qubits=qubits, mark_data=states)

            self.flip_operator = marked_state_oracle
        else:
            self.flip_operator = flip_operator

    def _initial_state(self, search_qubits: Sequence[int]) -> QCircuit:
        if self.in_operator is None:
            return apply_QGate(list(search_qubits), H)
        return self.in_operator(search_qubits)

    def _grover_operator(self, search_qubits: Sequence[int]) -> QCircuit:
        return amp_operator(
            in_operator=self.in_operator,
            flip_operator=self.flip_operator,
            q_input=list(search_qubits),
            q_flip=list(search_qubits),
            q_zero=list(search_qubits),
        )

    @staticmethod
    def count_from_phase(qnumber: int, phase: float) -> float:
        """Convert a normalized Grover eigenphase to a marked-state count."""
        if not isinstance(qnumber, int) or qnumber <= 0:
            raise ValueError("qnumber must be a positive integer")
        if not np.isfinite(phase) or phase < 0.0 or phase >= 1.0:
            raise ValueError("phase must be finite and in [0, 1)")
        return float((1 << qnumber) * np.sin(np.pi * phase) ** 2)

    def circuit(self):
        """Build the quantum-counting program and return its phase qubits."""
        qubits = QProg(self.qnumber + self.phase_bits).qubits()
        search_qubits = list(qubits[: self.qnumber])
        phase_qubits = list(qubits[self.qnumber :])

        program = QProg()
        program << self._initial_state(search_qubits)
        program << apply_QGate(phase_qubits, H)

        for index, control_qubit in enumerate(phase_qubits):
            for _ in range(1 << index):
                controlled_grover = self._grover_operator(search_qubits).control(
                    [control_qubit]
                )
                program << controlled_grover

        program << QFT(phase_qubits).dagger()
        return program, phase_qubits

    def run(self) -> QuantumCountingResult:
        """Execute quantum counting and return the dominant count estimate."""
        machine = CPUQVM()
        program, phase_qubits = self.circuit()
        machine.run(program, self.shots)
        probabilities = machine.result().get_prob_dict(phase_qubits)
        if not probabilities:
            raise RuntimeError("quantum counting returned no phase probabilities")

        phase_state = max(probabilities, key=probabilities.get)
        phase = int(phase_state, 2) / float(1 << self.phase_bits)
        estimate = self.count_from_phase(self.qnumber, phase)
        rounded = int(round(estimate))
        rounded = max(0, min(self.search_space_size, rounded))

        return QuantumCountingResult(
            estimated_count=estimate,
            rounded_count=rounded,
            phase=phase,
            phase_state=phase_state,
            phase_probability=float(probabilities[phase_state]),
            search_space_size=self.search_space_size,
        )
