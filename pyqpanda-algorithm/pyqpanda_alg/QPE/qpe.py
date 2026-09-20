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

"""Quantum Phase Estimation for PyQPanda3 circuits."""

from dataclasses import dataclass
from math import pi
from typing import Callable, Dict, Optional

from pyqpanda3.core import CPUQVM, H, QCircuit, QProg

from ..plugin import QFT


@dataclass(frozen=True)
class PhaseEstimationResult:
    """Quantum phase estimation result."""

    phase: float
    bitstring: str
    probability: float
    resolution: float
    probabilities: Dict[str, float]


def decode_phase(probabilities: Dict[str, float], precision_qubits: int) -> PhaseEstimationResult:
    """Decode the most likely phase from a counting-register distribution."""
    if precision_qubits < 1:
        raise ValueError("precision_qubits must be at least 1")
    if not probabilities:
        raise ValueError("probabilities cannot be empty")

    bitstring = max(probabilities, key=probabilities.get)
    if len(bitstring) != precision_qubits or any(bit not in "01" for bit in bitstring):
        raise ValueError(
            f"expected {precision_qubits}-bit binary keys, got {bitstring!r}"
        )

    denominator = 1 << precision_qubits
    return PhaseEstimationResult(
        phase=int(bitstring, 2) / denominator,
        bitstring=bitstring,
        probability=float(probabilities[bitstring]),
        resolution=1.0 / denominator,
        probabilities=dict(probabilities),
    )


def phase_to_eigenvalue(
    phase: float,
    evolution_time: float = 1.0,
    centered: bool = False,
) -> float:
    """Convert a phase to an eigenvalue for U = exp(i * H * t)."""
    if not 0.0 <= phase < 1.0:
        raise ValueError("phase must lie in [0, 1)")
    if evolution_time <= 0:
        raise ValueError("evolution_time must be positive")

    wrapped = phase - 1.0 if centered and phase >= 0.5 else phase
    return 2.0 * pi * wrapped / evolution_time


class QPE:
    """Standard Quantum Phase Estimation.

    The unitary callable receives the system-qubit list and returns a QCircuit
    implementing U. eigenstate_preparation may prepare an eigenstate before
    phase kickback.

    For algorithms that can construct powers of U more efficiently than
    repeating the base circuit, unitary_power(qubits, power) may be supplied
    and must return a circuit implementing U raised to power.
    """

    def __init__(
        self,
        unitary: Callable[[list[int]], QCircuit],
        system_qubits: int,
        precision_qubits: int = 5,
        eigenstate_preparation: Optional[Callable[[list[int]], QCircuit]] = None,
        unitary_power: Optional[Callable[[list[int], int], QCircuit]] = None,
        shots: int = 2048,
        machine=None,
    ) -> None:
        if not callable(unitary):
            raise TypeError("unitary must be callable")
        if eigenstate_preparation is not None and not callable(eigenstate_preparation):
            raise TypeError("eigenstate_preparation must be callable")
        if unitary_power is not None and not callable(unitary_power):
            raise TypeError("unitary_power must be callable")
        if system_qubits < 1:
            raise ValueError("system_qubits must be at least 1")
        if precision_qubits < 1:
            raise ValueError("precision_qubits must be at least 1")
        if shots < 1:
            raise ValueError("shots must be at least 1")

        self.unitary = unitary
        self.system_qubits = system_qubits
        self.precision_qubits = precision_qubits
        self.eigenstate_preparation = eigenstate_preparation
        self.unitary_power = unitary_power
        self.shots = shots
        self.machine = machine if machine is not None else CPUQVM()

    def _append_controlled_power(
        self,
        program: QProg,
        system_register: list[int],
        control_qubit: int,
        power: int,
    ) -> None:
        if self.unitary_power is not None:
            powered = self.unitary_power(system_register, power)
            if not isinstance(powered, QCircuit):
                raise TypeError("unitary_power must return QCircuit")
            program << powered.control([control_qubit])
            return

        for _ in range(power):
            base = self.unitary(system_register)
            if not isinstance(base, QCircuit):
                raise TypeError("unitary must return QCircuit")
            program << base.control([control_qubit])

    def build_program(self) -> tuple[QProg, list[int]]:
        """Build QPE and return the program plus its counting register."""
        qubits = QProg(self.system_qubits + self.precision_qubits).qubits()
        system_register = qubits[: self.system_qubits]
        counting_register = qubits[self.system_qubits :]

        program = QProg()
        if self.eigenstate_preparation is not None:
            preparation = self.eigenstate_preparation(system_register)
            if not isinstance(preparation, QCircuit):
                raise TypeError("eigenstate_preparation must return QCircuit")
            program << preparation

        for qubit in counting_register:
            program << H(qubit)

        # Match the register-power convention already used by QAE: counting
        # qubit i controls U^(2^i), followed by inverse QFT.
        for index, control in enumerate(counting_register):
            self._append_controlled_power(
                program,
                system_register,
                control,
                1 << index,
            )

        program << QFT(counting_register).dagger()
        return program, counting_register

    def run(self) -> PhaseEstimationResult:
        """Execute QPE and return the dominant phase and full distribution."""
        program, counting_register = self.build_program()
        self.machine.run(program, self.shots)
        probabilities = self.machine.result().get_prob_dict(counting_register)
        return decode_phase(probabilities, self.precision_qubits)
