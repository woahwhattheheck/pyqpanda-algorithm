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

"""Variational Quantum Linear Solver.

The solver owns the backend-independent VQLS mathematics. A caller may provide
any ansatz function mapping a real parameter vector to a statevector, including
a PyQPanda3 simulator or hardware adapter. A compact RY plus CNOT-ring reference
ansatz is included for examples and local experimentation.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


StateFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class VQLSResult:
    """Result returned by VQLS.solve."""

    parameters: np.ndarray
    state: np.ndarray
    solution: np.ndarray
    cost: float
    residual_norm: float
    iterations: int
    converged: bool
    cost_history: Tuple[float, ...]


def _normalize_state(state: Sequence[complex], dimension: int) -> np.ndarray:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size != dimension:
        raise ValueError(
            "ansatz state has dimension {0}, expected {1}".format(
                vector.size, dimension
            )
        )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("ansatz returned a zero or non-finite state")
    return vector / norm


def _apply_ry(
    state: np.ndarray, angle: float, qubit: int, num_qubits: int
) -> None:
    bit = 1 << qubit
    cosine = float(np.cos(angle / 2.0))
    sine = float(np.sin(angle / 2.0))
    for index in range(1 << num_qubits):
        if index & bit:
            continue
        paired = index | bit
        low = state[index]
        high = state[paired]
        state[index] = cosine * low - sine * high
        state[paired] = sine * low + cosine * high


def _apply_cnot(
    state: np.ndarray, control: int, target: int, num_qubits: int
) -> None:
    control_bit = 1 << control
    target_bit = 1 << target
    for index in range(1 << num_qubits):
        if not (index & control_bit) or (index & target_bit):
            continue
        paired = index | target_bit
        state[index], state[paired] = state[paired], state[index]


def ry_ring_state(
    parameters: Sequence[float], num_qubits: int, layers: int = 2
) -> np.ndarray:
    """Return a hardware-efficient real-amplitude RY/CNOT-ring statevector.

    Qubit zero is the least-significant statevector bit. Every layer applies
    one RY rotation per qubit followed by nearest-neighbour CNOTs. Systems with
    three or more qubits close the entangling chain into a ring.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    if layers < 1:
        raise ValueError("layers must be at least 1")

    params = np.asarray(parameters, dtype=float).reshape(-1)
    expected = num_qubits * layers
    if params.size != expected:
        raise ValueError(
            "RY-ring ansatz needs {0} parameters, received {1}".format(
                expected, params.size
            )
        )

    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1.0
    offset = 0
    for _ in range(layers):
        for qubit in range(num_qubits):
            _apply_ry(state, params[offset], qubit, num_qubits)
            offset += 1
        for qubit in range(num_qubits - 1):
            _apply_cnot(state, qubit, qubit + 1, num_qubits)
        if num_qubits > 2:
            _apply_cnot(state, num_qubits - 1, 0, num_qubits)
    return state


class VQLS:
    """Variational Quantum Linear Solver.

    The global objective is

        C(theta) = 1 - |<b|A|x(theta)>|^2 / <x(theta)|A^H A|x(theta)>.

    It reaches zero when the variational state is parallel to A^-1 b. Because a
    normalized quantum state cannot encode the classical solution magnitude,
    solution() restores the least-squares-optimal complex scale before
    reporting the residual.

    The optimizer is SPSA, so every update needs two gradient objective
    evaluations regardless of parameter count. This keeps the optimizer useful
    for shot-based backend adapters.

    Parameters
    ----------
    matrix:
        Square linear-system matrix. Its dimension must be a power of two.
        Hermiticity is not required because the cost uses A^H A.
    vector:
        Non-zero right-hand-side vector.
    state_fn:
        Callable mapping a real parameter vector to a statevector. When omitted,
        ry_ring_state is used.
    num_parameters:
        Parameter count for a custom state_fn. Inferred for the built-in ansatz.
    layers:
        Layers in the built-in RY/CNOT-ring ansatz.
    max_iterations:
        Maximum SPSA updates.
    tolerance:
        Cost considered converged.
    learning_rate:
        Initial SPSA learning-rate coefficient.
    perturbation:
        Initial SPSA finite-difference perturbation.
    seed:
        Random seed for reproducible perturbations and initialization.
    parameter_bounds:
        Optional clipping bounds.
    gradient_clip:
        Optional Euclidean gradient norm cap.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[complex]],
        vector: Sequence[complex],
        state_fn: Optional[StateFunction] = None,
        num_parameters: Optional[int] = None,
        layers: int = 2,
        max_iterations: int = 250,
        tolerance: float = 1e-6,
        learning_rate: float = 0.2,
        perturbation: float = 0.12,
        seed: Optional[int] = None,
        parameter_bounds: Optional[Tuple[float, float]] = (-np.pi, np.pi),
        gradient_clip: Optional[float] = 10.0,
    ):
        matrix_array = np.asarray(matrix, dtype=np.complex128)
        if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
            raise ValueError("matrix must be square")
        dimension = int(matrix_array.shape[0])
        if dimension < 2 or dimension & (dimension - 1):
            raise ValueError("matrix dimension must be a power of two >= 2")
        if not np.all(np.isfinite(matrix_array)):
            raise ValueError("matrix contains non-finite values")

        vector_array = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vector_array.size != dimension:
            raise ValueError("vector length must match matrix dimension")
        if not np.all(np.isfinite(vector_array)):
            raise ValueError("vector contains non-finite values")
        vector_norm = float(np.linalg.norm(vector_array))
        if vector_norm <= np.finfo(float).tiny:
            raise ValueError("vector must be non-zero")

        matrix_norm = float(np.linalg.norm(matrix_array, ord=2))
        if not np.isfinite(matrix_norm) or matrix_norm <= np.finfo(float).tiny:
            raise ValueError("matrix must have non-zero finite spectral norm")
        if np.linalg.matrix_rank(matrix_array) < dimension:
            raise ValueError("matrix must be non-singular")

        num_qubits = int(np.log2(dimension))
        if state_fn is None:
            if layers < 1:
                raise ValueError("layers must be at least 1")
            inferred_parameters = num_qubits * int(layers)
            if num_parameters is not None and int(num_parameters) != inferred_parameters:
                raise ValueError(
                    "built-in ansatz requires {0} parameters".format(
                        inferred_parameters
                    )
                )

            def built_in(parameters: np.ndarray) -> np.ndarray:
                return ry_ring_state(parameters, num_qubits, int(layers))

            self._state_fn = built_in
            self.num_parameters = inferred_parameters
        else:
            if num_parameters is None or int(num_parameters) < 1:
                raise ValueError(
                    "num_parameters must be supplied for a custom state_fn"
                )
            self._state_fn = state_fn
            self.num_parameters = int(num_parameters)

        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1")
        if not (0.0 < float(tolerance) < 1.0):
            raise ValueError("tolerance must be between 0 and 1")
        if float(learning_rate) <= 0.0 or float(perturbation) <= 0.0:
            raise ValueError("learning_rate and perturbation must be positive")
        if parameter_bounds is not None:
            low, high = map(float, parameter_bounds)
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError("parameter_bounds must satisfy finite low < high")
            self.parameter_bounds = (low, high)
        else:
            self.parameter_bounds = None
        if gradient_clip is not None and float(gradient_clip) <= 0.0:
            raise ValueError("gradient_clip must be positive when supplied")

        self.matrix = matrix_array
        self.vector = vector_array
        self.dimension = dimension
        self.num_qubits = num_qubits
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.learning_rate = float(learning_rate)
        self.perturbation = float(perturbation)
        self.seed = seed
        self.gradient_clip = (
            None if gradient_clip is None else float(gradient_clip)
        )
        self._normalized_vector = vector_array / vector_norm
        self._scaled_matrix = matrix_array / matrix_norm

    def state(self, parameters: Sequence[float]) -> np.ndarray:
        """Evaluate and normalize the configured variational ansatz."""

        params = np.asarray(parameters, dtype=float).reshape(-1)
        if params.size != self.num_parameters:
            raise ValueError(
                "expected {0} parameters, received {1}".format(
                    self.num_parameters, params.size
                )
            )
        if not np.all(np.isfinite(params)):
            raise ValueError("parameters contain non-finite values")
        return _normalize_state(self._state_fn(params.copy()), self.dimension)

    def cost(self, parameters: Sequence[float]) -> float:
        """Evaluate the scale-invariant global VQLS objective."""

        state = self.state(parameters)
        transformed = self._scaled_matrix @ state
        denominator = float(np.vdot(transformed, transformed).real)
        if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
            return 1.0
        overlap = np.vdot(self._normalized_vector, transformed)
        value = 1.0 - (float(abs(overlap) ** 2) / denominator)
        return float(np.clip(value, 0.0, 1.0))

    def solution(
        self, parameters: Sequence[float]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return state, scaled solution, and original-system residual norm."""

        state = self.state(parameters)
        transformed = self.matrix @ state
        denominator = np.vdot(transformed, transformed)
        if abs(denominator) <= np.finfo(float).tiny:
            raise ValueError("ansatz maps into a numerically null matrix direction")
        scale = np.vdot(transformed, self.vector) / denominator
        solution = scale * state
        residual = float(np.linalg.norm(self.matrix @ solution - self.vector))
        return state, solution, residual

    def _project(self, parameters: np.ndarray) -> np.ndarray:
        if self.parameter_bounds is None:
            return parameters
        low, high = self.parameter_bounds
        return np.clip(parameters, low, high)

    def solve(
        self, initial_parameters: Optional[Sequence[float]] = None
    ) -> VQLSResult:
        """Optimize the VQLS objective and return the best solution found."""

        rng = np.random.default_rng(self.seed)
        if initial_parameters is None:
            parameters = rng.uniform(
                -np.pi / 4.0, np.pi / 4.0, self.num_parameters
            )
        else:
            parameters = np.asarray(initial_parameters, dtype=float).reshape(-1)
            if parameters.size != self.num_parameters:
                raise ValueError(
                    "initial_parameters must contain {0} values".format(
                        self.num_parameters
                    )
                )
            if not np.all(np.isfinite(parameters)):
                raise ValueError("initial_parameters contain non-finite values")
        parameters = self._project(parameters.astype(float, copy=True))

        best_parameters = parameters.copy()
        best_cost = self.cost(parameters)
        history = []
        iterations = 0

        for step in range(1, self.max_iterations + 1):
            if best_cost <= self.tolerance:
                break

            ak = self.learning_rate / ((step + 10.0) ** 0.602)
            ck = self.perturbation / (step ** 0.101)
            delta = rng.choice(np.array([-1.0, 1.0]), self.num_parameters)

            plus = self._project(parameters + ck * delta)
            minus = self._project(parameters - ck * delta)
            gradient = ((self.cost(plus) - self.cost(minus)) / (2.0 * ck)) * delta

            if self.gradient_clip is not None:
                gradient_norm = float(np.linalg.norm(gradient))
                if gradient_norm > self.gradient_clip:
                    gradient *= self.gradient_clip / gradient_norm

            parameters = self._project(parameters - ak * gradient)
            current_cost = self.cost(parameters)
            history.append(current_cost)
            iterations = step

            if current_cost < best_cost:
                best_cost = current_cost
                best_parameters = parameters.copy()

        state, solution, residual = self.solution(best_parameters)
        return VQLSResult(
            parameters=best_parameters,
            state=state,
            solution=solution,
            cost=float(best_cost),
            residual_norm=residual,
            iterations=iterations,
            converged=bool(best_cost <= self.tolerance),
            cost_history=tuple(float(value) for value in history),
        )
