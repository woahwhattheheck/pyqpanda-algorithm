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

"""Adaptive Derivative-Assembled Pseudo-Trotter VQE (ADAPT-VQE).

Callers provide an energy evaluator that maps an ordered operator sequence and
its parameters to an expectation value. This keeps the adaptive optimizer
usable with CPUQVM, state-vector, cloud/QPU, or application-specific
expectation backends.

The built-in gradient path uses the exact parameter-shift rule for Pauli
rotations exp(-i * theta * P / 2). Backends with a different generator
convention can provide explicit gradient callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


OperatorId = Hashable
EnergyEvaluator = Callable[[Sequence[OperatorId], np.ndarray], float]
EnergyRequest = tuple[Sequence[OperatorId], np.ndarray]
BatchEnergyEvaluator = Callable[[Sequence[EnergyRequest]], Sequence[float]]
CandidateGradientEvaluator = Callable[
    [Sequence[OperatorId], np.ndarray, Sequence[OperatorId]], Sequence[float]
]
ParameterGradientEvaluator = Callable[[Sequence[OperatorId], np.ndarray], Sequence[float]]
IterationCallback = Callable[["AdaptIteration"], None]


@dataclass(frozen=True)
class AdaptVQEConfig:
    """Controls adaptive growth and classical re-optimization."""

    gradient_threshold: float = 1.0e-3
    max_adapt_iterations: int = 20
    parameter_shift: float = np.pi / 2.0
    optimizer_method: str = "BFGS"
    optimizer_maxiter: int = 120
    optimizer_gtol: float = 1.0e-6
    energy_tolerance: float = 1.0e-10
    allow_repeated_operators: bool = True

    def __post_init__(self) -> None:
        if self.gradient_threshold <= 0:
            raise ValueError("gradient_threshold must be positive")
        if self.max_adapt_iterations <= 0:
            raise ValueError("max_adapt_iterations must be positive")
        if not np.isfinite(self.parameter_shift) or self.parameter_shift == 0:
            raise ValueError("parameter_shift must be finite and non-zero")
        if self.optimizer_maxiter <= 0:
            raise ValueError("optimizer_maxiter must be positive")
        if self.optimizer_gtol <= 0:
            raise ValueError("optimizer_gtol must be positive")
        if self.energy_tolerance < 0:
            raise ValueError("energy_tolerance must be non-negative")


@dataclass(frozen=True)
class AdaptIteration:
    """One completed ADAPT growth and re-optimization cycle."""

    iteration: int
    selected_operator: OperatorId
    selected_gradient: float
    pool_gradient_norm: float
    energy_before: float
    energy_after: float
    energy_improvement: float
    parameters: tuple[float, ...]
    optimizer_success: bool
    optimizer_message: str


@dataclass(frozen=True)
class AdaptVQEResult:
    """Final result plus the complete adaptive trace."""

    energy: float
    operators: tuple[OperatorId, ...]
    parameters: np.ndarray
    converged: bool
    convergence_reason: str
    adapt_iterations: int
    energy_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    iterations: tuple[AdaptIteration, ...] = field(default_factory=tuple)

    @property
    def ansatz_depth(self) -> int:
        return len(self.operators)


class ADAPTVQE:
    """Grow an ansatz by ranking an operator pool by energy gradient."""

    def __init__(
        self,
        energy_evaluator: EnergyEvaluator,
        operator_pool: Iterable[OperatorId],
        *,
        batch_energy_evaluator: Optional[BatchEnergyEvaluator] = None,
        candidate_gradient_evaluator: Optional[CandidateGradientEvaluator] = None,
        parameter_gradient_evaluator: Optional[ParameterGradientEvaluator] = None,
        config: Optional[AdaptVQEConfig] = None,
        callback: Optional[IterationCallback] = None,
    ) -> None:
        self.energy_evaluator = energy_evaluator
        self.batch_energy_evaluator = batch_energy_evaluator
        self.candidate_gradient_evaluator = candidate_gradient_evaluator
        self.parameter_gradient_evaluator = parameter_gradient_evaluator
        self.config = config or AdaptVQEConfig()
        self.callback = callback

        self.operator_pool = tuple(operator_pool)
        if not self.operator_pool:
            raise ValueError("operator_pool must contain at least one operator")
        if len(set(self.operator_pool)) != len(self.operator_pool):
            raise ValueError("operator_pool identifiers must be unique")

    @staticmethod
    def _as_parameters(values: Sequence[float], expected: int) -> np.ndarray:
        parameters = np.asarray(values, dtype=float).reshape(-1)
        if parameters.size != expected:
            raise ValueError(
                f"expected {expected} variational parameters, got {parameters.size}"
            )
        if not np.all(np.isfinite(parameters)):
            raise ValueError("variational parameters must be finite")
        return parameters

    @staticmethod
    def _as_energy(value: Any) -> float:
        energy = float(np.real(value))
        if not np.isfinite(energy):
            raise ValueError(f"energy evaluator returned non-finite value: {value!r}")
        return energy

    def _evaluate(
        self, operators: Sequence[OperatorId], parameters: Sequence[float]
    ) -> float:
        params = self._as_parameters(parameters, len(operators))
        return self._as_energy(self.energy_evaluator(tuple(operators), params.copy()))

    def _evaluate_requests(self, requests: Sequence[EnergyRequest]) -> np.ndarray:
        if not requests:
            return np.empty(0, dtype=float)

        normalized: list[EnergyRequest] = []
        for operators, parameters in requests:
            ops = tuple(operators)
            params = self._as_parameters(parameters, len(ops))
            normalized.append((ops, params.copy()))

        if self.batch_energy_evaluator is None:
            return np.asarray(
                [self._evaluate(ops, params) for ops, params in normalized],
                dtype=float,
            )

        values = np.asarray(
            list(self.batch_energy_evaluator(normalized)), dtype=float
        ).reshape(-1)
        if values.size != len(normalized):
            raise ValueError(
                "batch_energy_evaluator returned "
                f"{values.size} energies for {len(normalized)} requests"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("batch_energy_evaluator returned a non-finite energy")
        return values

    def _shift_gradient(
        self,
        operators: Sequence[OperatorId],
        parameters: np.ndarray,
    ) -> np.ndarray:
        if not operators:
            return np.empty(0, dtype=float)

        shift = self.config.parameter_shift
        requests: list[EnergyRequest] = []
        for index in range(len(operators)):
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += shift
            minus[index] -= shift
            requests.extend(((operators, plus), (operators, minus)))

        energies = self._evaluate_requests(requests).reshape(-1, 2)
        return 0.5 * (energies[:, 0] - energies[:, 1])

    def _parameter_gradient(
        self, operators: Sequence[OperatorId], parameters: np.ndarray
    ) -> np.ndarray:
        if self.parameter_gradient_evaluator is not None:
            gradient = np.asarray(
                self.parameter_gradient_evaluator(tuple(operators), parameters.copy()),
                dtype=float,
            ).reshape(-1)
            if gradient.size != len(operators):
                raise ValueError(
                    "parameter_gradient_evaluator returned the wrong gradient length"
                )
            if not np.all(np.isfinite(gradient)):
                raise ValueError(
                    "parameter_gradient_evaluator returned non-finite values"
                )
            return gradient
        return self._shift_gradient(operators, parameters)

    def _candidate_gradients(
        self,
        operators: Sequence[OperatorId],
        parameters: np.ndarray,
        candidates: Sequence[OperatorId],
    ) -> np.ndarray:
        if self.candidate_gradient_evaluator is not None:
            gradient = np.asarray(
                self.candidate_gradient_evaluator(
                    tuple(operators), parameters.copy(), tuple(candidates)
                ),
                dtype=float,
            ).reshape(-1)
            if gradient.size != len(candidates):
                raise ValueError(
                    "candidate_gradient_evaluator returned the wrong gradient length"
                )
            if not np.all(np.isfinite(gradient)):
                raise ValueError(
                    "candidate_gradient_evaluator returned non-finite values"
                )
            return gradient

        shift = self.config.parameter_shift
        requests: list[EnergyRequest] = []
        for candidate in candidates:
            extended = tuple(operators) + (candidate,)
            plus = np.append(parameters, shift)
            minus = np.append(parameters, -shift)
            requests.extend(((extended, plus), (extended, minus)))

        energies = self._evaluate_requests(requests).reshape(-1, 2)
        return 0.5 * (energies[:, 0] - energies[:, 1])

    def _optimize(
        self,
        operators: Sequence[OperatorId],
        initial_parameters: np.ndarray,
    ):
        def objective(values: np.ndarray) -> float:
            return self._evaluate(operators, values)

        def jacobian(values: np.ndarray) -> np.ndarray:
            params = self._as_parameters(values, len(operators))
            return self._parameter_gradient(operators, params)

        return minimize(
            objective,
            x0=initial_parameters,
            jac=jacobian,
            method=self.config.optimizer_method,
            options={
                "maxiter": self.config.optimizer_maxiter,
                "gtol": self.config.optimizer_gtol,
            },
        )

    def run(
        self,
        initial_operators: Sequence[OperatorId] = (),
        initial_parameters: Optional[Sequence[float]] = None,
    ) -> AdaptVQEResult:
        """Run until pool-gradient convergence or the adaptive-iteration cap."""

        operators = list(initial_operators)
        if initial_parameters is None:
            parameters = np.zeros(len(operators), dtype=float)
        else:
            parameters = self._as_parameters(initial_parameters, len(operators))

        energy = self._evaluate(operators, parameters)
        energy_history = [energy]
        gradient_norm_history: list[float] = []
        iteration_history: list[AdaptIteration] = []

        converged = False
        reason = "maximum adaptive iterations reached"

        for adapt_index in range(1, self.config.max_adapt_iterations + 1):
            if self.config.allow_repeated_operators:
                candidates = list(self.operator_pool)
            else:
                selected = set(operators)
                candidates = [op for op in self.operator_pool if op not in selected]

            if not candidates:
                converged = True
                reason = "operator pool exhausted"
                break

            pool_gradients = self._candidate_gradients(
                operators, parameters, candidates
            )
            gradient_norm = float(np.linalg.norm(pool_gradients))
            gradient_norm_history.append(gradient_norm)

            if gradient_norm < self.config.gradient_threshold:
                converged = True
                reason = (
                    "operator-pool gradient norm "
                    f"{gradient_norm:.6g} is below threshold "
                    f"{self.config.gradient_threshold:.6g}"
                )
                break

            selected_index = int(np.argmax(np.abs(pool_gradients)))
            selected_operator = candidates[selected_index]
            selected_gradient = float(pool_gradients[selected_index])

            energy_before = energy
            operators.append(selected_operator)
            parameters = np.append(parameters, 0.0)

            optimized = self._optimize(operators, parameters)
            parameters = self._as_parameters(optimized.x, len(operators))
            energy = self._as_energy(optimized.fun)
            improvement = energy_before - energy

            record = AdaptIteration(
                iteration=adapt_index,
                selected_operator=selected_operator,
                selected_gradient=selected_gradient,
                pool_gradient_norm=gradient_norm,
                energy_before=energy_before,
                energy_after=energy,
                energy_improvement=improvement,
                parameters=tuple(float(x) for x in parameters),
                optimizer_success=bool(optimized.success),
                optimizer_message=str(optimized.message),
            )
            iteration_history.append(record)
            energy_history.append(energy)

            if self.callback is not None:
                self.callback(record)

            if (
                abs(improvement) <= self.config.energy_tolerance
                and gradient_norm
                <= max(
                    self.config.gradient_threshold * 10.0,
                    self.config.optimizer_gtol,
                )
            ):
                converged = True
                reason = "energy and pool-gradient changes are below tolerances"
                break

        return AdaptVQEResult(
            energy=energy,
            operators=tuple(operators),
            parameters=parameters.copy(),
            converged=converged,
            convergence_reason=reason,
            adapt_iterations=len(iteration_history),
            energy_history=tuple(energy_history),
            gradient_norm_history=tuple(gradient_norm_history),
            iterations=tuple(iteration_history),
        )
