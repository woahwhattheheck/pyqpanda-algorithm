"""Reusable SWAP-test fidelity estimation for PyQPanda3.

For normalized pure states |psi> and |phi>, the standard ancilla SWAP test has

    P(ancilla=0) = (1 + |<psi|phi>|^2) / 2.

This module exposes backend-independent reference helpers plus a lazy
PyQPanda3 circuit builder/CPUQVM runner for arbitrary equal-width state
preparation circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any


CircuitFactory = Callable[[Sequence[Any]], Any]


def _finite_complex(value: complex, *, name: str) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be finite complex numbers") from exc
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} entries must be finite")
    return result


def pure_state_fidelity(
    left: Sequence[complex],
    right: Sequence[complex],
) -> float:
    """Return normalized pure-state fidelity |<left|right>|^2.

    Inputs need not already be normalized. They must have equal non-zero length
    and non-zero finite norm. The helper intentionally avoids NumPy so the
    reference contract remains usable without a quantum runtime.
    """
    if len(left) == 0 or len(right) == 0:
        raise ValueError("state vectors must not be empty")
    if len(left) != len(right):
        raise ValueError("state vectors must have equal length")

    lhs = tuple(_finite_complex(value, name="left") for value in left)
    rhs = tuple(_finite_complex(value, name="right") for value in right)

    lhs_norm_sq = math.fsum(abs(value) ** 2 for value in lhs)
    rhs_norm_sq = math.fsum(abs(value) ** 2 for value in rhs)
    if lhs_norm_sq <= 0.0 or rhs_norm_sq <= 0.0:
        raise ValueError("state vectors must have non-zero norm")
    if not math.isfinite(lhs_norm_sq) or not math.isfinite(rhs_norm_sq):
        raise ValueError("state-vector norm must be finite")

    inner = sum(a.conjugate() * b for a, b in zip(lhs, rhs))
    fidelity = (abs(inner) ** 2) / (lhs_norm_sq * rhs_norm_sq)
    # Bound round-off while preserving fail-closed behavior for real errors.
    if fidelity < -1e-12 or fidelity > 1.0 + 1e-12:
        raise ValueError("computed fidelity lies outside [0, 1]")
    return min(1.0, max(0.0, float(fidelity)))


def reference_probabilities(fidelity: float) -> tuple[float, float]:
    """Return exact ancilla probabilities (P0, P1) for a SWAP test."""
    try:
        value = float(fidelity)
    except (TypeError, ValueError) as exc:
        raise ValueError("fidelity must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("fidelity must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("fidelity must lie in [0, 1]")
    p0 = (1.0 + value) / 2.0
    return p0, 1.0 - p0


def decode_fidelity(
    probabilities: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> float:
    """Decode fidelity from two ancilla outcome weights.

    Counts and probabilities are both accepted because the vector is normalized
    internally. Small floating-point excursions are clipped at the physical
    interval; materially invalid values fail closed.
    """
    if len(probabilities) != 2:
        raise ValueError("expected exactly two ancilla outcomes")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    values: list[float] = []
    for raw in probabilities:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome weights must be finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("outcome weights must be finite")
        if value < 0.0:
            raise ValueError("outcome weights must be non-negative")
        values.append(value)

    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("outcome weights must contain positive mass")
    p0 = values[0] / total
    fidelity = 2.0 * p0 - 1.0
    if fidelity < -tolerance or fidelity > 1.0 + tolerance:
        raise ValueError(
            "ancilla distribution is incompatible with a standard SWAP test"
        )
    return min(1.0, max(0.0, fidelity))


def overlap_magnitude_from_fidelity(fidelity: float) -> float:
    """Return |<psi|phi>| from a validated pure-state fidelity."""
    p0, _ = reference_probabilities(float(fidelity))
    decoded = 2.0 * p0 - 1.0
    return math.sqrt(max(0.0, decoded))


def _resolve_registers(
    width: int,
    ancilla_qubit: Any | None,
    left_qubits: Iterable[Any] | None,
    right_qubits: Iterable[Any] | None,
) -> tuple[Any, list[Any], list[Any]]:
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")

    ancilla = 0 if ancilla_qubit is None else ancilla_qubit
    left = list(range(1, 1 + width)) if left_qubits is None else list(left_qubits)
    right = (
        list(range(1 + width, 1 + 2 * width))
        if right_qubits is None
        else list(right_qubits)
    )
    if len(left) != width or len(right) != width:
        raise ValueError(f"expected exactly {width} qubits in each state register")

    labels = [str(ancilla)] + [str(q) for q in left + right]
    if len(labels) != len(set(labels)):
        raise ValueError("ancilla, left, and right registers must be disjoint")
    return ancilla, left, right


def swap_test_circuit(
    prepare_left: CircuitFactory,
    prepare_right: CircuitFactory,
    *,
    width: int,
    ancilla_qubit: Any | None = None,
    left_qubits: Iterable[Any] | None = None,
    right_qubits: Iterable[Any] | None = None,
):
    """Build the standard pure-state SWAP-test circuit without measurement.

    The preparation factories receive their complete equal-width register and
    must return QCircuit-compatible operations that prepare the compared states
    from |0...0>. Each register pair is swapped under common ancilla control.
    """
    if not callable(prepare_left) or not callable(prepare_right):
        raise ValueError("state preparation factories must be callable")
    ancilla, left, right = _resolve_registers(
        width,
        ancilla_qubit,
        left_qubits,
        right_qubits,
    )

    from pyqpanda3.core import H, QCircuit, SWAP

    circuit = QCircuit()
    left_prep = prepare_left(left)
    right_prep = prepare_right(right)
    if left_prep is None or right_prep is None:
        raise ValueError("state preparation factory returned None")

    circuit << left_prep
    circuit << right_prep
    circuit << H(ancilla)
    for left_qubit, right_qubit in zip(left, right):
        circuit << SWAP(left_qubit, right_qubit).control([ancilla])
    circuit << H(ancilla)
    return circuit


@dataclass(frozen=True)
class SwapTestResult:
    """CPUQVM result for one SWAP-test execution."""

    fidelity: float
    overlap_magnitude: float
    p0: float
    p1: float
    shots: int

    @property
    def probabilities(self) -> tuple[float, float]:
        return self.p0, self.p1


def run_swap_test(
    prepare_left: CircuitFactory,
    prepare_right: CircuitFactory,
    *,
    width: int,
    shots: int = 1024,
) -> SwapTestResult:
    """Execute a standard pure-state SWAP test on CPUQVM."""
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")

    ancilla = 0
    left = list(range(1, 1 + width))
    right = list(range(1 + width, 1 + 2 * width))

    from pyqpanda3.core import CPUQVM, QProg

    program = QProg()
    program << swap_test_circuit(
        prepare_left,
        prepare_right,
        width=width,
        ancilla_qubit=ancilla,
        left_qubits=left,
        right_qubits=right,
    )

    machine = CPUQVM()
    machine.run(program, shots)
    raw = machine.result().get_prob_list([ancilla])
    if len(raw) != 2:
        raise RuntimeError("CPUQVM returned an invalid ancilla probability vector")
    p0, p1 = float(raw[0]), float(raw[1])
    fidelity = decode_fidelity((p0, p1))
    return SwapTestResult(
        fidelity=fidelity,
        overlap_magnitude=math.sqrt(fidelity),
        p0=p0,
        p1=p1,
        shots=shots,
    )
