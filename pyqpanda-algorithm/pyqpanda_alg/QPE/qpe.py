"""Quantum Phase Estimation for PyQPanda3.

The module exposes two layers:

* Pure reference helpers for the exact ideal QPE distribution and deterministic
  phase decoding. These helpers do not import PyQPanda3.
* A lazy PyQPanda3 circuit builder/runner for arbitrary unitary circuits and
  caller-supplied eigenstate preparation.

Phase convention
----------------
Eigenphases are fractions in [0, 1): U|psi> = exp(2*pi*i*phase)|psi>.
Counting-register powers follow the same ordering used by pyqpanda_alg.QAE:
counting qubit i controls U**(2**i), followed by inverse QFT over the counting
register. Runtime bit strings are decoded using the integer convention returned
by PyQPanda3's probability dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


CircuitFactory = Callable[[Sequence[Any]], Any]


def _validate_precision_bits(precision_bits: int) -> int:
    if not isinstance(precision_bits, int) or isinstance(precision_bits, bool):
        raise ValueError("precision_bits must be an integer")
    if precision_bits <= 0:
        raise ValueError("precision_bits must be positive")
    if precision_bits > 24:
        raise ValueError("precision_bits must be <= 24")
    return precision_bits


def canonical_phase(phase: float) -> float:
    """Return a finite phase reduced into the half-open interval [0, 1)."""
    try:
        value = float(phase)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError("phase must be a finite real number")
    value %= 1.0
    # Avoid returning 1.0 through floating-point edge cases.
    return 0.0 if value == 1.0 else value


def circular_phase_distance(left: float, right: float) -> float:
    """Return the shortest distance between two phases on the unit circle."""
    a = canonical_phase(left)
    b = canonical_phase(right)
    delta = abs(a - b)
    return min(delta, 1.0 - delta)


def qpe_reference_probabilities(
    phase: float,
    precision_bits: int,
) -> tuple[float, ...]:
    """Return the ideal QPE counting-register distribution.

    For N = 2**precision_bits and output integer y, the probability is

        |(1/N) sum_k exp(2*pi*i*k*(phase-y/N))|**2.

    The implementation evaluates the closed-form Dirichlet-kernel expression,
    with an exact-grid branch that avoids the removable zero/zero singularity.
    """
    bits = _validate_precision_bits(precision_bits)
    phi = canonical_phase(phase)
    size = 1 << bits
    probabilities: list[float] = []

    for outcome in range(size):
        # math.remainder gives the numerically closest periodic displacement.
        delta = math.remainder(phi - outcome / size, 1.0)
        if abs(delta) <= 1e-15:
            probability = 1.0
        else:
            denominator = math.sin(math.pi * delta)
            if abs(denominator) <= 1e-15:
                probability = 1.0
            else:
                numerator = math.sin(math.pi * size * delta)
                probability = (numerator / (size * denominator)) ** 2
        probabilities.append(max(0.0, float(probability)))

    total = math.fsum(probabilities)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("reference distribution is not normalizable")
    return tuple(value / total for value in probabilities)


def _validate_probability_vector(
    probabilities: Sequence[float],
    precision_bits: int,
) -> tuple[float, ...]:
    bits = _validate_precision_bits(precision_bits)
    expected = 1 << bits
    if len(probabilities) != expected:
        raise ValueError(
            f"expected {expected} probabilities for {bits} precision bits"
        )
    values: list[float] = []
    for probability in probabilities:
        try:
            value = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError("probabilities must be finite numeric values") from exc
        if not math.isfinite(value):
            raise ValueError("probabilities must be finite")
        if value < 0.0:
            raise ValueError("probabilities must be non-negative")
        values.append(value)
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("probabilities must contain positive mass")
    return tuple(value / total for value in values)


def decode_phase(
    probabilities: Sequence[float],
    precision_bits: int,
) -> float:
    """Decode the maximum-likelihood grid phase from a QPE distribution."""
    values = _validate_probability_vector(probabilities, precision_bits)
    outcome = max(range(len(values)), key=lambda index: values[index])
    return outcome / (1 << precision_bits)


def decode_phase_mapping(
    probabilities: Mapping[str, int | float],
    precision_bits: int,
) -> float:
    """Decode a PyQPanda-style binary probability/count mapping.

    Keys are interpreted exactly as conventional binary integers, matching the
    existing QAE runner in pyqpanda-algorithm.
    """
    bits = _validate_precision_bits(precision_bits)
    if not probabilities:
        raise ValueError("probability mapping must not be empty")

    winner: str | None = None
    winner_weight = -1.0
    for raw_key, raw_weight in probabilities.items():
        key = str(raw_key)
        if len(key) != bits or any(ch not in "01" for ch in key):
            raise ValueError(
                f"every probability key must be a {bits}-bit binary string"
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("probability weights must be finite numbers") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("probability weights must be finite and non-negative")
        if weight > winner_weight or (
            weight == winner_weight and (winner is None or key < winner)
        ):
            winner = key
            winner_weight = weight

    if winner is None or winner_weight <= 0.0:
        raise ValueError("probability mapping must contain positive mass")
    return int(winner, 2) / (1 << bits)


def nearest_phase_grid_point(phase: float, precision_bits: int) -> float:
    """Return the nearest representable QPE phase grid point."""
    bits = _validate_precision_bits(precision_bits)
    phi = canonical_phase(phase)
    size = 1 << bits
    return (int(math.floor(phi * size + 0.5)) % size) / size


def _resolve_registers(
    precision_bits: int,
    target_width: int,
    counting_qubits: Iterable[Any] | None,
    target_qubits: Iterable[Any] | None,
) -> tuple[list[Any], list[Any]]:
    bits = _validate_precision_bits(precision_bits)
    if not isinstance(target_width, int) or isinstance(target_width, bool):
        raise ValueError("target_width must be an integer")
    if target_width <= 0:
        raise ValueError("target_width must be positive")

    counting = (
        list(range(bits))
        if counting_qubits is None
        else list(counting_qubits)
    )
    target = (
        list(range(bits, bits + target_width))
        if target_qubits is None
        else list(target_qubits)
    )
    if len(counting) != bits:
        raise ValueError(f"expected exactly {bits} counting qubits")
    if len(target) != target_width:
        raise ValueError(f"expected exactly {target_width} target qubits")
    labels = [str(qubit) for qubit in counting + target]
    if len(set(labels)) != len(labels):
        raise ValueError("counting and target qubits must be distinct")
    return counting, target


def qpe_circuit(
    unitary: CircuitFactory,
    *,
    target_width: int,
    precision_bits: int,
    prepare_eigenstate: CircuitFactory | None = None,
    counting_qubits: Iterable[Any] | None = None,
    target_qubits: Iterable[Any] | None = None,
):
    """Build a standard Quantum Phase Estimation circuit.

    unitary(target_qubits) must return a QCircuit implementing U.
    prepare_eigenstate(target_qubits), when provided, must return a circuit
    preparing an eigenstate of U from |0...0>.

    The circuit does not add measurements, allowing callers to compose it into
    larger programs or inspect exact probabilities.
    """
    if not callable(unitary):
        raise ValueError("unitary must be callable")
    if prepare_eigenstate is not None and not callable(prepare_eigenstate):
        raise ValueError("prepare_eigenstate must be callable")

    counting, target = _resolve_registers(
        precision_bits,
        target_width,
        counting_qubits,
        target_qubits,
    )

    from pyqpanda3.core import H, QCircuit
    from ..plugin import QFT

    circuit = QCircuit()
    if prepare_eigenstate is not None:
        circuit << prepare_eigenstate(target)

    for qubit in counting:
        circuit << H(qubit)

    # Match the power ordering already used by pyqpanda_alg.QAE.
    for power_index, control in enumerate(counting):
        repetitions = 1 << power_index
        for _ in range(repetitions):
            operation = unitary(target)
            if operation is None:
                raise ValueError("unitary factory returned None")
            circuit << operation.control([control])

    circuit << QFT(counting).dagger()
    return circuit


@dataclass(frozen=True)
class QPEResult:
    """Result of a CPUQVM QPE execution."""

    phase: float
    bitstring: str
    probabilities: tuple[tuple[str, float], ...]

    def probability_dict(self) -> dict[str, float]:
        return dict(self.probabilities)


def run_qpe(
    unitary: CircuitFactory,
    *,
    target_width: int,
    precision_bits: int = 6,
    prepare_eigenstate: CircuitFactory | None = None,
    shots: int = 1024,
) -> QPEResult:
    """Execute QPE on CPUQVM and return the maximum-likelihood eigenphase.

    The arbitrary unitary is supplied as a circuit factory so the same public
    API supports one- and multi-qubit eigenstates without matrix conversion.
    """
    bits = _validate_precision_bits(precision_bits)
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")

    counting = list(range(bits))
    target = list(range(bits, bits + target_width))

    from pyqpanda3.core import CPUQVM, QProg

    program = QProg()
    program << qpe_circuit(
        unitary,
        target_width=target_width,
        precision_bits=bits,
        prepare_eigenstate=prepare_eigenstate,
        counting_qubits=counting,
        target_qubits=target,
    )

    machine = CPUQVM()
    machine.run(program, shots)
    raw = machine.result().get_prob_dict(counting)
    normalized = {str(key): float(value) for key, value in raw.items()}
    phase = decode_phase_mapping(normalized, bits)
    winner = max(
        normalized,
        key=lambda key: (normalized[key], -int(key, 2)),
    )
    ordered = tuple(sorted(normalized.items(), key=lambda item: item[0]))
    return QPEResult(
        phase=phase,
        bitstring=winner,
        probabilities=ordered,
    )
