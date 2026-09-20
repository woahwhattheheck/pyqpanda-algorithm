"""Hadamard Test expectation estimation for PyQPanda3.

The Hadamard Test estimates the real or imaginary component of

    <psi|U|psi>

for a caller-supplied unitary U and eigenstate/general state preparation.

Pure probability helpers are independent of PyQPanda3. Circuit construction
and CPUQVM execution import PyQPanda3 lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any


CircuitFactory = Callable[[Sequence[Any]], Any]


def _component_name(component: str) -> str:
    value = str(component).lower()
    if value not in ("real", "imag"):
        raise ValueError("component must be 'real' or 'imag'")
    return value


def _finite_complex(value: complex) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expectation must be a finite complex number") from exc
    if not (
        math.isfinite(result.real)
        and math.isfinite(result.imag)
    ):
        raise ValueError("expectation must be finite")
    return result


def reference_probabilities(
    expectation: complex,
    component: str = "real",
    *,
    tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Return ideal ancilla probabilities for a Hadamard Test.

    For a unitary expectation z = <psi|U|psi>:

      real test: P(0) = (1 + Re(z)) / 2
      imag test: P(0) = (1 + Im(z)) / 2

    P(1) is the complementary probability.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    z = _finite_complex(expectation)
    if abs(z) > 1.0 + tolerance:
        raise ValueError("unitary expectation magnitude cannot exceed 1")
    name = _component_name(component)
    value = z.real if name == "real" else z.imag
    if value < -1.0 - tolerance or value > 1.0 + tolerance:
        raise ValueError("expectation component must lie in [-1, 1]")
    value = min(1.0, max(-1.0, value))
    p0 = (1.0 + value) / 2.0
    return p0, 1.0 - p0


def decode_component(probabilities: Sequence[float]) -> float:
    """Decode P(0)-P(1) from a two-outcome probability/count vector."""
    if len(probabilities) != 2:
        raise ValueError("expected exactly two ancilla outcomes")
    values: list[float] = []
    for raw in probabilities:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome weights must be finite numbers") from exc
        if not math.isfinite(value):
            raise ValueError("outcome weights must be finite")
        if value < 0:
            raise ValueError("outcome weights must be non-negative")
        values.append(value)
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("outcome weights must contain positive mass")
    return (values[0] - values[1]) / total


def combine_components(real: float, imag: float) -> complex:
    """Combine validated real/imag Hadamard-Test estimates."""
    values: list[float] = []
    for name, raw in (("real", real), ("imag", imag)):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} component must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} component must be finite")
        if value < -1.0 or value > 1.0:
            raise ValueError(f"{name} component must lie in [-1, 1]")
        values.append(value)
    estimate = complex(values[0], values[1])
    if abs(estimate) > 1.0 + 1e-12:
        raise ValueError("combined unitary expectation magnitude cannot exceed 1")
    return estimate


def _resolve_registers(
    target_width: int,
    ancilla_qubit: Any | None,
    target_qubits: Iterable[Any] | None,
) -> tuple[Any, list[Any]]:
    if not isinstance(target_width, int) or isinstance(target_width, bool):
        raise ValueError("target_width must be an integer")
    if target_width <= 0:
        raise ValueError("target_width must be positive")

    ancilla = 0 if ancilla_qubit is None else ancilla_qubit
    target = (
        list(range(1, target_width + 1))
        if target_qubits is None
        else list(target_qubits)
    )
    if len(target) != target_width:
        raise ValueError(f"expected exactly {target_width} target qubits")
    labels = [str(ancilla)] + [str(qubit) for qubit in target]
    if len(set(labels)) != len(labels):
        raise ValueError("ancilla and target qubits must be distinct")
    return ancilla, target


def hadamard_test_circuit(
    unitary: CircuitFactory,
    *,
    target_width: int,
    prepare_state: CircuitFactory | None = None,
    component: str = "real",
    ancilla_qubit: Any | None = None,
    target_qubits: Iterable[Any] | None = None,
):
    """Build a Hadamard Test circuit without measurements.

    unitary(target_qubits) must return a QCircuit implementing U.
    prepare_state(target_qubits), when provided, prepares |psi> from |0...0>.

    For the imaginary component, RX(-pi/2) rotates the ancilla Y observable
    onto Z before readout. The real component uses the standard final H gate.
    """
    if not callable(unitary):
        raise ValueError("unitary must be callable")
    if prepare_state is not None and not callable(prepare_state):
        raise ValueError("prepare_state must be callable")
    name = _component_name(component)
    ancilla, target = _resolve_registers(
        target_width,
        ancilla_qubit,
        target_qubits,
    )

    from pyqpanda3.core import H, RX, QCircuit

    circuit = QCircuit()
    if prepare_state is not None:
        prepared = prepare_state(target)
        if prepared is None:
            raise ValueError("prepare_state factory returned None")
        circuit << prepared

    circuit << H(ancilla)
    operation = unitary(target)
    if operation is None:
        raise ValueError("unitary factory returned None")
    circuit << operation.control([ancilla])

    if name == "real":
        circuit << H(ancilla)
    else:
        circuit << RX(ancilla, -math.pi / 2.0)

    return circuit


@dataclass(frozen=True)
class HadamardTestResult:
    """One component estimated by CPUQVM."""

    component: str
    value: float
    p0: float
    p1: float
    shots: int

    @property
    def probabilities(self) -> tuple[float, float]:
        return self.p0, self.p1


def run_hadamard_test(
    unitary: CircuitFactory,
    *,
    target_width: int,
    prepare_state: CircuitFactory | None = None,
    component: str = "real",
    shots: int = 1024,
) -> HadamardTestResult:
    """Execute one real/imag Hadamard Test component on CPUQVM."""
    name = _component_name(component)
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer")

    ancilla = 0
    target = list(range(1, target_width + 1))

    from pyqpanda3.core import CPUQVM, QProg

    program = QProg()
    program << hadamard_test_circuit(
        unitary,
        target_width=target_width,
        prepare_state=prepare_state,
        component=name,
        ancilla_qubit=ancilla,
        target_qubits=target,
    )

    machine = CPUQVM()
    machine.run(program, shots)
    raw = machine.result().get_prob_list([ancilla])
    if len(raw) != 2:
        raise RuntimeError("CPUQVM returned an invalid ancilla probability vector")
    p0, p1 = (float(raw[0]), float(raw[1]))
    value = decode_component((p0, p1))
    return HadamardTestResult(
        component=name,
        value=value,
        p0=p0,
        p1=p1,
        shots=shots,
    )


def run_complex_hadamard_test(
    unitary: CircuitFactory,
    *,
    target_width: int,
    prepare_state: CircuitFactory | None = None,
    shots: int = 1024,
) -> complex:
    """Estimate both components and return <psi|U|psi> as a complex number."""
    real = run_hadamard_test(
        unitary,
        target_width=target_width,
        prepare_state=prepare_state,
        component="real",
        shots=shots,
    ).value
    imag = run_hadamard_test(
        unitary,
        target_width=target_width,
        prepare_state=prepare_state,
        component="imag",
        shots=shots,
    ).value
    return combine_components(real, imag)
