# Hadamard Test

This module adds the Hadamard Test as a reusable expectation-estimation
primitive for pyqpanda-algorithm.

Given a state |psi> and a unitary U, the Hadamard Test estimates

    <psi|U|psi>.

The real and imaginary components can be measured independently with one
ancilla qubit. This is useful for variational algorithms, overlap/observable
estimation, matrix-element routines, and other hybrid quantum-classical
workflows.

## Quick start

    from pyqpanda_alg import HadamardTest
    from pyqpanda3.core import QCircuit, X, Z

    def prepare_one(qubits):
        circuit = QCircuit()
        circuit << X(qubits[0])
        return circuit

    def z_unitary(qubits):
        circuit = QCircuit()
        circuit << Z(qubits[0])
        return circuit

    result = HadamardTest.run_hadamard_test(
        z_unitary,
        target_width=1,
        prepare_state=prepare_one,
        component="real",
    )

    print(result.value)  # ideal value: -1

Because Z|1> = -|1>, the exact expectation is -1.

## API layers

Pure helpers do not import PyQPanda3:

- reference_probabilities(expectation, component)
- decode_component(probabilities)
- combine_components(real, imag)

Runtime helpers import PyQPanda3 lazily:

- hadamard_test_circuit(...)
- run_hadamard_test(...)
- run_complex_hadamard_test(...)

The circuit builder adds no measurement operations, so callers can compose the
Hadamard Test into a larger QProg. target_width may be greater than one; the
unitary and preparation factories receive the complete target register.

## Measurement convention

For z = <psi|U|psi>:

    real: P(0) = (1 + Re(z)) / 2
    imag: P(0) = (1 + Im(z)) / 2

The real circuit uses H on the ancilla after controlled-U. The imaginary
circuit uses RX(-pi/2), which rotates the ancilla Y observable onto the Z
measurement axis.

The decoded component is P(0)-P(1).

## Failure behavior

Invalid component names, non-finite probability weights, negative weights,
zero-mass outcome vectors, non-unitary reference expectations with magnitude
above one, invalid target widths, overlapping ancilla/target registers, and
invalid circuit factories fail explicitly instead of silently returning an
expectation value.

## Scope

This is a standalone algorithm primitive. It does not modify QAE, QPE, QAOA,
Grover, QARM, QSVR, QKMeans, Simon, QWalk, QEC, or SPSA behavior.
