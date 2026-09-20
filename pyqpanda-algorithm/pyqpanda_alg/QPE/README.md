# Quantum Phase Estimation (QPE)

This package adds a standalone Quantum Phase Estimation implementation to
pyqpanda-algorithm.

QPE estimates the eigenphase phi of a unitary U on an eigenstate |psi>:

    U |psi> = exp(2*pi*i*phi) |psi>,  0 <= phi < 1.

The implementation is split into two layers so algorithm behavior is useful
even when a PyQPanda3 runtime is not present:

1. Pure reference helpers compute the exact ideal finite-register QPE
   probability distribution, decode the maximum-likelihood grid point, and
   compare phases on the unit circle.
2. The PyQPanda3 layer builds the standard controlled-power circuit and runs it
   on CPUQVM. Imports are lazy, so importing the pure helpers does not require
   PyQPanda3.

## Public API

    from pyqpanda_alg import QPE

    probabilities = QPE.qpe_reference_probabilities(
        phase=0.375,
        precision_bits=4,
    )
    estimate = QPE.decode_phase(probabilities, 4)

For a runtime circuit, provide a factory returning the unitary QCircuit and,
optionally, a factory preparing its eigenstate:

    from pyqpanda_alg import QPE
    from pyqpanda3.core import QCircuit, X, Z

    def prepare_one(qubits):
        circuit = QCircuit()
        circuit << X(qubits[0])
        return circuit

    def z_unitary(qubits):
        circuit = QCircuit()
        circuit << Z(qubits[0])
        return circuit

    result = QPE.run_qpe(
        z_unitary,
        target_width=1,
        precision_bits=3,
        prepare_eigenstate=prepare_one,
    )
    print(result.phase)      # 0.5
    print(result.bitstring)  # counting-register maximum

Z|1> = -|1> = exp(2*pi*i*0.5)|1>, so this example has an exactly
representable phase.

## Conventions

- Phases are fractions of one full turn in the half-open interval [0, 1).
- Counting qubit i controls U raised to 2**i.
- The inverse QFT and returned bit-string integer convention follow the same
  ordering used by the existing pyqpanda_alg.QAE implementation.
- The circuit builder adds no measurement operations, so callers can compose
  it into larger programs.
- target_width can exceed one; the unitary factory receives the complete target
  register and may return any compatible QCircuit.
- precision_bits is capped at 24 in the pure helpers to avoid accidental
  exponential allocation of a probability vector.

## Reference distribution

For N = 2**m and integer output y, QPE assigns

    P(y) = |(1/N) * sum_{k=0}^{N-1}
                    exp(2*pi*i*k*(phi-y/N))|**2.

The implementation evaluates the corresponding Dirichlet-kernel expression,
including the removable singularity at exact grid points. The returned vector
is normalized with math.fsum.

## Failure behavior

Invalid precision, non-finite phases/probabilities, negative probability mass,
wrong distribution widths, empty runtime mappings, overlapping registers, and
invalid circuit factories fail with ValueError instead of silently producing a
phase estimate.

## Scope

This contribution is a general algorithm primitive. It does not change QAE,
Grover, Simon, SPSA, QKMeans, QARM, QSVR, QWalk, or QEC behavior.
