# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public Quantum Fourier Transform helpers.

The core package already carries the QFT circuit builders in
pyqpanda_alg.plugin for use by algorithms such as amplitude estimation.
This module promotes those primitives to a stable, discoverable public API and
adds the inverse transform and deterministic resource accounting.

Qubit order follows the existing package convention: the input list is ordered
from least-significant qubit to most-significant qubit. with_swaps=True returns
the complete textbook QFT with output-order correction; setting it to False
returns only the Hadamard/controlled-phase network.
"""

from pyqpanda3.core import QCircuit

from ..plugin import QFT as _full_qft
from ..plugin import qft as _phase_qft


def _normalize_qubits(qubit_list):
    """Return a validated list of distinct, non-negative qubit indices."""
    if isinstance(qubit_list, int):
        qubits = [qubit_list]
    elif isinstance(qubit_list, (list, tuple)):
        qubits = list(qubit_list)
    else:
        raise TypeError("qubit_list must be an int, list[int], or tuple[int, ...]")

    if not qubits:
        raise ValueError("qubit_list must not be empty")

    for index, qubit in enumerate(qubits):
        if not isinstance(qubit, int):
            raise TypeError(
                f"qubit_list[{index}] must be an integer, not {type(qubit)}"
            )
        if qubit < 0:
            raise ValueError(
                f"qubit_list[{index}] must be non-negative, got {qubit}"
            )

    if len(set(qubits)) != len(qubits):
        raise ValueError("qubit_list must contain distinct qubit indices")

    return qubits


def qft(qubit_list, with_swaps: bool = True) -> QCircuit:
    """Build a Quantum Fourier Transform circuit.

    Args:
        qubit_list: One qubit index or an ordered sequence of distinct qubit
            indices, least-significant qubit first.
        with_swaps: Include the final bit-reversal SWAP network when True.
            Set to False when the caller manages output order externally.

    Returns:
        A pyqpanda3.core.QCircuit implementing the requested QFT.
    """
    if not isinstance(with_swaps, bool):
        raise TypeError("with_swaps must be a bool")

    qubits = _normalize_qubits(qubit_list)
    return _full_qft(qubits) if with_swaps else _phase_qft(qubits)


def inverse_qft(qubit_list, with_swaps: bool = True) -> QCircuit:
    """Build the inverse Quantum Fourier Transform circuit."""
    return qft(qubit_list, with_swaps=with_swaps).dagger()


def qft_resource_counts(qubit_count: int, with_swaps: bool = True) -> dict:
    """Return exact gate-family counts for the package QFT construction."""
    if not isinstance(qubit_count, int):
        raise TypeError("qubit_count must be an integer")
    if qubit_count < 1:
        raise ValueError("qubit_count must be at least 1")
    if not isinstance(with_swaps, bool):
        raise TypeError("with_swaps must be a bool")

    return {
        "hadamard": qubit_count,
        "controlled_phase": qubit_count * (qubit_count - 1) // 2,
        "swap": qubit_count // 2 if with_swaps else 0,
    }


QFT = qft
IQFT = inverse_qft

__all__ = [
    "qft",
    "inverse_qft",
    "qft_resource_counts",
    "QFT",
    "IQFT",
]
