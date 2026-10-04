# QOracleLearning

`QOracleLearning` adds two textbook oracle-learning algorithms to `pyqpanda-algorithm`:

- **Deutsch-Jozsa**: one quantum oracle query distinguishes a promised constant Boolean function from a balanced one.
- **Bernstein-Vazirani**: one quantum oracle query recovers the complete hidden affine bit string `s` in `f(x)=s·x XOR b`.

## Bit-order contract

All public secret strings are **q0-first**. `"101"` means `q0=1`, `q1=0`, `q2=1`. Truth-table index `x` uses ordinary integer bits, so q0 is the least-significant bit. Execution uses `get_prob_list()` and decodes the winning integer basis index explicitly, avoiding display-string endianness ambiguity.

## Oracle construction

`truth_table_phase_oracle()` accepts an explicit truth table and directly synthesizes the diagonal phase oracle `(-1)^f(x)`. This generic teaching path is worst-case `O(2^n)` gates and is not claimed to be scalable.

`affine_phase_oracle()` uses the structure of Bernstein-Vazirani: each `s_i=1` contributes one Z gate, so the oracle is `O(n)`. The affine bias is a global phase and is intentionally omitted.

## Example

```python
from pyqpanda_alg import QOracleLearning

secret, probs = QOracleLearning.run_bernstein_vazirani("1011")
assert secret == "1011"

oracle = QOracleLearning.affine_truth_table("101")
kind, probs = QOracleLearning.run_deutsch_jozsa(oracle)
assert kind == "balanced"
```

## Independent correctness reference

`walsh_probabilities()` computes the exact classical Walsh-Hadamard spectrum. It is deliberately independent of PyQPanda3 and is used to test every hidden string through five qubits under both affine biases. The test suite also covers promise rejection, non-affine rejection, bit-order round trips, phase-oracle signs, and CPUQVM end-to-end execution.

The module imports PyQPanda3 lazily: pure truth-table / reference helpers remain usable for validation and documentation tooling without initializing a quantum backend.


## Fast classical reference

`walsh_probabilities()` evaluates the complete spectrum using integer
Walsh-Hadamard butterflies in O(N log N) time and O(N) workspace, where N is
the truth-table length. The signed sums and basis-index order are unchanged;
normalization and squaring occur only after the integer transform. This is a
faster classical reference, not a quantum-simulator or quantum-advantage claim.
The circuit constructors, public tuple format and input validation are unchanged.

A three-repeat, alternating-order comparison against frozen source
`b04476c0e4486d8b6e3bc503965d8fd1dac9192d` on Python 3.13.5 / Linux x86_64
(AMD EPYC 9V74) measured these medians:

| Truth-table entries | Direct reference | Fast reference | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 4.617 ms | 0.118 ms | 39.10x |
| 1,024 | 82.668 ms | 0.690 ms | 119.83x |
| 4,096 | 1,444.574 ms | 3.416 ms | 422.89x |

All timed outputs were exactly equal. The benchmark additionally compares all
276 Boolean truth tables of lengths 2, 4 and 8 against the original direct
implementation. These are single-host measurements on deterministic generated
inputs, not universal speed guarantees or end-to-end application timings.

From the repository root, with the original commit available locally:

```sh
python benchmarks/benchmark_oracle_walsh.py --output walsh-benchmark.json
```

For a source archive without Git history, pass `--baseline-file` pointing to
the original `oracle_learning.py`; the script verifies its exact Git blob.
Raw samples, source identities, input hashes and environment are retained in
[`benchmarks/results/oracle-walsh-20261004.json`](../../../benchmarks/results/oracle-walsh-20261004.json).
The maintained `Test_walsh_reference.py` independently uses direct signed sums
and checks input preservation and rejected inputs.
