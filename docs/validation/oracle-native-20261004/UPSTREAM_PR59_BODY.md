关联 #13（2026 本源杯开源创新赛道｜代码贡献）。

Adds `pyqpanda_alg.QOracleLearning`: Deutsch-Jozsa constant-vs-balanced classification and Bernstein-Vazirani recovery of q0-first affine secrets `f(x)=s·x XOR b`, with validated truth-table/bit-order contracts, generic phase-oracle synthesis, an O(n) affine BV oracle, deterministic probability-list decoding, an independent Walsh-Hadamard reference, documentation, an example and focused regression coverage.

## Source

The original contribution remains frozen at `b04476c0e4486d8b6e3bc503965d8fd1dac9192d`, against `develop` at `5f973efccb84bc193157d1ccebe32137e307293b`. The PR was opened September 15, 2026. This update adds execution evidence for those unchanged source bytes; it does not replace the submitted implementation.

## Native execution — October 4, 2026

All **9 maintained tests passed, with zero failures, errors or skips**, including both PyQPanda3 CPUQVM end-to-end cases, using PyQPanda3 0.5.0 and Python 3.11.16 on Ubuntu 24.04.5. The focused suite completed in 1.480 seconds. The public example also ran: BV recovered `1011`, peak probability was `1.0000000000000004` within floating-point roundoff, Deutsch-Jozsa classified the example as `balanced`, and P(0...0) was 0.0.

[Completed run](https://github.com/woahwhattheheck/pyqpanda-algorithm/actions/runs/37200720587) · [Retained source-bound results and reproduction instructions](https://github.com/woahwhattheheck/pyqpanda-algorithm/tree/440210a67e84ed29bb156fa77251b2c8e69aa386/docs/validation/oracle-native-20261004)

The normal package imports and maintained pytest configuration were used. The runner installed the additional pandas/scikit-learn dependencies and Allure reporting plugin required by that configuration; the product and tests were not modified. Earlier collection failures are documented separately and are not counted as passing tests. The full repository suite and real quantum hardware were not exercised.

## Scope

Public bit strings are q0-first; truth-table q0 is the least-significant bit. Generic explicit-table phase synthesis is pedagogical worst-case O(2^n); the affine BV oracle is O(n), and affine bias is a global phase. PyQPanda3 imports in the oracle module are lazy. Simon and QARM are outside scope. No merge, reward or payment claim.