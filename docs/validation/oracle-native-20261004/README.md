# Native acceptance of the frozen OracleLearning contribution

On October 4, 2026, the unchanged source of existing OriginQ PR 59 at
`b04476c0e4486d8b6e3bc503965d8fd1dac9192d` passed all **9 maintained tests**,
with **0 failures, 0 errors and 0 skips**, in the normal package import and
pytest configuration. Both CPUQVM end-to-end cases executed using PyQPanda3
0.5.0, Python 3.11.16 and Ubuntu 24.04.5. The focused suite took 1.480 seconds;
this is a test-suite duration, not a performance benchmark.

The separate public example recovered the BV secret `1011`, reported peak
probability `1.0000000000000004` (floating-point roundoff), classified the
Deutsch-Jozsa example as `balanced`, and returned P(0...0) = 0.0.

- [Existing contribution](https://github.com/OriginQ/pyqpanda-algorithm/pull/59)
- [Completed native run](https://github.com/woahwhattheheck/pyqpanda-algorithm/actions/runs/37200720587)
- [Exact isolated controller](https://github.com/woahwhattheheck/pyqpanda-algorithm/blob/ee8c6b3ea75a56de83a77255c100ba9d60ae3779/.github/workflows/oracle-native-acceptance.yml)
- [Per-case JUnit results](tests.xml), [environment](environment.txt),
  [example output](example.log), [checked-out source identity](source-sha.txt).

The downloaded artifact is `11303055284`, SHA-256
`ad85a8801a126e30f158a0b9885746ba5c89931c771bd6ee5c7942633650300d`. It contains
the complete source archive, test and install logs, source tree and file-hash
manifest. The compact raw results linked above are retained here independently
of the artifact's 14-day retention.

## Reproduction

Run from the frozen source checkout:

```sh
python -m pip install -r pyqpanda-algorithm/requirements.txt \
  'pyqpanda3==0.5.0' 'pytest==9.1.1' scikit-learn pandas allure-pytest
PYTHONPATH=pyqpanda-algorithm python -m pytest -q \
  test/QOracleLearning/Test_oracle_learning.py --junitxml=tests.xml
PYTHONPATH=pyqpanda-algorithm python \
  pyqpanda-algorithm/example/QAlgBase/testeg_QOracleLearning.py
```

The package's eager imports require scikit-learn and pandas; its pytest.ini
requires the Allure reporting plugin. These runner dependencies were added
without changing the frozen source, replacing its imports or suppressing tests.
The first run stopped before collection on the missing Allure plugin
(run `37200211980`, artifact `11301997987`, ZIP SHA-256
`86b15b3b94b5f78e5d8625c1cd24ee5c95d15634f4597c1fc703f627304aef8f`).
The second stopped during collection on missing pandas
(run `37200439879`, artifact `11302985060`, ZIP SHA-256
`eb0e81c350c2afc41ffc724c6f8bb0f37774a04a8f830ea211f2d6563328d1e5`).
Neither failed attempt is counted as executing or passing the suite.

## Scope

This result applies to the frozen original, not by itself to later successors.
The fast-Walsh successor has separate measured source/output evidence. No full
repository suite, real quantum hardware, sponsor acceptance, award or payment
is claimed. The April 10–September 30 entry window has ended; this is
follow-through on the September 15 entry, not a new competition submission.
The original frozen contribution branch and its source attribution are unchanged.
