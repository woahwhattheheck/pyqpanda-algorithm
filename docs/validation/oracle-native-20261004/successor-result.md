# Fast-Walsh successor: completed native execution

Source `440210a67e84ed29bb156fa77251b2c8e69aa386` passed all **11 focused maintained tests**, with **0 failures, 0 errors and 0 skips**, on October 4, 2026. The original nine cases and both new fast-reference regressions executed through the normal package import and pytest configuration. Both real CPUQVM cases and the public example passed. The focused suite took 1.447 seconds; this is not a runtime benchmark.

[Completed run 37201125301](https://github.com/woahwhattheheck/pyqpanda-algorithm/actions/runs/37201125301) · [JUnit](successor-tests.xml) · [Raw benchmark samples](../../../benchmarks/results/oracle-walsh-runner-20261004.json)

The workflow checked that the product and benchmark bytes exactly matched the recorded source hashes before execution. The source archive and complete logs are retained in artifact `11302774522`, ZIP SHA-256 `63169ee5bbceee2a9b4f4919feda842e0fb2dd734278d2f0e9240aba66aed4eb`, with 14-day retention. This document, JUnit and raw benchmark JSON retain the compact result independently of that expiry.

## Second-host benchmark

Python 3.11.16 on an Intel Xeon 6973P-C GitHub-hosted Linux runner independently repeated the same source-checked, three-repeat benchmark against original `b04476c`. All timed outputs and all 276 exhaustive small Boolean tables matched exactly.

| Entries | Original median | Successor median | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 3.757 ms | 0.108 ms | 34.75x |
| 1,024 | 62.269 ms | 0.518 ms | 120.29x |
| 4,096 | 1,005.993 ms | 2.456 ms | 409.58x |

The first host's results remain separately recorded, not replaced with whichever values are fastest. These are single-process classical reference measurements, not simulator or end-to-end application speedups. The circuit helpers are unchanged. No full repository suite or real quantum hardware execution is claimed.

## Publication boundary

This successor targets the separate fork integration branch and does not alter the frozen original sponsor PR59. Native evidence for the original is recorded in [README.md](README.md). A single attempt to update the original PR's stale execution description and unnecessary agent bylines returned HTTP 403 `Resource not accessible by integration`; that metadata edit was **not applied** by this session. [The exact prepared replacement](UPSTREAM_PR59_BODY.md) is available for an already-authorized publisher, who should re-read PR59 before applying it. No duplicate sponsor/Gitee submission, award or payment is claimed.
