# TabICL diagnostic status

The Colab bundle contains two successful TabICL runs and 142 failures, all
`IndexError: pop from empty list`. The median failed call lasted 0.010 seconds;
the longest lasted 5.892 seconds. Only the first season's first seed's baseline
and Elo direct regressions succeeded. This pattern does not identify a cause.
No traceback, actual device, or PyTorch version was saved by the original runner.

Local validation on 2026-10-26:

- TabICL 2.2.0, Python 3.12.7, PyTorch 2.14.1, macOS CPU.
- Four small repeated synthetic regression fits passed.
- Twelve actual-data calls at 256 training rows and 32 test rows passed.
- Twelve actual-data calls at 5,000 training rows and 64 test rows passed:
  two test seasons, three feature specifications, direct and two-stage prediction,
  four ensemble estimators. The fitted devices were recorded as CPU. Model
  call time totaled 112.97 seconds.
- Cached regressor: `tabicl-regressor-v2-20260212.ckpt`, 114,324,594 bytes.
- Local weight SHA256:
  `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`.

These checks do not reproduce the original Linux/Python 3.13.16 Colab
environment, its unknown PyTorch version, GPU execution, full prediction batch,
or interleaving with TabPFN. They therefore do not establish that the bug is fixed.
No speculative dependency change or numerical workaround has been applied.

`tabicl_diagnostic.py --include-tabpfn --device cuda` provides a bounded
reproduction in the Colab environment: 5,000 training rows, 64 test rows,
24 calls alternating TabPFN and TabICL. It records environment and actual fitted
devices, cached weight hash, failure stage, and a token-redacted traceback;
it stops at the first failure. See the README for the exact Colab command.
If it passes, the next controlled check is to increase the prediction batch
to the original full season before retrying all failed configurations.

The main runner now saves per-run runtime/device information and redacted
tracebacks. New bundle reports label incomplete forecast coverage. Existing
successful checkpoints remain usable; older metadata cannot prove GPU use.

Validation: 13 unit tests passed, including matched-row rejection, seed-aware
interval units, incomplete-coverage labels and token redaction. The updated
wrapper completed a six-cell Ridge smoke run and created its results ZIP.
