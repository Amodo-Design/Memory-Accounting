"""The main of every test file: run its test_ functions (or only those named on the command line), each whatever
failed before it, print a traceback for each failure and a summary, and exit 1 if any failed. pytest runs the same
files (README, Test); this runner needs nothing installed.

    uv run python tests/test_net.py                                   # every test in the file
    uv run python tests/test_ui.py test_page_live_run_late_alerts_and_unchecked test_token

A test that cannot run here (no CUDA, no node) calls skip(reason): pytest and this runner report it skipped, not
passed. A test that skips part way (its CPU checks done, the GPU's not) is skipped too: it did not all run.

Some tests leave daemon prover threads running (on a GPU host, with CUDA work in flight); interpreter teardown under
them aborts in C++ after every test has passed, so the runner leaves with os._exit, after flushing."""

import os
import sys
import time
import traceback
import unittest


def skip(reason: str):
    __tracebackhide__ = True  # pytest's report then names the test's line, not this one
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def run(ns: dict) -> None:
    tests = {n: f for n, f in ns.items() if n.startswith("test_") and callable(f)}
    names = sys.argv[1:] or list(tests)
    unknown = [n for n in names if n not in tests]
    if unknown:
        print("no such test:", ", ".join(unknown), file=sys.stderr)
        _exit(2)
    failed, skipped = [], []
    for name in names:
        start = time.monotonic()
        try:
            tests[name]()
        except KeyboardInterrupt:
            print("interrupted in", name, file=sys.stderr)
            _exit(130)
        except unittest.SkipTest as e:
            print(f"skip {name}: {e}")
            skipped.append(name)
        except BaseException:  # an assertion, an error, a SystemExit: the next test still runs
            sys.stdout.flush()
            traceback.print_exc()
            sys.stderr.flush()
            print(f"FAIL {name} ({time.monotonic() - start:.1f} s)")
            failed.append(name)
        else:
            print("ok", name)
    if skipped:
        print(f"{len(skipped)} of {len(names)} skipped: {', '.join(skipped)}")
    if failed:
        print(f"{len(failed)} of {len(names)} failed: {', '.join(failed)}")
    _exit(1 if failed else 0)
