"""pytest leaves as tests/_runner.py does: with os._exit, after its report. Some tests leave daemon prover threads
running (on a GPU host, with CUDA work in flight); interpreter teardown under them aborts in C++ ("terminate called
without an active exception", exit 134) after every test has passed."""

import os
import sys

import pytest

_status = None


def pytest_sessionfinish(session, exitstatus):
    global _status
    _status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if _status is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_status)
