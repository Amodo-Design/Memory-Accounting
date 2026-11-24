"""Run the real live scheduler and check queues with an in-memory datagram transport."""
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from memory_accounting.core import wire
from memory_accounting.core.benchmark import read_verifier_handoff
from memory_accounting.core.params import Keys
from memory_accounting.setup.commit import leaves_of, merkle_root
from memory_accounting.setup.manifest import BENCHMARK, BUNDLE, GpuEntry, Manifest
from memory_accounting.verifier import net


def make_manifest(folder, checked):
    source = folder / "manifest.json"
    words, block = 512, 1024
    commit_key = bytes(range(16))
    gpu = GpuEntry(0, None, "", "", "")
    if checked:
        image = np.zeros(words, dtype=np.uint64)
        gpu.image_file = str(folder / "reference.npy")
        np.save(gpu.image_file, image)
        root = merkle_root(leaves_of(image, block, commit_key)).hex()
        gpu.image_root = gpu.commit_root = root
    Manifest("fixture", 0, "", [], block, 0 if checked else words, 77, words, words * 8, commit_key.hex() if checked else "",
             Keys.from_seed(77).as_list(), gpus=[gpu], kind=BUNDLE if checked else BENCHMARK, r=0, benchmark_dynamic_bytes=2304).save(source)
    return source


def exercise(checked, folder, dynamic_fraction=1.0):
    manifest = make_manifest(folder, checked)
    ready = folder / "settings.json"
    log = folder / "events.jsonl"
    seen = []
    reply = None

    def transmit(sock, args, packet):
        nonlocal reply
        magic = int.from_bytes(packet[:8], "little")
        seen.append(magic)
        if magic in (wire.ARM_MAGIC, wire.ARM_DYN_MAGIC):
            _, token, block, *_ = wire.ARM.unpack(packet)
            if magic == wire.ARM_DYN_MAGIC:
                reply = wire.READY_DYN.pack(magic, token, 288, block, 1, 0)
            else:
                reply = wire.READY.pack(magic, token)
        else:
            _, nonce, *_ = wire.CH.unpack(packet)
            reply = wire.RE.pack(nonce, 42, 43)  # deliberately wrong
        return True

    def receive(sock, args, match, *rest):
        assert match(reply)
        return reply

    with ExitStack() as stack:
        for target, value in (
            ("memory_accounting.verifier.ui.server.start", MagicMock()), ("memory_accounting.verifier.net.socket.socket", MagicMock()),
            ("memory_accounting.verifier.net._prover_stats", MagicMock(return_value=None)),
            ("memory_accounting.verifier.net._send", transmit), ("memory_accounting.verifier.net._await", receive),
        ):
            stack.enter_context(patch(target, value))
        dynamic = stack.enter_context(patch.object(net, "_check_dynamic", return_value={"digest_ok": True, "replayed": True}))
        net.main(["live", "--host", "127.0.0.1", "--manifest", str(manifest), "--filler-seed", "4d",
                  "--benchmark-ready-file", str(ready), "--out", str(log), "--verify-threads", "1",
                  "--allowed-B", "1K", "--min-r", "0", "--n", "1", "--dyn-n", "1", "--dyn-B", "1K",
                  "--dynamic", "--round-mix", "pool-bytes", "--drain", "--warmup", "1", "--calibrate", "4", "--interval", "0.001",
                  "--dynamic-check-fraction", str(dynamic_fraction),
                  "--strategies", "honest", "--max-verifications", "6", "--timing", "user",
                  *([] if checked else ["--skip-response-checks"])])
        assert bool(dynamic.call_count) == (checked and dynamic_fraction == 1.0)
    assert {wire.ARM_MAGIC, wire.MAGIC, wire.ARM_DYN_MAGIC, wire.DYN_MAGIC}.issubset(seen)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    rounds = [e for e in events if e["type"] == "verification"]
    assert {e["round"] for e in rounds} == {"static", "dynamic"}
    checks = [e for e in events if e["type"] == "check"]
    if checked:
        assert any(e.get("digest_ok") is False for e in checks)
        if dynamic_fraction < 1.0:
            assert all(e.get("check_sampled_out") for e in rounds if e["round"] == "dynamic")
            assert all(not e.get("check_sampled_out") for e in rounds if e["round"] == "static")
    else:
        assert not checks and all(e["response_checks_skipped"] for e in rounds)
        assert not any(folder.glob("*.npy"))
        try:
            Manifest.load(manifest)
        except ValueError as exc:
            assert "no reference image" in str(exc)
        else:
            raise AssertionError("checked verifier accepted timing-only layout")
    config = read_verifier_handoff(ready, manifest, not checked)
    assert config["options"]["calibrate"] == 4 and config["response_checks"] == checked
    assert config["static_shape"]["links"] == 1 and config["dynamic_shape"][0] == 1024
    schedule = config["round_schedule"]
    assert (schedule["static_bytes"], schedule["dynamic_bytes"]) == (4096, 2048)
    assert schedule["static_round_probability"] == 2 / 3
    assert schedule["dynamic_challenge_fraction"] == 1 / 3
    assert schedule["aggregate_wait_mean_s"] == .0005
    assert "calibration_samples" in config["baseline"]
    manifest.write_text(manifest.read_text() + "\n")
    try:
        read_verifier_handoff(ready, manifest, not checked)
    except ValueError as exc:
        assert "different exported manifest" in str(exc)
    else:
        raise AssertionError("accepted a stale handoff")


def test_both_throughput_modes():
    for checked in (True, False):
        with tempfile.TemporaryDirectory() as temp:
            exercise(checked, Path(temp))


def test_dynamic_sampling_leaves_static_checks_enabled():
    with tempfile.TemporaryDirectory() as temp:
        exercise(True, Path(temp), dynamic_fraction=1e-12)


def test_benchmark_prover_measures_both_modes():
    """Run the normal benchmark CLI with a small inference stand-in; the real engine needs an H200."""
    import sys
    from types import SimpleNamespace

    from scripts import bench_throughput as bench
    from memory_accounting.core.benchmark import file_digest

    def one(checked, folder):
        manifest = make_manifest(folder, checked)
        go = folder / "go"
        output = folder / "result.json"
        generation = []
        worker = MagicMock()

        def generate(requests, sampling, use_tqdm):
            if isinstance(requests[0], str):
                tokens = [1]
            else:
                generation.append(requests)
                tokens = [1] * sampling.max_tokens
                if not go.exists():
                    go.write_text(json.dumps({"schema": "posm-throughput-verifier-v1",
                                              "manifest_sha256": file_digest(manifest), "response_checks": checked}))
            return [SimpleNamespace(outputs=[SimpleNamespace(text="same", token_ids=tokens)]) for _ in requests]

        def rpc(method, args=(), kwargs=None):
            if method == "posm_export":
                assert kwargs["benchmark_only"] == (not checked)
                return [str(manifest)]
            if method == "posm_dynamic_setup":
                assert kwargs["encoded"]
            if method == "posm_info":
                return [{"served": 2 * len(generation), "static_served": len(generation), "dynamic": {"served": len(generation)}}]
            return [{}]

        worker.generate.side_effect = generate
        worker.collective_rpc.side_effect = rpc
        with ExitStack() as stack:
            stack.enter_context(patch.object(bench, "build_llm", return_value=worker))
            stack.enter_context(patch.object(bench, "prover_runtime", return_value={}))
            stack.enter_context(patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SimpleNamespace)}))
            stack.enter_context(patch.object(sys, "argv", ["bench", "full", "--batches", "3", "--prompt-len", "5",
                                                            "--gen-len", "4", "--calls", "1", "--out", str(output),
                                                            "--go-file", str(go), *([] if checked else ["--skip-response-checks"])]))
            bench.main()
        data = json.loads(output.read_text())
        assert data["posm"]["output_matches"]
        assert data["batches"][0]["calls"][0]["tokens"] == 12
        assert data["batches"][0]["served"] == {"static": 1, "dynamic": 1}
        assert data["reproduction"]["verifier"]["response_checks"] == checked
        state = data["reproduction"]["generation"]["rng_before_first_batch"]
        rng = np.random.default_rng()
        rng.bit_generator.state = state
        assert bench.requests(rng, 3, 5) == generation[-2]
        assert bench.requests(rng, 3, 5) == generation[-1]
        return generation[-1]

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        assert one(True, Path(a)) == one(False, Path(b))


def test_breakdown_stages_are_cumulative():
    import sys
    from types import SimpleNamespace
    from scripts import bench_full, bench_throughput

    args = SimpleNamespace(enc_key="123", filler_seed="abc", dyn_block="1M", plain_dynamic=False,
                           bundle="bundle", port=5558, verifier_addr=["localhost"], skip_response_checks=False)
    expected = {
        "settings": [],
        "encoded": ["posm_convert", "posm_hbm"],
        "dynamic": ["posm_dynamic_setup", "posm_convert", "posm_hbm"],
        "full": ["posm_dynamic_setup", "posm_convert", "posm_export", "posm_serve", "posm_hbm"],
    }
    assert [stage for stage, _ in bench_full.BREAKDOWN_STAGES] == list(expected)
    assert "--dynamic" in bench_full.VERIFIER_DEFAULTS and "--with-dyn-live" not in bench_full.VERIFIER_DEFAULTS
    with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=SimpleNamespace)}):
        for stage, methods in expected.items():
            worker = MagicMock()
            worker.generate.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="same")])]
            worker.collective_rpc.return_value = [{}]
            bench_throughput.setup_posm(worker, stage, args)
            calls = worker.collective_rpc.call_args_list
            assert [call.args[0] for call in calls] == methods
            if "posm_dynamic_setup" in methods:
                assert calls[0].kwargs["kwargs"]["encoded"]

def test_pool_mix_counts_individual_challenges():
    # S:D = 8:3. With two static chains per round, round weights must be 8:6.
    weights = net._pool_round_weights(8192, 3500, 1024, 2)
    assert weights == (8192, 6144)
    for draw, kind in [(0, "static"), (8191, "static"), (8192, "dynamic"), (14335, "dynamic")]:
        with patch.object(net._SYSRAND, "randrange", return_value=draw) as rand:
            assert net._pick_pool_round(weights) == kind
            rand.assert_called_once_with(14336)
    for size in (None, 0, 128):
        try:
            net._pool_round_weights(8192, size, 1024, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a missing or unchallengeable dynamic pool")
