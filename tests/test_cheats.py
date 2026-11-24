"""Cheats that change HBM (evict, evict_dram, tamper), on a GPU: what each posture leaves in memory, what
each answers, and that the honest posture puts everything back (skipped without CUDA). On any machine:
the prover-side bookkeeping around them (which strategies are pre-armed, a timed-out disarm, buffers for
g > 1, strategies that cannot be built, the drain gate)."""

import pathlib
import sys
from typing import ClassVar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from _runner import skip

KEY = 0x0123456789ABCDEF0123456789ABCDEF
SEED = 0x5EED5EED5EED5EED


def _setup():
    from memory_accounting.core.backend import DeviceBackend
    from memory_accounting.prover.filler import FillerRegion
    from memory_accounting.prover.responder import ResidentArray
    from memory_accounting.prover.stream_decode import convert
    from tests.test_stream_decode import Toy

    torch.manual_seed(1)
    model = Toy(d=512, n=4).cuda()
    sd = convert(model, enc_key=KEY)  # Θ' on the GPU
    be = DeviceBackend("cuda")
    fw = 24 * (1 << 17)  # 24 MiB of filler after Θ'
    buf = torch.empty(fw, dtype=torch.int64, device="cuda")
    tmp = torch.empty_like(buf)
    be.filler(tmp, SEED, sd.payload_words)
    be.setup_at(tmp, buf, KEY, sd.payload_words)
    be.sync()
    fill = FillerRegion(buf, fw)
    ra = ResidentArray(
        sd.segments() + fill.segments(),
        KEY,
        "cuda",
        strategies=("honest", "evict", "evict_dram", "tamper"),
        evict_fraction=0.25,
        tamper_fraction=0.5,
    )
    return sd, fill, ra


def test_postures_change_hbm_and_honest_restores_it():
    if not torch.cuda.is_available():
        skip("no CUDA")
    from memory_accounting.core.params import Challenge, Keys
    from memory_accounting.prover.strategies import EvictDramProver, HonestProver

    sd, fill, ra = _setup()
    keys = Keys.deterministic(1)
    theta0, fill0 = sd.theta.clone(), fill.tensor.clone()
    honest, dram = HonestProver(ra, keys), EvictDramProver(ra, keys)
    nb = ra.num_blocks(1 << 20)
    chs = [Challenge(1 << 20, 8, 256, 1000 + i, i % nb, 1024) for i in range(6)]
    ref = [honest.respond(c)[:2] for c in chs]

    ra.set_posture("evict")  # the tail of the filler now holds other data
    e = ra.evict_words
    assert (
        not torch.equal(fill.tensor[-e:], fill0[-e:])
        and torch.equal(fill.tensor[:-e], fill0[:-e])
        and torch.equal(sd.theta, theta0)
    )
    assert [dram.respond(c)[:2] for c in chs] == ref  # evict_dram reads the kept filler from host memory: still right
    wrong = sum(honest.respond(c)[:2] != r for c, r in zip(chs, ref))
    assert wrong > 0, "no chain touched the evicted quarter"  # answering from what is there: wrong digests

    ra.set_posture("honest")
    assert torch.equal(fill.tensor, fill0) and torch.equal(sd.theta, theta0)
    assert [honest.respond(c)[:2] for c in chs] == ref

    ra.set_posture("tamper")  # weights changed in place, filler untouched
    assert torch.equal(fill.tensor, fill0) and not torch.equal(sd.theta, theta0)
    bw = (1 << 20) // 8
    changed = [
        j
        for j in range(sd.theta.numel() // bw)
        if not torch.equal(sd.theta[j * bw : (j + 1) * bw], theta0[j * bw : (j + 1) * bw])
    ]
    assert changed == ra.tamper_blocks
    from memory_accounting.core.backend import DeviceBackend

    be = DeviceBackend("cuda")
    g = bw // 16
    a = torch.empty(15 * g, dtype=torch.int64, device="cuda")
    b = torch.empty_like(a)
    j = changed[0]
    be.decode_unpack(sd.theta, a, j * bw, g, KEY)
    be.decode_unpack(theta0, b, j * bw, g, KEY)
    be.sync()
    diff = a.view(torch.int32) ^ b.view(torch.int32)
    assert bool((diff == 1).all()), "tamper must flip exactly the lowest bit of every 32-bit element"
    ra.set_posture("honest")
    assert torch.equal(sd.theta, theta0) and torch.equal(fill.tensor, fill0)


def test_share_cheats_answer_like_honest():
    """Every cheat on a share of the array (ShareCheat), answered by the honest instance, gives honest's digests
    and trace, pre-armed or not: the share read as the cheat reads it (encode_otf, other_gpu, also over the whole
    array), or the chain stopped before the share's links and the host answering them (dram_copy, cpu), with
    stops at the first link and at several links of one chain among them."""
    if not torch.cuda.is_available():
        skip("no CUDA")
    from memory_accounting.core.backend import DeviceBackend
    from memory_accounting.core.memory import Hbm
    from memory_accounting.core.params import Challenge, Keys, MemoryConfig
    from memory_accounting.prover.strategies import CpuProver, HonestProver, ShareCheat

    be = DeviceBackend("cuda")
    for image in (False, True):
        hbm = Hbm(be, MemoryConfig(words=(64 << 20) // 8, image=image))
        keys = Keys.deterministic(2)
        honest, cpu = HonestProver(hbm, keys), CpuProver(hbm, keys, threads=4)
        nb = hbm.num_blocks(1 << 20)
        chs = [Challenge(1 << 20, 8, 256, 7000 + i, (7 * i) % nb, 1024) for i in range(12)]
        ref = [honest.respond(c)[:3] for c in chs]
        try:
            for cheat in ShareCheat.KINDS:
                for f in (0.25, 0.5) + ((1.0,) if cheat in ("encode_otf", "other_gpu") else ()):
                    honest.share = sh = ShareCheat(cheat, hbm, f, cpu=cpu)
                    assert [honest.respond(c)[:3] for c in chs] == ref, (cheat, f, image)
                    for c, r in zip(chs, ref):
                        honest.arm(c)
                        assert honest.fire(c)[:3] == r, (cheat, f, image, "armed")
                    if sh.stops:
                        tail = [sum(j >= sh.jstop(c) for j in r[2]) for c, r in zip(chs, ref)]
                        assert max(tail) >= 2 and any(c.j0 >= sh.jstop(c) for c in chs), (
                            "no chain stopped twice, or at its first link"
                        )
            honest.share = None
            assert [honest.respond(c)[:3] for c in chs] == ref
        finally:
            honest.share = None
            cpu.close()


def test_every_device_chain_is_prearmed():
    """Honest and the cheats that run the same device chain are all pre-armed at ARM, so none pays a CUDA
    launch and a sync inside the timed window that the other does not. dram_copy (a host round trip per
    link) and cpu (no device) have no one chain to pre-launch."""
    from memory_accounting.core.params import Challenge
    from memory_accounting.prover.responder import Responder
    from memory_accounting.prover.strategies import PROVERS

    armed = []
    r = Responder.__new__(Responder)
    r.prearm = True
    r.provers = {}
    for name, cls in PROVERS.items():
        p = cls.__new__(cls)
        p.arm = lambda shape, name=name: armed.append(name)
        r.provers[name] = p
    shape = Challenge(1 << 16, 4, 0, 0, 0, 1024, 1)
    for name in PROVERS:
        r._prearm(name, shape)
    assert set(armed) == set(PROVERS) - {"dram_copy", "cpu"}, armed


def _bare_device_prover(cls=None):
    from memory_accounting.prover.strategies import DeviceProver

    p = (cls or DeviceProver).__new__(cls or DeviceProver)
    p.name, p.kernel = "honest", "tensor"
    p._armed_shape = (1 << 16, 4, 0, 1024, 1)
    p._flag_np, p._done_np = np.zeros(1, np.int64), np.zeros(1, np.int64)
    return p


def test_timed_out_disarm_fails_and_blocks_the_next_arm():
    """A graph that does not finish within the disarm timeout may still set done and write its output later;
    a next arm must not clear done and fire into it. The disarm raises; arm waits for the old done flag."""
    from memory_accounting.core.params import Challenge

    prover = _bare_device_prover()
    try:
        prover.disarm(timeout_s=0.01)
        raise AssertionError("a timed-out disarm passed silently")
    except RuntimeError as e:
        assert "timed out" in str(e)
    assert prover._armed_shape is None and prover._stale and prover._flag_np[0] == 2
    try:
        prover.arm(Challenge(1 << 16, 4, 0, 0, 0, 1024, 1))
        raise AssertionError("armed over a graph that had not finished")
    except RuntimeError as e:
        assert "not finished" in str(e)
    prover._done_np[0] = 1  # the old graph returns at last
    prover._clear_stale()
    assert not prover._stale


def test_device_prover_buffers_for_g_above_one():
    """The chain kernels need groups x g partial sums; the prover sized them for g = 1, so every g > 1 was
    refused by the extension ("partials buffer needs groups * g words")."""
    from types import SimpleNamespace

    from memory_accounting.core.params import Challenge

    prover = _bare_device_prover()

    def z(n, dt=torch.int64):
        return torch.zeros(max(n, 1), dtype=dt)

    prover.hbm = SimpleNamespace(
        backend=SimpleNamespace(
            power_buffer=z, matrix_buffer=lambda n: z(n, torch.int8), block_buffer=lambda n: z(n, torch.int8)
        )
    )
    prover.dev, prover.groups, prover.max_groups = "cpu", 6, 6
    prover.partials = torch.zeros(6, dtype=torch.int64)
    prover.pw = prover.mbuf = prover.ablk = None
    for g in (1, 4, 2):
        prover._ensure_buffers(Challenge(1 << 16, 4, 256, 0, 0, 1024, g))
        assert prover.partials.numel() == 6 * g, (g, prover.partials.numel())


def test_strategies_without_their_tables_are_not_built():
    """On a plain array (the standalone prover's Hbm) partial, evict_dram, evict and tamper have nothing to
    run over: refused at construction, so the prover skips them and does not report them as served (they
    would otherwise answer as honest under a cheat's name, or fail at every challenge)."""
    from types import SimpleNamespace

    from memory_accounting.core.params import Keys
    from memory_accounting.prover.strategies import PROVERS

    hbm = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), theta_prime=None, lora=None, partial_nseg=0, evict_nseg=0
    )
    for name in ("partial", "evict_dram", "evict", "tamper", "lora"):
        try:
            PROVERS[name](hbm, Keys.deterministic(1), groups=4)
            raise AssertionError(f"{name} built without its tables")
        except RuntimeError:
            pass


def test_drain_gate_released_when_a_forward_raises():
    import threading

    from memory_accounting.prover.serve import gate_forward

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise ValueError("boom")

    gate = threading.Lock()
    m = Boom()
    gate_forward(m, gate)
    try:
        m(torch.zeros(1))
        raise AssertionError("no exception")
    except ValueError:
        pass
    assert not gate.locked(), "a failed forward left the drain gate held: the responder would wait forever"
    m2 = Boom()
    # fails before the gate is taken
    m2.register_forward_pre_hook(lambda mod, a: (_ for _ in ()).throw(KeyError("pre")))
    gate_forward(m2, gate)
    gate.acquire()  # the responder holds it
    try:
        m2(torch.zeros(1))
    except KeyError:
        pass
    assert gate.locked(), "a forward that never took the gate released the responder's hold"
    gate.release()


def test_responder_stops_on_a_port_in_use():
    """A port another process holds stops the prover: Responder.start raises for the challenge port (UDP) and for
    the records' TCP port of the same number, and gives the UDP port back if only the TCP one failed; the provers'
    startup check refuses either before any setup. Before, the bind failed on the responder's thread and the
    prover went on generating, answering nothing (a stray UDP echo server held a demo host's port for 24 days)."""
    import socket
    from types import SimpleNamespace

    from memory_accounting.core.evidence import EvidenceServer, EvidenceStore
    from memory_accounting.core.params import Keys
    from memory_accounting.prover.responder import Responder, require_free_ports

    fake = SimpleNamespace(
        region=SimpleNamespace(device="cpu"),
        frozen=False,
        armed=False,
        serve_evidence=lambda bind, port: EvidenceServer(EvidenceStore(), bind, port).start(),
        close=lambda: None,
        release=lambda: None,
    )
    for kind, name in ((socket.SOCK_DGRAM, "UDP"), (socket.SOCK_STREAM, "TCP")):
        holder = socket.socket(socket.AF_INET, kind)
        holder.bind(("127.0.0.1", 0))
        if kind == socket.SOCK_STREAM:
            holder.listen()
        port = holder.getsockname()[1]
        try:
            r = Responder(None, Keys.deterministic(1), bind="127.0.0.1", port=port, dynamic=fake)
            try:
                r.start()
                raise AssertionError(f"started with the {name} port held")
            except OSError as e:
                assert f"{name} port {port}" in str(e), e
            assert not r.is_alive()
            try:
                require_free_ports(port, "127.0.0.1")
                raise AssertionError(f"the startup check passed with the {name} port held")
            except SystemExit as e:
                assert f"{name} port {port}" in str(e), e
        finally:
            holder.close()
        if name == "TCP":  # the UDP port the failed start had bound is free again
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", port))
            s.close()
            require_free_ports(port, "127.0.0.1", tcp=False)


def test_responder_answers_only_the_verifier():
    """Responder(verifier_addrs=...): datagrams from any other address are dropped unread (no ARM, posture,
    counters or answer); None accepts any sender. Run here on the CPU with a stand-in dynamic region."""
    import socket
    import time
    from types import SimpleNamespace

    from memory_accounting.core.params import Keys
    from memory_accounting.prover.responder import Responder
    from memory_accounting.verifier import net

    fake = SimpleNamespace(
        region=SimpleNamespace(device="cpu"),
        frozen=False,
        armed=False,
        serve_evidence=lambda *a: None,
        close=lambda: None,
        release=lambda: None,
    )
    real = torch.cuda.set_device, torch.cuda.current_device
    torch.cuda.set_device, torch.cuda.current_device = (lambda *a: None), (lambda: 0)  # the responder thread on the CPU
    try:
        for allowed, answered in (("192.0.2.7", False), ("localhost", True), (None, True)):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            r = Responder(
                None, Keys.deterministic(1), bind="127.0.0.1", port=port, dynamic=fake, verifier_addrs=allowed
            )
            r.start()
            time.sleep(0.2)
            c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                got = net._prover_stats(c, SimpleNamespace(host="127.0.0.1", port=port), 0.5)
            finally:
                c.close()
                r.stop()
                r.join(2.0)
            assert (got is not None) == answered and r.dropped == (0 if answered else 1), (allowed, got, r.dropped)
    finally:
        torch.cuda.set_device, torch.cuda.current_device = real
    try:
        Responder(None, Keys.deterministic(1), dynamic=fake, verifier_addrs="no-such-host.invalid")
        raise AssertionError("accepted")
    except ValueError:
        pass


def test_responder_survives_a_failed_send_and_always_releases():
    """An OSError from sendto (EHOSTUNREACH, ENETUNREACH, EPERM from a firewall) loses that answer only: the
    responder releases what it holds and serves the next datagram. Whatever else ends its loop, it releases the
    frozen region on the way out. Before: the OSError ended the thread with the region frozen, and inference
    waited for it forever. Its in-process timings are bounded (a prover serves for days)."""
    import socket as S
    import time
    from types import SimpleNamespace

    import memory_accounting.prover.responder as R
    from memory_accounting.core import wire
    from memory_accounting.core.params import Keys
    from memory_accounting.verifier import net

    released = []

    def arm(*a, **k):
        raise TypeError("an unexpected failure in the freeze")

    fake = SimpleNamespace(
        region=SimpleNamespace(device="cpu"),
        frozen=False,
        armed=False,
        serve_evidence=lambda *a: None,
        close=lambda: None,
        release=lambda: released.append(1),
        arm=arm,
    )

    class FlakySock(S.socket):
        fails: ClassVar[list] = [1]  # one failed send, shared by every instance

        def sendto(self, *a):
            if FlakySock.fails:
                FlakySock.fails.pop()
                raise OSError(65, "No route to host")
            return super().sendto(*a)

    real = torch.cuda.set_device, torch.cuda.current_device, R.socket
    torch.cuda.set_device, torch.cuda.current_device = (lambda *a: None), (lambda: 0)
    R.socket = SimpleNamespace(socket=FlakySock, AF_INET=S.AF_INET, SOCK_DGRAM=S.SOCK_DGRAM, timeout=S.timeout)
    try:
        s = S.socket(S.AF_INET, S.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        r = R.Responder(None, Keys.deterministic(1), bind="127.0.0.1", port=port, dynamic=fake)
        assert r.svc_ns.maxlen == r.dt_ns.maxlen == r.gap_ns.maxlen == r.prof["gpu_ns"].maxlen == R.PROFILE_KEEP
        r.start()
        time.sleep(0.2)
        client = S.socket(S.AF_INET, S.SOCK_DGRAM)
        try:
            args = SimpleNamespace(host="127.0.0.1", port=port)
            assert net._prover_stats(client, args, 0.5) is None  # its answer could not be sent
            assert released and r.is_alive()
            assert net._prover_stats(client, args, 1.0) is not None, "the responder stopped after a failed send"
            n = len(released)
            client.sendto(wire.ARM.pack(wire.ARM_DYN_MAGIC, 7, 1 << 16, 4, 0, 1024, 1, 0), ("127.0.0.1", port))
            r.join(2.0)
            # before the freeze, and on the way out
            assert not r.is_alive() and len(released) >= n + 2, len(released) - n
        finally:
            client.close()
            r.stop()
    finally:
        torch.cuda.set_device, torch.cuda.current_device, R.socket = real


if __name__ == "__main__":
    from _runner import run

    run(globals())
