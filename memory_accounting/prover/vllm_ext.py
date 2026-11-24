"""vLLM worker extension: run the model on Θ' (stream-decode) and answer PoSM challenges in-process.

Usage (vLLM >= 0.8 style worker extensions):

    llm = LLM(model=..., worker_extension_cls="memory_accounting.prover.vllm_ext.PosmWorkerExtension", enforce_eager=True)
    llm.collective_rpc("posm_convert", args=(enc_key_hex, filler_seed_hex, scratch_fraction))
    llm.collective_rpc("posm_serve", args=(port,))
    info = llm.collective_rpc("posm_info")

The extension methods run inside each worker process with `self.model_runner.model` available.
Only tensor-parallel size 1 is supported here (one model copy per GPU, the doc's stage-1 setting).

Dynamic memory (KV pages of the running batch + the paused layer's activations, bit-exact replay on
the same GPU type with batch-invariant kernels) is `posm_dynamic_setup` on the prover and the
`posm_replay_*` methods on the verifier; see `memory_accounting.models.vllm` for the settings both sides need.
"""

from __future__ import annotations

import torch


class PosmWorkerExtension:
    posm_sd = None
    posm_resp = None
    posm_filler = None
    posm_vdyn = None
    posm_capture = None
    posm_gpu = 0
    posm_filler_seed = None  # the verifier's filler seed, once posm_convert has it (the hash keys derive from it)

    def posm_set_cyclic_gc(self, enabled: bool) -> bool:
        """Diagnostic only: toggle cyclic GC between completed inference batches."""
        import gc

        if enabled:
            gc.enable()
        else:
            gc.disable()
        return gc.isenabled()

    def posm_set_native_wait(self, enabled: bool) -> bool:
        """Diagnostic only: compare the Python and native completion-flag waits."""
        from . import strategies

        strategies._NATIVE_WAIT = bool(enabled)
        return strategies._NATIVE_WAIT

    def _model(self):
        mr = getattr(self, "model_runner", None)
        model = getattr(mr, "model", None) if mr is not None else None
        if model is None:
            raise RuntimeError("worker has no model_runner.model")
        return model

    def posm_export_weight_reference(self, path: str) -> dict:
        """Export plaintext before conversion on a trusted model-loading host."""
        from ..setup.stream_reference import export_reference

        if self.posm_sd is not None:
            raise RuntimeError("export the trusted reference before conversion")
        return export_reference(self._model(), path)

    def posm_convert(
        self,
        enc_key_hex: str,
        filler_seed_hex: str = "0",
        scratch_fraction: float = 0.10,
        layer_re: str = r"layers\.(\d+)\.",
        module_re: str = r"layers\.(\d+)$",
        filler_max_bytes: int | None = None,
        reserve_bytes: int = 0,
        gpu: int | None = None,
    ) -> dict:
        """Encode the resident parameters into Θ', install per-layer decode hooks, fill free HBM minus the
        scratch allowance with the verifier-seeded filler (encoded under the key derived from the filler seed and
        this GPU's index, `memory_accounting.setup.filler.static_filler_key`, as the verifier derives it). gpu: this GPU's
        index as the verifier knows it (default the worker's device index, which is 0 on every GPU with one
        engine per GPU); `posm_export` names the bundle's GPU entry after it. Returns the segment layout."""
        from .filler import allocate_filler
        from .stream_decode import convert

        model = self._model()
        dev = next(model.parameters()).device
        if reserve_bytes < 0:  # auto: vLLM's own profiled activation peak, plus our buffers made after the filler
            peak = int(getattr(self, "peak_activation_memory", 0) or 0)
            reserve_bytes = peak + (1 << 30) if peak else 4 << 30
            print(
                f"posm: scratch reserve {reserve_bytes / (1 << 30):.2f} GiB (vLLM's profiled activation peak {peak / (1 << 30):.2f} GiB + 1 GiB)",
                flush=True,
            )
        torch.cuda.synchronize(dev)
        self.posm_sd = convert(model, int(enc_key_hex, 16), layer_re=layer_re, module_re=module_re)
        torch.cuda.synchronize(dev)
        if getattr(self.posm_sd, "convert_peak_bytes", None):
            print(
                f"posm: converted; at most {self.posm_sd.convert_peak_bytes / (1 << 30):.2f} GiB of the GPU in use while converting",
                flush=True,
            )
        self.posm_gpu = int(gpu) if gpu is not None else (dev.index or 0)
        self.posm_filler_seed = int(filler_seed_hex, 16)
        self.posm_filler = allocate_filler(
            int(filler_seed_hex, 16),
            int(enc_key_hex, 16),
            self.posm_sd.payload_words,
            scratch_fraction,
            dev,
            max_bytes=filler_max_bytes,
            reserve_bytes=reserve_bytes,
            gpu=self.posm_gpu,
        )
        return self.posm_info()

    def _posm_keys(self, seed_hex: str | None = None):
        """The challenge-hash keys, derived from the verifier's filler seed (`Keys.from_seed`): the seed given here,
        else the one `posm_convert` filled with, else (dynamic memory only) the keys `posm_dynamic_setup` derived."""
        from ..core.params import Keys

        if seed_hex is not None:
            return Keys.from_seed(int(seed_hex, 16))
        if self.posm_filler_seed is not None:
            return Keys.from_seed(self.posm_filler_seed)
        if self.posm_vdyn is not None:
            return self.posm_vdyn.dyn.keys
        raise RuntimeError("no filler seed to derive the hash keys from: call posm_convert or posm_dynamic_setup first")

    def posm_serve(
        self,
        port: int = 5555,
        filler_seed_hex: str | None = None,
        strategies: str = "honest",
        canon_device: str | None = None,
        drain: bool = True,
        hold_ms: float = 100.0,
        partial_fraction: float = 0.10,
        lora_fraction: float = 0.05,
        evict_fraction: float = 0.10,
        tamper_fraction: float = 0.10,
        verifier_addrs: list[str] | None = None,
        dyn_evict_fraction: float = 0.5,
        capture_evidence: bool = True,
        compact_share_fraction: float | None = None,
        dynamic_timing_shares: bool = False,
    ) -> str:
        """Start the responder. With drain (default) the worker's step methods run under the responder's
        gate, so an armed challenge waits for the step in flight and then runs on an idle GPU.
        filler_seed_hex: the verifier's filler seed, which the challenge-hash keys derive from (default: the one
        `posm_convert` was given; this used to be a hash key of the prover's choosing)."""
        from .responder import ResidentArray, Responder

        if self.posm_sd is None and self.posm_vdyn is None:
            raise RuntimeError("call posm_convert (static) and/or posm_dynamic_setup (dynamic) first")
        dev = str(next(self._model().parameters()).device)
        strats = tuple(s for s in strategies.split(",") if s)
        resident = None
        if self.posm_sd is not None:
            segs = self.posm_sd.segments() + (self.posm_filler.segments() if self.posm_filler else [])
            share_span = None
            if compact_share_fraction is not None:
                import hashlib

                block_words = (1 << 20) // 8
                count = sum(words for _, words, _ in segs) // block_words
                chosen = max(1, int(count * compact_share_fraction))
                weight_blocks = self.posm_sd.theta.numel() // block_words
                if chosen >= weight_blocks:
                    raise ValueError("compact share does not fit in the encoded weights")
                seed = int(self.posm_filler_seed)
                selection = int.from_bytes(
                    hashlib.blake2b(seed.to_bytes(8, "little"), digest_size=8, person=b"memory_accounting/share").digest(),
                    "little",
                )
                start = (1 + selection % (weight_blocks - chosen)) * block_words
                share_span = start, start + chosen * block_words
                print(f"posm compact 1% weight share: blocks [{start // block_words}, {start // block_words + chosen}), "
                      f"{chosen} of {count} static blocks", flush=True)
            resident = ResidentArray(
                segs,
                self.posm_sd.enc_key,
                dev,
                strategies=strats,
                canon_device=canon_device,
                segment_keys=(self.posm_sd.enc_key, self.posm_filler.key)
                if self.posm_filler and self.posm_filler.words else (self.posm_sd.enc_key,),
                partial_fraction=partial_fraction,
                lora_fraction=lora_fraction,
                evict_fraction=evict_fraction,
                tamper_fraction=tamper_fraction,
                share_span=share_span,
            )
        keys = self._posm_keys(filler_seed_hex)
        if self.posm_vdyn is not None and keys.as_list() != self.posm_vdyn.dyn.keys.as_list():
            raise RuntimeError(
                "posm_serve and posm_dynamic_setup derived different hash keys: pass posm_dynamic_setup the filler seed "
                "(filler_seed_hex) when the pad seed is not the filler seed"
            )
        if self.posm_vdyn is not None and not capture_evidence:
            if self.posm_vdyn.enc is None or "dyn_dram" in strats:
                raise ValueError("timing-only dynamic response requires an encoded pool in the honest posture")
            dynamic = self.posm_vdyn.dyn
            dynamic.capture_evidence = False
            dynamic.stage_fn = lambda: {"in_flight": self.posm_vdyn.in_flight}
        from .responder import watch_gc

        self.posm_gc = watch_gc()  # worker GC after setup; expose counters to the throughput recorder
        self.posm_resp = Responder(
            resident,
            keys,
            port=port,
            strategies=strats,
            drain=drain,
            hold_ms=hold_ms,
            dynamic=self.posm_vdyn.dyn if self.posm_vdyn is not None else None,
            verifier_addrs=verifier_addrs,
            dyn_evict_fraction=dyn_evict_fraction,
            dynamic_timing_shares=dynamic_timing_shares,
        )
        if drain:
            resp = self.posm_resp
            for name in ("execute_model", "sample_tokens", "execute_dummy_batch"):  # the executor calls these by name
                orig = getattr(self, name, None)
                if orig is None:
                    continue

                def gated(*a, _orig=orig, **kw):
                    return resp.step(_orig, *a, **kw)

                setattr(self, name, gated)
        self.posm_resp.start()
        static = (
            f"{resident.words * 8 >> 20} MiB static in {resident.nseg} segments"
            if resident is not None
            else "no static array"
        )
        return (
            f"posm responder on :{port}, {static}"
            + (", dynamic region over the running batch" if self.posm_vdyn is not None else "")
            + (", drain window" if drain else "")
        )

    def posm_export(
        self, out_dir: str, filler_seed_hex: str = "0", gpu: int | None = None,
        benchmark_only: bool = False, include_dynamic_size: bool = False,
    ) -> str:
        """Write the verifier's bundle (one-way image, commitment, manifest without the weights' key) for this
        worker's GPU; gpu: its index as the verifier knows it (default: the one `posm_convert` keyed the filler for;
        another index is refused, since the filler is encoded under that GPU's key). The manifest's hash keys
        are the ones derived from the filler seed (this took a hash key before)."""
        return self.posm_sd.export_bundle(
            out_dir,
            filler=self.posm_filler,
            filler_seed=int(filler_seed_hex, 16),
            gpu=self.posm_gpu if gpu is None else gpu,
            benchmark_only=benchmark_only,
            benchmark_dynamic_bytes=self.posm_vdyn.region.nbytes if include_dynamic_size else None,
        )

    def posm_dynamic_setup(
        self,
        block: str = "64K",
        pad_seed_hex: str = "0",
        include_bytes: bool = False,
        filler_seed_hex: str | None = None,
        gpu: int | None = None,
        encoded: bool = True,
    ) -> dict:
        """Prover: the dynamic region over the whole KV pool and the paused layer's activations, pause points
        at every layer and step boundary, free and unwritten slots holding filler. Call before posm_serve, and before
        posm_convert when encoded: the encoded pool takes about 1/8 of the pool more memory, which the static filler
        must leave. gpu: this GPU's index as the verifier knows it (keys the dynamic filler; default the worker's
        device index, which is 0 on every GPU with one engine per GPU). encoded (v0.17, the default): the pool and the
        activation rows held encoded (`memory_accounting.prover.kvenc`), attention on a plaintext scratch of one layer.
        filler_seed_hex: the verifier's filler seed, which the challenge-hash keys derive from (default: the pad
        seed, which is the filler seed unless the launcher was given another)."""
        from ..core.params import parse_bytes
        from .vllm_serve import VllmDynamic

        keys = self._posm_keys(filler_seed_hex or pad_seed_hex)
        self.posm_vdyn = VllmDynamic(
            self, keys, parse_bytes(block), int(pad_seed_hex, 16), include_bytes, gpu=gpu, encoded=encoded
        )
        return dict(self.posm_vdyn.info)

    def posm_replay_setup(self) -> dict:
        """Verifier: install the capture hooks; returns the engine facts the records are checked against."""
        import torch

        from ..verifier.vllm_replay import ReplayCapture

        self.posm_capture = ReplayCapture(self)
        kv = self.posm_capture.kv[0]
        vc = self.vllm_config
        return {
            "block_size": kv.shape[2],
            "page_shape": list(kv.shape[1:]),
            "page_stride": list(kv[0].stride()),
            "dtype": str(vc.model_config.dtype).replace("torch.", ""),
            "kv_dtype": str(kv.dtype).replace("torch.", ""),
            "gpu_name": torch.cuda.get_device_name(kv.device),
            "kv_layout": str(getattr(vc.cache_config, "kv_cache_layout", None)),
        }

    def posm_replay_arm(self, num_computed: int, scheduled: int, act_layer: int, in_flight: bool, nblocks: int) -> bool:
        self.posm_capture.arm(num_computed, scheduled, act_layer, in_flight, nblocks)
        return True

    def posm_replay_take(self) -> dict:
        return self.posm_capture.take()

    def posm_stop(self) -> str:
        if self.posm_resp:
            self.posm_resp.stop()
            self.posm_resp.join(timeout=2)
        return "stopped"

    def posm_enable_timing_trace(self) -> bool:
        """Opt in to per-static-response timing breakdowns for diagnostics."""
        if self.posm_resp is None:
            raise RuntimeError("start the responder before enabling timing trace")
        self.posm_resp.trace_enabled = True
        return True

    def posm_take_timing_trace(self) -> list[dict]:
        return self.posm_resp.take_timing_trace() if self.posm_resp else []

    def posm_info(self) -> dict:
        sd = self.posm_sd
        if sd is None:
            return {
                "converted": False,
                "served": self.posm_resp.served if self.posm_resp else 0,
                "static_served": self.posm_resp.static_served if self.posm_resp else 0,
                "dynamic": (
                    {
                        "served": self.posm_vdyn.dyn.served,
                        "mid_step": self.posm_vdyn.dyn.paused_in_flight,
                        "between_steps": self.posm_vdyn.dyn.paused_between,
                        "idle": self.posm_vdyn.dyn.idle,
                        "held_off": self.posm_vdyn.pause.held_off,
                    }
                    if self.posm_vdyn
                    else None
                ),
            }
        return {
            "converted": True,
            "payload_words": sd.payload_words,
            "theta_bytes": sd.theta.numel() * 8,
            "scratch_bytes": sd.scratch.numel(),
            "resident_bytes": sd.plain_bytes - sd.scratch.numel(),
            "units": len(sd.units),
            "filler_words": self.posm_filler.words if self.posm_filler else 0,
            "served": self.posm_resp.served if self.posm_resp else 0,
            "static_served": self.posm_resp.static_served if self.posm_resp else 0,
            "armed": self.posm_resp.armed if self.posm_resp else 0,
            "hold_expired": self.posm_resp.hold_expired if self.posm_resp else 0,
            "posture": self.posm_resp.posture if self.posm_resp else None,
            "dynamic": (
                {
                    "served": self.posm_vdyn.dyn.served,
                    "mid_step": self.posm_vdyn.dyn.paused_in_flight,
                    "between_steps": self.posm_vdyn.dyn.paused_between,
                    "idle": self.posm_vdyn.dyn.idle,
                    "held_off": self.posm_vdyn.pause.held_off,
                    "static_freezes": self.posm_vdyn.dyn.static_freezes,
                    "slow_freezes": self.posm_vdyn.dyn.slow_freezes,
                    "timing_live_arms": self.posm_vdyn.dyn.timing_live_arms,
                    "timing_live_pages": self.posm_vdyn.dyn.timing_live_pages,
                    "timing_live_arms_by_posture": self.posm_vdyn.dyn.timing_live_arms_by_posture,
                    "timing_write_refreshes": self.posm_vdyn.dyn.timing_write_refreshes,
                    "freed_blocks": self.posm_vdyn.pool.freed_blocks,
                    "free_ms_max": round(self.posm_vdyn.pool.free_s_max * 1e3, 2),
                }
                if self.posm_vdyn
                else None
            ),
            "hbm": self.posm_hbm(),
        }

    def posm_diagnostics(self) -> dict:
        """Lightweight counters sampled around throughput calls, outside their timed interval."""
        gc_state = getattr(self, "posm_gc", None)
        gc_info = ({"collections": gc_state["collections"],
                    "gen2_collections": gc_state["gen2_collections"],
                    "objects_collected": gc_state["objects_collected"],
                    "uncollectable_objects": gc_state["uncollectable_objects"],
                    "slow_collections": gc_state["slow"],
                    "total_ms": gc_state["total_ns"] / 1e6,
                    "slow_ms": gc_state["slow_ns"] / 1e6,
                    "max_ms": gc_state["max_ns"] / 1e6}
                   if gc_state is not None else None)
        evidence_store = (self.posm_vdyn.dyn.store.stats() if self.posm_vdyn and self.posm_resp else None)
        return {"gc": gc_info, "evidence_store": evidence_store}

    def posm_hbm(self) -> str:
        """Memory as the prover sees it: static (Θ' + filler) and dynamic (KV pool + activations) challenged, the rest not."""
        from .filler import hbm_line

        static = (
            (self.posm_sd.theta.numel() + (self.posm_filler.words if self.posm_filler else 0)) * 8
            if self.posm_sd
            else 0
        )
        dyn = self.posm_vdyn.region.nbytes if self.posm_vdyn else 0
        dev = next(self._model().parameters()).device
        return hbm_line(dev, static, dyn, self.posm_vdyn.dyn.free_min if self.posm_vdyn else None)
