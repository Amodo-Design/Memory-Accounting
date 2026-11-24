"""Memory probe for the vLLM prover: every CUDA storage in the worker, named by the object that holds it."""
import gc

import torch
from memory_accounting.prover.vllm_ext import PosmWorkerExtension

SKIP = (str, bytes, int, float, bool, type(None), type, torch.device, torch.dtype)


def _walk(obj, path, out, seen, depth):
    if isinstance(obj, SKIP) or depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            out.append((path, obj))
        return
    if isinstance(obj, torch.nn.Module):
        for n, p in list(obj.named_parameters()) + list(obj.named_buffers()):
            out.append((path + ".<params>", p))
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:4096]:
            _walk(v, path, out, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in list(obj)[:4096]:
            _walk(v, path, out, seen, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and type(obj).__module__.startswith("memory_accounting."):
        for k, v in list(d.items()):
            _walk(v, f"{path}.{k}" if path.count(".") < 3 else path, out, seen, depth + 1)


class MeasureExt(PosmWorkerExtension):
    def posm_mem_reset(self) -> str:
        dev = next(self._model().parameters()).device
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        return "reset"

    def posm_mem_probe(self, label: str = "") -> dict:
        dev = next(self._model().parameters()).device
        torch.cuda.synchronize(dev)
        gc.collect()
        free, total = torch.cuda.mem_get_info(dev)
        alloc, resv = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        roots = [
            ("vLLM KV pool tensors (runner.kv_caches)", getattr(getattr(self, "model_runner", None), "kv_caches", None)),
            ("theta", getattr(self.posm_sd, "theta", None) if self.posm_sd else None),
            ("filler", self.posm_filler),
            ("dyn", self.posm_vdyn),
            ("sd", self.posm_sd),
            ("resp", self.posm_resp),
            ("model", self._model()),
        ]
        owner, seen_obj = {}, set()
        for name, root in roots:
            found = []
            _walk(root, name, found, seen_obj, 0)
            for path, t in found:
                st = t.untyped_storage()
                owner.setdefault(st.data_ptr(), (path, st.nbytes(), tuple(t.shape), str(t.dtype)))
        others = {}
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor) and o.is_cuda:
                    st = o.untyped_storage()
                    if st.data_ptr() and st.data_ptr() not in owner:
                        others.setdefault(st.data_ptr(), (st.nbytes(), tuple(o.shape), str(o.dtype)))
            except Exception:
                pass
        groups = {}
        for path, nb, shape, dt in owner.values():
            g = groups.setdefault(path, [0, 0, None])
            g[0] += nb
            g[1] += 1
            if g[2] is None or nb > g[2][0]:
                g[2] = (nb, shape, dt)
        top_other = sorted(others.values(), reverse=True)[:20]
        return {
            "label": label,
            "max_num_seqs": getattr(getattr(getattr(self, "vllm_config", None), "scheduler_config", None), "max_num_seqs", None),
            "total": total,
            "device_free": free,
            "encoded_weight_bytes": self.posm_sd.theta.numel() * 8 if self.posm_sd else 0,
            "static_filler_bytes": self.posm_filler.words * 8 if self.posm_filler else 0,
            "dynamic_pool_bytes": self.posm_vdyn.region.nbytes if self.posm_vdyn else 0,
            "cuda_allocator_retries": torch.cuda.memory_stats(dev).get("num_alloc_retries", 0),
            "cuda_allocator_ooms": torch.cuda.memory_stats(dev).get("num_ooms", 0),
            "used": total - free,
            "torch_allocated": alloc,
            "torch_reserved": resv,
            "torch_peak_allocated": torch.cuda.max_memory_allocated(dev),
            "torch_peak_reserved": torch.cuda.max_memory_reserved(dev),
            "vllm_profiled_peak": int(getattr(self, "peak_activation_memory", 0) or 0),
            "groups": {k: {"bytes": v[0], "storages": v[1], "largest": v[2]} for k, v in groups.items()},
            "other_bytes": sum(v[0] for v in others.values()),
            "other_storages": len(others),
            "other_top": top_other,
            "hbm_line": self.posm_hbm() if (self.posm_sd or self.posm_vdyn) else None,
        }
