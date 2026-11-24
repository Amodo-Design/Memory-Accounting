"""Independent big-integer vectors and CPU/CUDA agreement for packed rows."""
import sys
import numpy as np
from memory_accounting.core.backend import CpuLib
from memory_accounting.core.codec_profile import PROFILE
from memory_accounting.core.reference import siphash24_128
from memory_accounting.prover.kvenc import encode_np, decode_np, KV_POS, enc_words, word_groups
from memory_accounting.setup.pack import pack89

assert PROFILE=="p89_element_packed128"
key=(0xFEDCBA9876543210<<64) | 0x123456789ABCDEFA
q0=16*13
p=(1<<89)-1
def reference(src,q):
    blob=src.tobytes();result=bytearray()
    for j in range((len(blob)+120)//121):
        bits=0
        for k in range(11):
            idx=(q//16+j)*11+k
            lo,hi=siphash24_128(key&((1<<64)-1),key>>64,idx,0x5457454B00000001,0,0)
            tweak=((hi<<64|lo)&p)%p
            start=j*121+k*11
            x=(int.from_bytes(blob[start:start+11],'little')+tweak)%p
            root=pow(x,1<<87,p)
            bits|=(root | (int(root*root%p!=x)<<89)) << (90*k)
        result.extend(bits.to_bytes(128,'little'))
    return bytes(result)
raw=np.random.default_rng(111).integers(0,256,968*2,dtype=np.uint8).view(np.uint64)
lib=CpuLib();payload=pack89(raw.view(np.uint8));enc=np.empty_like(payload)
lib.setup_at(payload,enc,key,q0); assert enc.tobytes()==reference(raw,q0)
out=np.zeros_like(raw)
# Weight streaming takes the full allocation and a global offset into it.
theta=np.concatenate([np.zeros(q0,dtype=np.uint64),enc]);lib.decode_unpack(theta,out,q0,2,key);assert np.array_equal(out,raw)
short=raw[:31];dyn=encode_np(short,q0,key);assert dyn.tobytes()==reference(short,KV_POS|q0);assert np.array_equal(short,decode_np(dyn,q0,key,len(short)))
assert enc_words(8192)*8==69376
from memory_accounting.prover.kvenc import EncStore, empty_np
zeros = np.zeros(121, dtype=np.uint64)
expected_zero_words = np.frombuffer(reference(zeros, KV_POS | q0), dtype=np.uint64)
for first, count in ((3, 3), (10, 12), (15, 1), (16, 27)):
    assert np.array_equal(empty_np(count, q0 + first, key), expected_zero_words[first:first+count])
for n in (1, 2, 3, 8, 21):
    st = EncStore(None, 8192).prefix(n)
    assert st.unit_words == n * 16
if '--cuda' in sys.argv:
    import torch
    from memory_accounting.core.backend import DeviceBackend
    from memory_accounting.prover.kvenc import Codec
    gpu=DeviceBackend("cuda")
    r=torch.from_numpy(raw.view(np.int64)).cuda(); payload_gpu=torch.empty(len(payload),dtype=torch.int64,device='cuda')
    gpu.pack(r,payload_gpu,2); assert payload_gpu.cpu().numpy().tobytes()==payload.tobytes()
    theta_gpu=torch.zeros(len(theta),dtype=torch.int64,device='cuda');gpu.setup_at(payload_gpu,theta_gpu[q0:],key,q0)
    assert theta_gpu[q0:].cpu().numpy().tobytes()==enc.tobytes()
    decoded=torch.zeros_like(r);gpu.decode_unpack(theta_gpu,decoded,q0,2,key);assert torch.equal(decoded,r)
    codec=Codec(key, device="cuda"); source=r[:31].clone(); dest=torch.zeros(len(dyn),dtype=torch.int64,device='cuda')
    codec.encode_range(source,dest,q0,0,31);assert dest.cpu().numpy().tobytes()==dyn.tobytes()
    restored=torch.zeros_like(source);codec.decode_range(dest,q0,restored,0,31);assert torch.equal(source,restored)
    source[15]^=777; codec.encode_range(source,dest,q0,15,16)
    assert dest.cpu().numpy().tobytes()==reference(source.cpu().numpy(),KV_POS|q0)
print('PASS',PROFILE,sys.argv[1:], 'static vectors, position-specific keys, 34 zero bits, KV tail and page size')
