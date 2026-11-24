// Image-hash microbenchmark: the one-way image of 32-byte groups on the GPU, the previous SipHash-2-4-128 x 2
// construction against the BLAKE3 keyed single compression now in posm_core.h (image_group).
//
//   mkdir -p memory_accounting/build && nvcc -O3 -arch=sm_90 -I memory_accounting/kernels -o memory_accounting/build/bench_image scripts/bench_image_cuda.cu
//   memory_accounting/build/bench_image [log2 groups, default 26] [images per thread in the ALU test, default 64]
//
// Two tests per function:
//   mem: one thread per group, read 4 words, image, write 4 words (what posm_image_kernel and the per-link
//        block fetch do; 64 bytes of traffic per group, so it is compute-bound unless the hash is very cheap)
//   alu: each thread images its own group `iters` times in registers (input = previous image), no memory:
//        the pure instruction cost.
// It first checks the device's image_group against the host's (the same header, compiled for both).
// SASS instructions of one image (each mem kernel is one image plus a few loads, stores and the index):
//   for f in mem_siphash mem_blake3; do echo $f $(cuobjdump -sass -fun $f memory_accounting/build/bench_image | grep -cE '^ +/\*[0-9a-f]{4}\*/'); done
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "posm_core.h"

#define IMG_DOMAIN_A U64C(0x494D4147454F4E45)   /* the SipHash image (v0.12.0 and before) */
#define IMG_DOMAIN_B U64C(0x494D41474554574F)
INLINE void image_group_sip(u64 g, const u64* w, u64* t) {
    struct u128 a = siphash24_128(IMG_DOMAIN_A ^ g, IMG_DOMAIN_B, w[0], w[1], w[2], w[3]);
    struct u128 b = siphash24_128(IMG_DOMAIN_B ^ g, IMG_DOMAIN_A, w[0], w[1], w[2], w[3]);
    t[0] = a.lo; t[1] = a.hi; t[2] = b.lo; t[3] = b.hi;
}

#define MEM_KERNEL(name, fn)                                                                   \
extern "C" __global__ void name(const u64* in, u64* out, u64 groups) {                          \
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (g >= groups) return;                                                                    \
    u64 w[4] = { in[4 * g], in[4 * g + 1], in[4 * g + 2], in[4 * g + 3] }, t[4];                \
    fn(g, w, t);                                                                                \
    out[4 * g] = t[0]; out[4 * g + 1] = t[1]; out[4 * g + 2] = t[2]; out[4 * g + 3] = t[3];     \
}
#define ALU_KERNEL(name, fn)                                                                   \
extern "C" __global__ void name(u64* out, u64 threads, int iters) {                             \
    u64 g = (u64)blockIdx.x * blockDim.x + threadIdx.x;                                         \
    if (g >= threads) return;                                                                   \
    u64 w[4] = { splitmix64(g), splitmix64(g + 1), splitmix64(g + 2), splitmix64(g + 3) }, t[4];\
    for (int i = 0; i < iters; i++) {                                                           \
        fn(g + i, w, t);                                                                        \
        w[0] = t[0]; w[1] = t[1]; w[2] = t[2]; w[3] = t[3];                                     \
    }                                                                                           \
    out[g] = w[0] ^ w[1] ^ w[2] ^ w[3];                                                         \
}
MEM_KERNEL(mem_siphash, image_group_sip)
MEM_KERNEL(mem_blake3, image_group)
ALU_KERNEL(alu_siphash, image_group_sip)
ALU_KERNEL(alu_blake3, image_group)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char** argv) {
    int lg = argc > 1 ? atoi(argv[1]) : 26, iters = argc > 2 ? atoi(argv[2]) : 64;
    u64 groups = 1ULL << lg;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs; %llu groups (%.1f GiB per array)\n", prop.name, prop.multiProcessorCount,
           (unsigned long long)groups, groups * 32.0 / (1 << 30));
    u64 *in, *out; CK(cudaMalloc(&in, groups * 32)); CK(cudaMalloc(&out, groups * 32));
    u64* h = (u64*)malloc(groups * 32);
    for (u64 i = 0; i < 4 * groups; i++) h[i] = splitmix64(i);
    CK(cudaMemcpy(in, h, groups * 32, cudaMemcpyHostToDevice));
    const unsigned TB = 256, blocks = (unsigned)((groups + TB - 1) / TB);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // correctness: device image == host image (same header) on a sample of groups
    mem_blake3<<<blocks, TB>>>(in, out, groups); CK(cudaDeviceSynchronize());
    u64* d = (u64*)malloc(groups * 32); CK(cudaMemcpy(d, out, groups * 32, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (u64 g = 0; g < groups; g += 1 + (groups >> 12)) {
        u64 t[4]; image_group(g, h + 4 * g, t);
        for (int i = 0; i < 4; i++) bad += t[i] != d[4 * g + i];
    }
    { u64 z[4] = {0, 0, 0, 0}, t[4]; image_group(0, z, t);       // known answer (BLAKE3 keyed_hash, see tests)
      bad += t[0] != U64C(0x7a19ad0e633de21e) || t[3] != U64C(0x85c907abef4a4e57); }
    printf("device image == host image: %s\n", bad ? "MISMATCH" : "ok");

    const char* names[2] = { "siphash", "blake3" };
    for (int f = 0; f < 2; f++) {
        float best = 1e30f, ms;
        for (int rep = 0; rep < 10; rep++) {
            CK(cudaEventRecord(e0));
            if (f) mem_blake3<<<blocks, TB>>>(in, out, groups); else mem_siphash<<<blocks, TB>>>(in, out, groups);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        printf("mem %-8s %8.3f ms  %7.2f Ggroups/s  %7.1f GB/s of Theta' imaged\n", names[f], best,
               groups / (best * 1e-3) / 1e9, groups * 32.0 / (best * 1e-3) / 1e9);
    }
    u64 threads = (u64)prop.multiProcessorCount * 2048 * 4;   // a few waves of full occupancy
    unsigned ablocks = (unsigned)((threads + TB - 1) / TB);
    for (int f = 0; f < 2; f++) {
        float best = 1e30f, ms;
        for (int rep = 0; rep < 10; rep++) {
            CK(cudaEventRecord(e0));
            if (f) alu_blake3<<<ablocks, TB>>>(out, threads, iters); else alu_siphash<<<ablocks, TB>>>(out, threads, iters);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        double n = (double)threads * iters;
        printf("alu %-8s %8.3f ms  %7.2f Gimages/s  (%.3f ns per image per SM)\n", names[f], best,
               n / (best * 1e-3) / 1e9, best * 1e6 * prop.multiProcessorCount / n);
    }
    CK(cudaGetLastError());
    return bad != 0;
}
