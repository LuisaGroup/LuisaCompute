#include <hip/hip_runtime.h>

extern "C" __global__ void accel_archive_checksum(
    const unsigned int *__restrict__ words,
    unsigned int total_words,
    unsigned int tail_mask,
    unsigned int start_word,
    unsigned int *__restrict__ out_hash) {
    unsigned int a0 = 0u;
    unsigned int a1 = 0u;
    unsigned int a2 = 0u;
    unsigned int a3 = 0u;

    unsigned int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int grid_stride = gridDim.x * blockDim.x;
    for (unsigned int idx = start_word + tid_global; idx < total_words; idx += grid_stride) {
        unsigned int w = words[idx];
        if (idx + 1u == total_words) {
            w &= tail_mask;
        }
        unsigned int x = w ^ (idx * 0x9E3779B1u + 0x85EBCA6Bu);
        x ^= (x >> 16u);
        x *= 0x7FEB352Du;
        x ^= (x >> 15u);
        x *= 0x846CA68Bu;
        x ^= (x >> 16u);

        unsigned int y = (w + 0x27D4EB2Du) ^ ((idx ^ 0xA5A55A5Au) * 0xC2B2AE35u);
        y ^= (y >> 16u);
        y *= 0x85EBCA6Bu;
        y ^= (y >> 13u);
        y *= 0xC2B2AE35u;
        y ^= (y >> 16u);

        a0 += x;
        a1 ^= x;
        a2 += y;
        a3 ^= y;
    }

    __shared__ unsigned int s0[256];
    __shared__ unsigned int s1[256];
    __shared__ unsigned int s2[256];
    __shared__ unsigned int s3[256];
    unsigned int tid = threadIdx.x;
    s0[tid] = a0;
    s1[tid] = a1;
    s2[tid] = a2;
    s3[tid] = a3;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] ^= s1[tid + stride];
            s2[tid] += s2[tid + stride];
            s3[tid] ^= s3[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        atomicAdd(&out_hash[0], s0[0]);
        atomicXor(&out_hash[1], s1[0]);
        atomicAdd(&out_hash[2], s2[0]);
        atomicXor(&out_hash[3], s3[0]);
    }
}
