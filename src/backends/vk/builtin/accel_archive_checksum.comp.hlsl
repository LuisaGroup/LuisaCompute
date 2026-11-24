ByteAddressBuffer _Src : register(t0);
RWByteAddressBuffer _Out : register(u1);

struct PushConstants {
    uint total_words;
    uint tail_mask;
    uint base_byte_offset;
    uint start_word;
    uint grid_stride;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

groupshared uint s0[256];
groupshared uint s1[256];
groupshared uint s2[256];
groupshared uint s3[256];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    uint a0 = 0u;
    uint a1 = 0u;
    uint a2 = 0u;
    uint a3 = 0u;
    for (uint idx = pc.start_word + dtid.x; idx < pc.total_words; idx += pc.grid_stride) {
        uint w = _Src.Load(pc.base_byte_offset + idx * 4u);
        if (idx + 1u == pc.total_words) {
            w &= pc.tail_mask;
        }
        uint x = w ^ (idx * 0x9E3779B1u + 0x85EBCA6Bu);
        x ^= (x >> 16u);
        x *= 0x7FEB352Du;
        x ^= (x >> 15u);
        x *= 0x846CA68Bu;
        x ^= (x >> 16u);

        uint y = (w + 0x27D4EB2Du) ^ ((idx ^ 0xA5A55A5Au) * 0xC2B2AE35u);
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
    uint tid = gtid.x;
    s0[tid] = a0;
    s1[tid] = a1;
    s2[tid] = a2;
    s3[tid] = a3;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] ^= s1[tid + stride];
            s2[tid] += s2[tid + stride];
            s3[tid] ^= s3[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        uint unused_val = 0u;
        _Out.InterlockedAdd(0u, s0[0], unused_val);
        _Out.InterlockedXor(4u, s1[0], unused_val);
        _Out.InterlockedAdd(8u, s2[0], unused_val);
        _Out.InterlockedXor(12u, s3[0], unused_val);
    }
}
