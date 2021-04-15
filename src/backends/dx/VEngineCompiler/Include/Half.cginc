#ifndef VENGINE_HALF_INCLUDE
    #define VENGINE_HALF_INCLUDE
    //IEEE 754 Float16
    uint select(uint a, uint b, bool c)
    {
        if (!c)
        {
            return a;
        }
        return b;
    }

    uint2 select(uint2 a, uint2 b, bool c)
    {
        if (!c)
        {
            return a;
        }
        return b;
    }

    uint3 select(uint3 a, uint3 b, bool c)
    {
        if (!c)
        {
            return a;
        }
        return b;
    }

    uint4 select(uint4 a, uint4 b, bool c)
    {
        if (!c)
        {
            return a;
        }
        return b;
    }

    uint2 select(uint2 a, uint2 b, bool2 c)
    {
        return uint2(c.x ? b.x : a.x, c.y ? b.y : a.y);
    }

    uint3 select(uint3 a, uint3 b, bool3 c)
    {
        return uint3(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z);
    }

    uint4 select(uint4 a, uint4 b, bool4 c)
    {
        return uint4(c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w);
    }

    float f16tof32(uint x)
    {
        uint num = (x & 32767) << 13;
        uint num2 = num & 260046848;
        uint num3 = num + 939524096 + select(0, 939524096, num2 == 260046848);
        return asfloat(select(num3, asuint(asfloat(num3 + 8388608) - 6.10351563E-05f), num2 == 0) | ((x & 32768) << 16));
    }

    float2 f16tof32(uint2 x)
    {
        uint2 lhs = (x & 32767) << 13;
        uint2 lhs2 = lhs & 260046848;
        uint2 value_uint = lhs + 939524096 + select(0, 939524096, lhs2 == 260046848);
        return asfloat(select(value_uint, asuint(asfloat(value_uint + 8388608) - 6.10351563E-05f), lhs2 == 0) | ((x & 32768) << 16));
    }

    float3 f16tof32(uint3 x)
    {
        uint3 lhs = (x & 32767) << 13;
        uint3 lhs2 = lhs & 260046848;
        uint3 value_uint = lhs + 939524096 + select(0, 939524096, lhs2 == 260046848);
        return asfloat(select(value_uint, asuint(asfloat(value_uint + 8388608) - 6.10351563E-05f), lhs2 == 0) | ((x & 32768) << 16));
    }

    float4 f16tof32(uint4 x)
    {
        uint4 lhs = (x & 32767) << 13;
        uint4 lhs2 = lhs & 260046848;
        uint4 value_uint = lhs + 939524096 + select(0u, 939524096, lhs2 == 260046848);
        return asfloat(select(value_uint, asuint(asfloat(value_uint + 8388608) - 6.10351563E-05f), lhs2 == 0) | ((x & 32768) << 16));
    }
#endif