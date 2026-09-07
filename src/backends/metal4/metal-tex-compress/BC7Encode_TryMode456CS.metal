#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct type_cbCS
{
    uint g_tex_width;
    uint g_num_block_x;
    uint g_format;
    uint g_mode_id;
    uint g_start_block_id;
    uint g_num_total_blocks;
    float g_alpha_weight;
};

struct type_RWStructuredBuffer_v4uint
{
    uint4 _m0[1];
};

struct BufferShared
{
    uint4 pixel;
    uint error;
    uint mode;
    uint _partition;
    uint index_selector;
    uint rotation;
    uint4 endPoint_low;
    uint4 endPoint_high;
    uint4 endPoint_low_quantized;
    uint4 endPoint_high_quantized;
};

constant uint4 _136 = {};

struct spvDescriptorSetBuffer0
{
    constant type_cbCS* cbCS [[id(0)]];
    texture2d<float> g_Input [[id(1)]];
    device type_RWStructuredBuffer_v4uint* g_OutBuff [[id(2)]];
};

constant spvUnsafeArray<uint, 16> _119 = spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u });
constant spvUnsafeArray<uint, 16> _120 = spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<uint, 16> _121 = spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<spvUnsafeArray<uint, 16>, 3> _122 = spvUnsafeArray<spvUnsafeArray<uint, 16>, 3>({ spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u }), spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }), spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }) });
constant spvUnsafeArray<uint, 64> _123 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });
constant spvUnsafeArray<uint, 64> _124 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<uint, 64> _125 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u });
constant spvUnsafeArray<spvUnsafeArray<uint, 64>, 3> _126 = spvUnsafeArray<spvUnsafeArray<uint, 64>, 3>({ spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u }) });

kernel void TryMode456CS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<BufferShared, 64> shared_temp;
    uint _145 = gl_LocalInvocationIndex / 16u;
    uint _151 = ((*spvDescriptorSet0.cbCS).g_start_block_id + (gl_WorkGroupID.x * 4u)) + _145;
    uint _152 = _145 * 16u;
    uint _153 = gl_LocalInvocationIndex - _152;
    uint _156 = _151 / (*spvDescriptorSet0.cbCS).g_num_block_x;
    bool _161 = _153 < 16u;
    if (_161)
    {
        int3 _169 = int3(uint3(((_151 - (_156 * (*spvDescriptorSet0.cbCS).g_num_block_x)) * 4u) + (_153 % 4u), (_156 * 4u) + (_153 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = clamp(uint4(spvDescriptorSet0.g_Input.read(uint2(_169.xy), _169.z) * 255.0), uint4(0u), uint4(255u));
        shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[gl_LocalInvocationIndex].pixel;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[gl_LocalInvocationIndex].pixel;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _182 = _153 < 8u;
    if (_182)
    {
        uint _187 = gl_LocalInvocationIndex + 8u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_187].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_187].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _196 = _153 < 4u;
    if (_196)
    {
        uint _201 = gl_LocalInvocationIndex + 4u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_201].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_201].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _210 = _153 < 2u;
    if (_210)
    {
        uint _215 = gl_LocalInvocationIndex + 2u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_215].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_215].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _224 = _153 < 1u;
    if (_224)
    {
        uint _229 = gl_LocalInvocationIndex + 1u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_229].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_229].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    spvUnsafeArray<uint4, 2> _141;
    _141[0u] = shared_temp[_152].endPoint_low;
    _141[1u] = shared_temp[_152].endPoint_high;
    uint2 _252;
    uint _253;
    if (_182)
    {
        bool _248 = 0u == (_153 & 1u);
        _252 = select(uint2(1u, 2u), uint2(2u, 1u), bool2(_248));
        _253 = _248 ? 0u : 1u;
    }
    else
    {
        _252 = uint2(2u);
        _253 = 0u;
    }
    uint _871;
    uint _872;
    uint _873;
    if (_153 < 12u)
    {
        uint _299;
        if (_210 || (8u == _153))
        {
            _299 = 0u;
        }
        else
        {
            uint _298;
            if (_196 || (9u == _153))
            {
                _141[0u] = uint4(_141[0u].w, _141[0u].y, _141[0u].z, _141[0u].x);
                _141[1u] = uint4(_141[1u].w, _141[1u].y, _141[1u].z, _141[1u].x);
                _298 = 1u;
            }
            else
            {
                uint _297;
                if ((_153 < 6u) || (10u == _153))
                {
                    _141[0u] = uint4(_141[0u].x, _141[0u].w, _141[0u].z, _141[0u].y);
                    _141[1u] = uint4(_141[1u].x, _141[1u].w, _141[1u].z, _141[1u].y);
                    _297 = 2u;
                }
                else
                {
                    bool _287 = _182 || (11u == _153);
                    if (_287)
                    {
                        _141[0u] = uint4(_141[0u].x, _141[0u].y, _141[0u].w, _141[0u].z);
                        _141[1u] = uint4(_141[1u].x, _141[1u].y, _141[1u].w, _141[1u].z);
                    }
                    _297 = _287 ? 3u : 0u;
                }
                _298 = _297;
            }
            _299 = _298;
        }
        if (_182)
        {
            uint _311 = _141[0u].w;
            uint4 _312 = uint4(_311);
            uint4 _320 = (((((_141[0u].xyzz << uint4(8u)) + _141[0u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(3u);
            uint4 _322 = _320 | (_320 >> uint4(5u));
            _141[0u] = uint4(_322.x, _322.y, _322.z, _141[0u].w);
            uint4 _326 = uint4((((((_312 << uint4(8u)) + _312) * uint4(63u)) + uint4(32768u)) >> uint4(16u)).x) << uint4(2u);
            _141[0u].w = (_326 | (_326 >> uint4(6u))).x;
            uint _338 = _141[1u].w;
            uint4 _339 = uint4(_338);
            uint4 _347 = (((((_141[1u].xyzz << uint4(8u)) + _141[1u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(3u);
            uint4 _349 = _347 | (_347 >> uint4(5u));
            _141[1u] = uint4(_349.x, _349.y, _349.z, _141[1u].w);
            uint4 _353 = uint4((((((_339 << uint4(8u)) + _339) * uint4(63u)) + uint4(32768u)) >> uint4(16u)).x) << uint4(2u);
            _141[1u].w = (_353 | (_353 >> uint4(6u))).x;
        }
        else
        {
            uint4 _365 = (((((_141[0u].xyzz << uint4(8u)) + _141[0u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(1u);
            uint4 _367 = _365 | (_365 >> uint4(7u));
            _141[0u] = uint4(_367.x, _367.y, _367.z, _141[0u].w);
            uint4 _378 = (((((_141[1u].xyzz << uint4(8u)) + _141[1u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(1u);
            uint4 _380 = _378 | (_378 >> uint4(7u));
            _141[1u] = uint4(_380.x, _380.y, _380.z, _141[1u].w);
        }
        bool _386 = 1u == _299;
        uint4 _402;
        if (_386)
        {
            _402 = uint4(shared_temp[_152].pixel.w, shared_temp[_152].pixel.y, shared_temp[_152].pixel.z, shared_temp[_152].pixel.x);
        }
        else
        {
            uint4 _401;
            if (2u == _299)
            {
                _401 = uint4(shared_temp[_152].pixel.x, shared_temp[_152].pixel.w, shared_temp[_152].pixel.z, shared_temp[_152].pixel.y);
            }
            else
            {
                uint4 _400;
                if (3u == _299)
                {
                    _400 = uint4(shared_temp[_152].pixel.x, shared_temp[_152].pixel.y, shared_temp[_152].pixel.w, shared_temp[_152].pixel.z);
                }
                else
                {
                    _400 = shared_temp[_152].pixel;
                }
                _401 = _400;
            }
            _402 = _401;
        }
        uint4 _403 = _141[1u];
        uint4 _404 = _141[0u];
        int4 _406 = int4(_403 - _404);
        int _407 = _406.x;
        int _409 = _406.y;
        int _411 = _406.z;
        int _416 = _406.w;
        int2 _420 = int2(uint2(uint(((_407 * _407) + (_409 * _409)) + (_411 * _411)), uint(_416 * _416)));
        uint3 _424 = _402.xyz - _141[0u].xyz;
        uint3 _427 = _402.xyz - _141[0u].xyz;
        uint3 _442 = _402.xyz - _141[1u].xyz;
        uint3 _445 = _402.xyz - _141[1u].xyz;
        int4 _470;
        if (int(((_424.x * _427.x) + (_424.y * _427.y)) + (_424.z * _427.z)) > int(((_442.x * _445.x) + (_442.y * _445.y)) + (_442.z * _445.z)))
        {
            int3 _462 = -_406.xyz;
            uint4 _464 = _141[0u];
            _141[0u] = uint4(_141[1u].x, _141[1u].y, _141[1u].z, _141[0u].w);
            _141[1u] = uint4(_464.x, _464.y, _464.z, _141[1u].w);
            _470 = int4(_462.x, _462.y, _462.z, _406.w);
        }
        else
        {
            _470 = _406;
        }
        int4 _494;
        if (int((_402.w - _141[0u].w) * (_402.w - _141[0u].w)) > int((_402.w - _141[1u].w) * (_402.w - _141[1u].w)))
        {
            int4 _491 = _470;
            _491.w = -_470.w;
            uint _492 = _141[0u].w;
            _141[0u].w = _141[1u].w;
            _141[1u].w = _492;
            _494 = _491;
        }
        else
        {
            _494 = _470;
        }
        uint _496;
        _496 = 0u;
        uint _497;
        for (uint _499 = 0u; _499 < 16u; _496 = _497, _499++)
        {
            uint _504 = _152 + _499;
            uint4 _522;
            if (_386)
            {
                _522 = uint4(shared_temp[_504].pixel.w, shared_temp[_504].pixel.y, shared_temp[_504].pixel.z, shared_temp[_504].pixel.x);
            }
            else
            {
                uint4 _521;
                if (2u == _299)
                {
                    _521 = uint4(shared_temp[_504].pixel.x, shared_temp[_504].pixel.w, shared_temp[_504].pixel.z, shared_temp[_504].pixel.y);
                }
                else
                {
                    uint4 _520;
                    if (3u == _299)
                    {
                        _520 = uint4(shared_temp[_504].pixel.x, shared_temp[_504].pixel.y, shared_temp[_504].pixel.w, shared_temp[_504].pixel.z);
                    }
                    else
                    {
                        _520 = shared_temp[_504].pixel;
                    }
                    _521 = _520;
                }
                _522 = _521;
            }
            uint3 _524 = uint3(_494.xyz);
            uint3 _528 = _522.xyz - _141[0u].xyz;
            int _540 = int(((_524.x * _528.x) + (_524.y * _528.y)) + (_524.z * _528.z));
            int _541 = _420.x;
            uint _557 = ((_541 <= 0) || (_540 <= 0)) ? 0u : ((_540 < _541) ? _126[_252.x][uint((float(_540) * 63.499988555908203125) / float(_541))] : _126[_252.x][63]);
            int _564 = int(uint(_494.w) * (_522.w - _141[0u].w));
            int _565 = _420.y;
            uint _581 = ((_565 <= 0) || (_564 <= 0)) ? 0u : ((_564 < _565) ? _126[_252.y][uint((float(_564) * 63.499988555908203125) / float(_565))] : _126[_252.y][63]);
            uint3 _596 = (((uint3(64u - _122[_252.x][_557]) * _141[0u].xyz) + (uint3(_122[_252.x][_557]) * _141[1u].xyz)) + uint3(32u)) >> uint3(6u);
            uint4 _597 = uint4(_596.x, _596.y, _596.z, _136.w);
            _597.w = ((((64u - _122[_252.y][_581]) * _141[0u].w) + (_122[_252.y][_581] * _141[1u].w)) + 32u) >> 6u;
            uint _610 = _596.x;
            uint4 _617;
            uint4 _618;
            if (_610 < _522.x)
            {
                uint4 _615 = _597;
                _615.x = _522.x;
                uint4 _616 = _522;
                _616.x = _610;
                _617 = _616;
                _618 = _615;
            }
            else
            {
                _617 = _522;
                _618 = _597;
            }
            uint4 _626;
            uint4 _627;
            if (_618.y < _617.y)
            {
                uint4 _624 = _618;
                _624.y = _617.y;
                uint4 _625 = _617;
                _625.y = _618.y;
                _626 = _625;
                _627 = _624;
            }
            else
            {
                _626 = _617;
                _627 = _618;
            }
            uint4 _635;
            uint4 _636;
            if (_627.z < _626.z)
            {
                uint4 _633 = _627;
                _633.z = _626.z;
                uint4 _634 = _626;
                _634.z = _627.z;
                _635 = _634;
                _636 = _633;
            }
            else
            {
                _635 = _626;
                _636 = _627;
            }
            uint4 _644;
            uint4 _645;
            if (_636.w < _635.w)
            {
                uint4 _642 = _636;
                _642.w = _635.w;
                uint4 _643 = _635;
                _643.w = _636.w;
                _644 = _642;
                _645 = _643;
            }
            else
            {
                _644 = _636;
                _645 = _635;
            }
            uint4 _646 = _644 - _645;
            uint4 _662;
            if (_386)
            {
                _662 = uint4(_646.w, _646.y, _646.z, _646.x);
            }
            else
            {
                uint4 _661;
                if (2u == _299)
                {
                    _661 = uint4(_646.x, _646.w, _646.z, _646.y);
                }
                else
                {
                    uint4 _660;
                    if (3u == _299)
                    {
                        _660 = uint4(_646.x, _646.y, _646.w, _646.z);
                    }
                    else
                    {
                        _660 = _646;
                    }
                    _661 = _660;
                }
                _662 = _661;
            }
            float _675 = float(_662.w);
            _497 = _496 + uint(float(((_662.x * _662.x) + (_662.y * _662.y)) + (_662.z * _662.z)) + (((*spvDescriptorSet0.cbCS).g_alpha_weight * _675) * _675));
        }
        _871 = _299;
        _872 = _182 ? 4u : 5u;
        _873 = _496;
    }
    else
    {
        uint _868;
        uint _869;
        if (_161)
        {
            uint _682 = _153 - 12u;
            uint2 _142 = uint2(_682 >> 0u, _682 >> 1u) & uint2(1u);
            spvUnsafeArray<uint4, 2> _140;
            for (uint _688 = 0u; _688 < 2u; )
            {
                _140[_688] = _141[_688] & uint4(4294967294u);
                _140[_688] |= uint4(_142[_688]);
                _141[_688] = _140[_688];
                _688++;
                continue;
            }
            uint4 _705 = _141[1u];
            uint4 _706 = _141[0u];
            int4 _708 = int4(_705 - _706);
            int _709 = _708.x;
            int _711 = _708.y;
            int _713 = _708.z;
            int _715 = _708.w;
            int _719 = (((_709 * _709) + (_711 * _711)) + (_713 * _713)) + (_715 * _715);
            uint4 _720 = uint4(_708);
            uint4 _722 = shared_temp[_152].pixel - _141[0u];
            int _738 = int((((_720.x * _722.x) + (_720.y * _722.y)) + (_720.z * _722.z)) + (_720.w * _722.w));
            int4 _754;
            if (((_719 > 0) && (_738 >= 0)) && (uint(float(_738) * 63.499988555908203125) > uint(32 * _719)))
            {
                uint4 _752 = _141[0u];
                _141[0u] = _141[1u];
                _141[1u] = _752;
                _754 = -_708;
            }
            else
            {
                _754 = _708;
            }
            uint _756;
            _756 = 0u;
            uint _757;
            for (uint _759 = 0u; _759 < 16u; _756 = _757, _759++)
            {
                uint _764 = _152 + _759;
                uint4 _766 = shared_temp[_764].pixel;
                uint4 _767 = uint4(_754);
                uint4 _769 = _766 - _141[0u];
                int _785 = int((((_767.x * _769.x) + (_767.y * _769.y)) + (_767.z * _769.z)) + (_767.w * _769.w));
                uint _800 = ((_719 <= 0) || (_785 <= 0)) ? 0u : ((_785 < _719) ? _126[0][uint((float(_785) * 63.499988555908203125) / float(_719))] : _126[0][63]);
                uint4 _813 = (((uint4(64u - _122[0][_800]) * _141[0u]) + (uint4(_122[0][_800]) * _141[1u])) + uint4(32u)) >> uint4(6u);
                uint _814 = _813.x;
                uint4 _821;
                uint4 _822;
                if (_814 < _766.x)
                {
                    uint4 _819 = _813;
                    _819.x = _766.x;
                    uint4 _820 = _766;
                    _820.x = _814;
                    _821 = _820;
                    _822 = _819;
                }
                else
                {
                    _821 = _766;
                    _822 = _813;
                }
                uint4 _830;
                uint4 _831;
                if (_822.y < _821.y)
                {
                    uint4 _828 = _822;
                    _828.y = _821.y;
                    uint4 _829 = _821;
                    _829.y = _822.y;
                    _830 = _829;
                    _831 = _828;
                }
                else
                {
                    _830 = _821;
                    _831 = _822;
                }
                uint4 _839;
                uint4 _840;
                if (_831.z < _830.z)
                {
                    uint4 _837 = _831;
                    _837.z = _830.z;
                    uint4 _838 = _830;
                    _838.z = _831.z;
                    _839 = _838;
                    _840 = _837;
                }
                else
                {
                    _839 = _830;
                    _840 = _831;
                }
                uint4 _848;
                uint4 _849;
                if (_840.w < _839.w)
                {
                    uint4 _846 = _840;
                    _846.w = _839.w;
                    uint4 _847 = _839;
                    _847.w = _840.w;
                    _848 = _846;
                    _849 = _847;
                }
                else
                {
                    _848 = _840;
                    _849 = _839;
                }
                uint4 _850 = _848 - _849;
                uint _851 = _850.x;
                uint _853 = _850.y;
                uint _855 = _850.z;
                float _863 = float(_850.w);
                _757 = _756 + uint(float(((_851 * _851) + (_853 * _853)) + (_855 * _855)) + (((*spvDescriptorSet0.cbCS).g_alpha_weight * _863) * _863));
            }
            _868 = _682;
            _869 = _756;
        }
        else
        {
            _868 = 0u;
            _869 = 4294967295u;
        }
        _871 = _868;
        _872 = _161 ? 6u : 0u;
        _873 = _869;
    }
    shared_temp[gl_LocalInvocationIndex].error = _873;
    shared_temp[gl_LocalInvocationIndex].mode = _872;
    shared_temp[gl_LocalInvocationIndex].index_selector = _253;
    shared_temp[gl_LocalInvocationIndex].rotation = _871;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_182)
    {
        uint _881 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_881].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_881].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_881].mode;
            shared_temp[gl_LocalInvocationIndex].index_selector = shared_temp[_881].index_selector;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_881].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_196)
    {
        uint _897 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_897].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_897].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_897].mode;
            shared_temp[gl_LocalInvocationIndex].index_selector = shared_temp[_897].index_selector;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_897].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_210)
    {
        uint _913 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_913].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_913].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_913].mode;
            shared_temp[gl_LocalInvocationIndex].index_selector = shared_temp[_913].index_selector;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_913].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_224)
    {
        uint _929 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_929].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_929].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_929].mode;
            shared_temp[gl_LocalInvocationIndex].index_selector = shared_temp[_929].index_selector;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_929].rotation;
        }
        (*spvDescriptorSet0.g_OutBuff)._m0[_151] = uint4(shared_temp[gl_LocalInvocationIndex].error, (shared_temp[gl_LocalInvocationIndex].index_selector << 31u) | shared_temp[gl_LocalInvocationIndex].mode, 0u, shared_temp[gl_LocalInvocationIndex].rotation);
    }
}

