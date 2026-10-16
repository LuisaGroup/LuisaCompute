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

struct alignas(16) type_cbCS
{
    uint g_tex_width;
    uint g_num_block_x;
    uint g_format;
    uint g_mode_id;
    uint g_start_block_id;
    uint g_num_total_blocks;
};

struct type_RWStructuredBuffer_v4uint
{
    uint4 _m0[1];
};

struct SharedData
{
    float3 pixel;
    int3 pixel_ph;
    float3 pixel_hr;
    float pixel_lum;
    float error;
    uint best_mode;
    uint best_partition;
    int3 endPoint_low;
    int3 endPoint_high;
    float endPoint_lum_low;
    float endPoint_lum_high;
};

constant bool _154 = {};

struct spvDescriptorSetBuffer0
{
    type_cbCS cbCS;
    texture2d<float> g_Input;
    ulong _;
    device type_RWStructuredBuffer_v4uint* g_OutBuff;
};

constant spvUnsafeArray<uint, 14> _155 = spvUnsafeArray<uint, 14>({ 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u });
constant spvUnsafeArray<bool, 14> _156 = spvUnsafeArray<bool, 14>({ true, true, true, true, true, true, true, true, true, false, false, true, true, true });
constant spvUnsafeArray<uint4, 14> _171 = spvUnsafeArray<uint4, 14>({ uint4(10u, 5u, 5u, 5u), uint4(7u, 6u, 6u, 6u), uint4(11u, 5u, 4u, 4u), uint4(11u, 4u, 5u, 4u), uint4(11u, 4u, 4u, 5u), uint4(9u, 5u, 5u, 5u), uint4(8u, 6u, 5u, 5u), uint4(8u, 5u, 6u, 5u), uint4(8u, 5u, 5u, 6u), uint4(6u), uint4(10u), uint4(11u, 9u, 9u, 9u), uint4(12u, 8u, 8u, 8u), uint4(16u, 4u, 4u, 4u) });
constant spvUnsafeArray<uint, 64> _172 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });
constant spvUnsafeArray<int, 16> _173 = spvUnsafeArray<int, 16>({ 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 });

kernel void TryModeG10CS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<SharedData, 64> shared_temp;
    uint _180 = gl_LocalInvocationIndex / 16u;
    uint _186 = (spvDescriptorSet0.cbCS.g_start_block_id + (gl_WorkGroupID.x * 4u)) + _180;
    uint _187 = _180 * 16u;
    uint _188 = gl_LocalInvocationIndex - _187;
    uint _191 = _186 / spvDescriptorSet0.cbCS.g_num_block_x;
    if (_188 < 16u)
    {
        int3 _204 = int3(uint3(((_186 - (_191 * spvDescriptorSet0.cbCS.g_num_block_x)) * 4u) + (_188 % 4u), (_191 * 4u) + (_188 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = spvDescriptorSet0.g_Input.read(uint2(_204.xy), _204.z).xyz;
        shared_temp[gl_LocalInvocationIndex].pixel = precise::max(shared_temp[gl_LocalInvocationIndex].pixel, float3(0.0));
        float3 _213 = shared_temp[gl_LocalInvocationIndex].pixel;
        uint _216 = as_type<uint>(half2(float2(_213.x, 0.0)));
        uint _219 = as_type<uint>(half2(float2(_213.y, 0.0)));
        uint _222 = as_type<uint>(half2(float2(_213.z, 0.0)));
        uint3 _223 = uint3(_216, _219, _222);
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(float2(as_type<half2>(_216)).x, float2(as_type<half2>(_219)).x, float2(as_type<half2>(_222)).x);
        shared_temp[gl_LocalInvocationIndex].pixel_lum = dot(shared_temp[gl_LocalInvocationIndex].pixel_hr, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
        int3 _259;
        do
        {
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                _259 = int3((_223 << uint3(6u)) / uint3(31u));
                break;
            }
            else
            {
                bool3 _244 = _223 == uint3(31743u);
                _259 = select(select(-int3(((uint3(32767u) & _223) << uint3(5u)) / uint3(31u)), int3(-32767), _244), select(int3((_223 << uint3(5u)) / uint3(31u)), int3(32767), _244), _223 < uint3(32768u));
                break;
            }
            break; // unreachable workaround
        } while(false);
        shared_temp[gl_LocalInvocationIndex].pixel_ph = _259;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[gl_LocalInvocationIndex].pixel_ph;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[gl_LocalInvocationIndex].pixel_ph;
        shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[gl_LocalInvocationIndex].pixel_lum;
        shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[gl_LocalInvocationIndex].pixel_lum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_188 < 8u)
    {
        uint _274 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_274].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_274].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_274].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_274].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_274].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_274].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _295 = _188 < 4u;
    if (_295)
    {
        uint _300 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_300].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_300].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_300].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_300].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_300].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_300].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _321 = _188 < 2u;
    if (_321)
    {
        uint _326 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_326].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_326].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_326].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_326].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_326].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_326].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _347 = _188 < 1u;
    if (_347)
    {
        uint _352 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_352].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_352].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_352].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_352].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_352].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_352].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_188 == 0u)
    {
        int3 _377 = shared_temp[_187].endPoint_low;
        float3 _381 = float3(shared_temp[_187].endPoint_high - _377);
        float _382 = dot(_381, _381);
        float _387 = dot(_381, float3(shared_temp[_187].pixel_ph - _377));
        if (((_382 > 0.0) && (_387 >= 0.0)) && (uint((_387 * 63.499988555908203125) / _382) > 32u))
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_187].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_high = _377;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_295)
    {
        float3 _407 = float3(shared_temp[_187].endPoint_high - shared_temp[_187].endPoint_low);
        float _408 = dot(_407, _407);
        uint _409 = _188 + 10u;
        spvUnsafeArray<int3, 2> _412 = spvUnsafeArray<int3, 2>({ shared_temp[_187].endPoint_low, shared_temp[_187].endPoint_high });
        spvUnsafeArray<int3, 2> _177 = _412;
        int _414 = int(_171[_409].x);
        for (int _416 = 0; _416 < 2; _416++)
        {
            uint _422 = uint(_416);
            int3 _478;
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                _478 = select(select((_177[_422] << (int3(_414) & int3(31))) >> int3(16), int3((1 << (_414 & 31)) - 1), _177[_422] == int3(65535)), _177[_422], (int3(int(_414 >= 15)) | int3(_177[_422] == int3(0))) != int3(0));
            }
            else
            {
                int _440 = _414 - 1;
                int _442 = 1 << (_440 & 31);
                int3 _446 = int3(_440) & int3(31);
                int3 _450 = -_177[_422];
                _478 = select(select(select(-((_450 << _446) >> int3(15)), int3(1 - _442), _450 == int3(32767)), select((_177[_422] << _446) >> int3(15), int3(_442 - 1), _177[_422] == int3(32767)), _177[_422] >= int3(0)), _177[_422], (int3(int(_414 >= 16)) | int3(_177[_422] == int3(0))) != int3(0));
            }
            _177[_422] = _478;
        }
        if (_156[_409])
        {
            _177[1u] -= _177[0u];
        }
        bool _489;
        _489 = _154;
        bool _490;
        for (int _492 = 0; _492 < 2; _489 = _490, _492++)
        {
            uint _497 = uint(_492);
            int3 _499 = _177[_497];
            int3 _545;
            if (_156[_409])
            {
                bool3 _512 = bool3(_499.y >= 0);
                uint3 _514 = uint3(uint(_499.y));
                uint3 _518 = uint3(1u) << ((_171[_409].yzw - uint3(1u)) & uint3(31u));
                bool3 _519 = _514 >= _518;
                bool3 _523 = uint3(uint(-_499.y)) > _518;
                int3 _533 = _499;
                _533.x = int(uint(_499.x) & ((1u << (_171[_409].x & 31u)) - 1u));
                _533.y = int(select(select(_514 & ((uint3(1u) << (_171[_409].yzw & uint3(31u))) - uint3(1u)), _518, _523), select(_514, _518 - uint3(1u), _519), _512).x);
                _545 = _533;
                _490 = any(select(_523, _519, _512));
            }
            else
            {
                _545 = int3(uint3(_499) & uint3((1u << (_171[_409].x & 31u)) - 1u));
                _490 = false;
            }
            _177[_497] = _545;
        }
        bool _548 = spvDescriptorSet0.cbCS.g_format == 96u;
        if (_548)
        {
            uint3 _555 = uint3(1u) << ((uint3(_171[_409].x) - uint3(1u)) & uint3(31u));
            _177[0u] = int3(select(uint3(_177[0u]), (uint3(_177[0u]) & (_555 - uint3(1u))) - _555, (uint3(_177[0u]) & _555) != uint3(0u)));
        }
        if (_548 || _156[_409])
        {
            uint3 _576 = uint3(1u) << ((_171[_409].yzw - uint3(1u)) & uint3(31u));
            _177[1u] = int3(select(uint3(_177[1u]), (uint3(_177[1u]) & (_576 - uint3(1u))) - _576, (uint3(_177[1u]) & _576) != uint3(0u)));
        }
        if (_156[_409])
        {
            _177[1u] += _177[0u];
        }
        for (int _598 = 0; _598 < 2; _598++)
        {
            uint _604 = uint(_598);
            int3 _652;
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                int3 _651;
                if (_171[_409].x < 15u)
                {
                    _651 = select(_177[_604], select(((_177[_604] << int3(16)) + int3(32768)) >> (int3(_414) & int3(31)), int3(65535), _177[_604] == int3((1 << (_414 & 31)) - 1)), _177[_604] != int3(0));
                }
                else
                {
                    _651 = _177[_604];
                }
                _652 = _651;
            }
            else
            {
                int3 _634;
                if (_171[_409].x < 16u)
                {
                    int3 _616 = abs(_177[_604]);
                    int _618 = _414 - 1;
                    int3 _630 = select(_616, select(((_616 << int3(15)) + int3(16384)) >> (int3(_618) & int3(31)), int3(32767), _616 >= int3((1 << (_618 & 31)) - 1)), _616 != int3(0));
                    _634 = select(_630, -_630, select(uint3(1u), uint3(0u), _177[_604] >= int3(0)) > uint3(0u));
                }
                else
                {
                    _634 = _177[_604];
                }
                _652 = _634;
            }
            _177[_604] = _652;
        }
        float _657;
        _657 = 0.0;
        bool _655;
        float _658;
        spvUnsafeArray<int, 16> aWeight4;
        bool _654 = false;
        uint _659 = 0u;
        for (; _659 < 16u; _654 = _655, _657 = _658, _659++)
        {
            uint _664 = _187 + _659;
            float _669 = dot(_407, float3(shared_temp[_664].pixel_ph - shared_temp[_187].endPoint_low));
            int _687 = int(((_408 <= 0.0) || (_669 <= 0.0)) ? 0u : ((_669 < _408) ? _172[uint((_669 * 63.499988555908203125) / _408)] : _172[63]));
            if (!_654)
            {
                aWeight4 = _173;
            }
            _655 = _654 ? _654 : true;
            int3 _700 = (((_177[0u] * int3(64 - aWeight4[_687])) + (_177[1u] * int3(aWeight4[_687]))) + int3(32)) >> int3(6);
            int3 _718;
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                _718 = (_700 * int3(31)) >> int3(6);
            }
            else
            {
                int3 _711 = select((_700 * int3(31)) >> int3(5), -((_700 * int3(-31)) >> int3(5)), _700 < int3(0));
                _718 = select(_711, (-_711) | int3(32768), _711 < int3(0));
            }
            uint3 _719 = uint3(_718);
            float3 _732 = float3(float2(as_type<half2>(_719.x)).x, float2(as_type<half2>(_719.y)).x, float2(as_type<half2>(_719.z)).x) - shared_temp[_664].pixel_hr;
            _658 = _657 + dot(_732, _732);
        }
        shared_temp[gl_LocalInvocationIndex].error = _489 ? 100000002004087734272.0 : _657;
        shared_temp[gl_LocalInvocationIndex].best_mode = _155[_409];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_321)
    {
        uint _743 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_743].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_743].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_743].best_mode;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_347)
    {
        uint _757 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_757].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_757].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_757].best_mode;
        }
        (*spvDescriptorSet0.g_OutBuff)._m0[_186] = uint4(as_type<uint>(shared_temp[gl_LocalInvocationIndex].error), shared_temp[gl_LocalInvocationIndex].best_mode, 0u, 0u);
    }
}

