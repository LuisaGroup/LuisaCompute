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

struct type_StructuredBuffer_v4uint
{
    uint4 _m0[1];
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

struct spvDescriptorSetBuffer0
{
    type_cbCS cbCS;
    texture2d<float> g_Input;
    const device type_StructuredBuffer_v4uint* g_InBuff;
    device type_RWStructuredBuffer_v4uint* g_OutBuff;
};

constant spvUnsafeArray<bool, 14> _187 = spvUnsafeArray<bool, 14>({ true, true, true, true, true, true, true, true, true, false, false, true, true, true });
constant spvUnsafeArray<uint4, 14> _202 = spvUnsafeArray<uint4, 14>({ uint4(10u, 5u, 5u, 5u), uint4(7u, 6u, 6u, 6u), uint4(11u, 5u, 4u, 4u), uint4(11u, 4u, 5u, 4u), uint4(11u, 4u, 4u, 5u), uint4(9u, 5u, 5u, 5u), uint4(8u, 6u, 5u, 5u), uint4(8u, 5u, 6u, 5u), uint4(8u, 5u, 5u, 6u), uint4(6u), uint4(10u), uint4(11u, 9u, 9u, 9u), uint4(12u, 8u, 8u, 8u), uint4(16u, 4u, 4u, 4u) });
constant spvUnsafeArray<uint, 32> _203 = spvUnsafeArray<uint, 32>({ 52428u, 34952u, 61166u, 60616u, 51328u, 65260u, 65224u, 60544u, 51200u, 65516u, 65152u, 59392u, 65512u, 65280u, 65520u, 61440u, 63248u, 142u, 28928u, 2254u, 140u, 29456u, 12544u, 36046u, 2188u, 12560u, 26214u, 13932u, 6120u, 4080u, 29070u, 14748u });
constant spvUnsafeArray<uint, 32> _204 = spvUnsafeArray<uint, 32>({ 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 2u, 8u, 2u, 2u, 8u, 8u, 15u, 2u, 8u, 2u, 2u, 8u, 8u, 2u, 2u });
constant spvUnsafeArray<uint, 64> _205 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<uint, 64> _206 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });

kernel void EncodeBlockCS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<SharedData, 64> shared_temp;
    uint _225 = gl_LocalInvocationIndex / 32u;
    uint _231 = (spvDescriptorSet0.cbCS.g_start_block_id + (gl_WorkGroupID.x * 2u)) + _225;
    uint _232 = _225 * 32u;
    uint _233 = gl_LocalInvocationIndex - _232;
    uint _236 = _231 / spvDescriptorSet0.cbCS.g_num_block_x;
    bool _241 = _233 < 16u;
    if (_241)
    {
        int3 _249 = int3(uint3(((_231 - (_236 * spvDescriptorSet0.cbCS.g_num_block_x)) * 4u) + (_233 % 4u), (_236 * 4u) + (_233 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = spvDescriptorSet0.g_Input.read(uint2(_249.xy), _249.z).xyz;
        shared_temp[gl_LocalInvocationIndex].pixel = precise::max(shared_temp[gl_LocalInvocationIndex].pixel, float3(0.0));
        shared_temp[gl_LocalInvocationIndex].pixel_lum = dot(shared_temp[gl_LocalInvocationIndex].pixel, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
        uint3 _271 = uint3(as_type<uint>(half2(float2(shared_temp[gl_LocalInvocationIndex].pixel.x, 0.0))), as_type<uint>(half2(float2(shared_temp[gl_LocalInvocationIndex].pixel.y, 0.0))), as_type<uint>(half2(float2(shared_temp[gl_LocalInvocationIndex].pixel.z, 0.0))));
        int3 _296;
        do
        {
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                _296 = int3((_271 << uint3(6u)) / uint3(31u));
                break;
            }
            else
            {
                bool3 _281 = _271 == uint3(31743u);
                _296 = select(select(-int3(((uint3(32767u) & _271) << uint3(5u)) / uint3(31u)), int3(-32767), _281), select(int3((_271 << uint3(5u)) / uint3(31u)), int3(32767), _281), _271 < uint3(32768u));
                break;
            }
            break; // unreachable workaround
        } while(false);
        shared_temp[gl_LocalInvocationIndex].pixel_ph = _296;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _299 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_231])[1];
    uint _301 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_231])[2];
    if (_233 < 32u)
    {
        uint _305 = _233 & 15u;
        uint _306 = _232 + _305;
        float2 _354;
        int3 _355;
        int3 _356;
        if (_241)
        {
            float2 _351;
            int3 _352;
            int3 _353;
            if (_299 > 10u)
            {
                _351 = float2(shared_temp[_306].pixel_lum);
                _352 = shared_temp[_306].pixel_ph;
                _353 = shared_temp[_306].pixel_ph;
            }
            else
            {
                bool _342 = 0u == ((_203[_301] >> (_233 & 31u)) & 1u);
                float2 _346;
                if (_342)
                {
                    _346 = float2(shared_temp[_306].pixel_lum);
                }
                else
                {
                    _346 = float2(3.4028234663852885981170418348452e+38, -3.4028234663852885981170418348452e+38);
                }
                bool3 _347 = bool3(_342);
                _351 = _346;
                _352 = select(int3(int(0x80000000)), shared_temp[_306].pixel_ph, _347);
                _353 = select(int3(2147483647), shared_temp[_306].pixel_ph, _347);
            }
            _354 = _351;
            _355 = _352;
            _356 = _353;
        }
        else
        {
            float2 _330;
            int3 _331;
            int3 _332;
            if (_299 <= 10u)
            {
                bool _322 = 1u == ((_203[_301] >> (_305 & 31u)) & 1u);
                float2 _326;
                if (_322)
                {
                    _326 = float2(shared_temp[_306].pixel_lum);
                }
                else
                {
                    _326 = float2(3.4028234663852885981170418348452e+38, -3.4028234663852885981170418348452e+38);
                }
                bool3 _327 = bool3(_322);
                _330 = _326;
                _331 = select(int3(int(0x80000000)), shared_temp[_306].pixel_ph, _327);
                _332 = select(int3(2147483647), shared_temp[_306].pixel_ph, _327);
            }
            else
            {
                _330 = float2(3.4028234663852885981170418348452e+38, -3.4028234663852885981170418348452e+38);
                _331 = int3(int(0x80000000));
                _332 = int3(2147483647);
            }
            _354 = _330;
            _355 = _331;
            _356 = _332;
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _356;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _355;
        shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = _354.x;
        shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = _354.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _363 = _233 & 15u;
    if (_363 < 8u)
    {
        uint _369 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_369].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_369].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_369].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_369].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_369].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_369].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_363 < 4u)
    {
        uint _395 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_395].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_395].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_395].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_395].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_395].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_395].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_363 < 2u)
    {
        uint _421 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_421].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_421].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_421].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_421].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_421].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_421].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_363 < 1u)
    {
        uint _447 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_low > shared_temp[_447].endPoint_lum_low)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_low = shared_temp[_447].endPoint_low;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_low = shared_temp[_447].endPoint_lum_low;
        }
        if (shared_temp[gl_LocalInvocationIndex].endPoint_lum_high < shared_temp[_447].endPoint_lum_high)
        {
            shared_temp[gl_LocalInvocationIndex].endPoint_high = shared_temp[_447].endPoint_high;
            shared_temp[gl_LocalInvocationIndex].endPoint_lum_high = shared_temp[_447].endPoint_lum_high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _468 = _233 < 2u;
    if (_468)
    {
        uint _472 = _232 + (_233 * 16u);
        int3 _474 = shared_temp[_472].endPoint_low;
        int3 _476 = shared_temp[_472].endPoint_high;
        uint _484;
        if ((1u == _233) && (_299 <= 10u))
        {
            _484 = _204[_301];
        }
        else
        {
            _484 = 0u;
        }
        float3 _486 = float3(_476 - _474);
        float _487 = dot(_486, _486);
        int3 _490 = shared_temp[_232 + _484].pixel_ph;
        float _493 = dot(_486, float3(_490 - _474));
        bool3 _502 = bool3(((_487 > 0.0) && (_493 >= 0.0)) && (uint((_493 * 63.499988555908203125) / _487) > 32u));
        shared_temp[gl_LocalInvocationIndex].endPoint_low = select(_474, _476, _502);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = select(_476, _474, _502);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 _691;
    if (_241)
    {
        bool _509 = _299 > 10u;
        uint _515;
        if (_509)
        {
            _515 = 0u;
        }
        else
        {
            _515 = _203[_301];
        }
        float _548;
        float3 _549;
        if (((_515 >> (_233 & 31u)) & 1u) != 0u)
        {
            uint _535 = _232 + 1u;
            float3 _541 = float3(shared_temp[_535].endPoint_high - shared_temp[_535].endPoint_low);
            _548 = dot(_541, float3(shared_temp[gl_LocalInvocationIndex].pixel_ph - shared_temp[_535].endPoint_low));
            _549 = _541;
        }
        else
        {
            float3 _528 = float3(shared_temp[_232].endPoint_high - shared_temp[_232].endPoint_low);
            _548 = dot(_528, float3(shared_temp[gl_LocalInvocationIndex].pixel_ph - shared_temp[_232].endPoint_low));
            _549 = _528;
        }
        float _550 = dot(_549, _549);
        uint4 _685;
        if (_509)
        {
            uint _660 = ((_550 <= 0.0) || (_548 <= 0.0)) ? 0u : ((_548 < _550) ? _206[uint((_548 * 63.499988555908203125) / _550)] : _206[63]);
            uint4 _684;
            if (_233 == 0u)
            {
                uint4 _683 = uint4(0u);
                _683.z = 0u | (_660 << 1u);
                _684 = _683;
            }
            else
            {
                uint4 _680;
                if (_233 < 8u)
                {
                    uint4 _679 = uint4(0u);
                    _679.z = 0u | (_660 << ((_233 * 4u) & 31u));
                    _680 = _679;
                }
                else
                {
                    uint4 _674 = uint4(0u);
                    _674.w = 0u | (_660 << (((_233 - 8u) * 4u) & 31u));
                    _680 = _674;
                }
                _684 = _680;
            }
            _685 = _684;
        }
        else
        {
            uint _566 = ((_550 <= 0.0) || (_548 <= 0.0)) ? 0u : ((_548 < _550) ? _205[uint((_548 * 63.499988555908203125) / _550)] : _205[63]);
            int _570 = int(_204[_301] != 2u);
            uint4 _647;
            if (_233 == 0u)
            {
                uint4 _646 = uint4(0u);
                _646.z = 0u | (_566 << 18u);
                _647 = _646;
            }
            else
            {
                uint4 _643;
                if (_233 < 3u)
                {
                    uint4 _642 = uint4(0u);
                    _642.z = 0u | (_566 << ((20u + ((_233 - 1u) * 3u)) & 31u));
                    _643 = _642;
                }
                else
                {
                    uint4 _635;
                    if (_233 < 5u)
                    {
                        uint4 _634 = uint4(0u);
                        _634.z = 0u | (_566 << (((25u + ((_233 - 3u) * 3u)) + uint(_570)) & 31u));
                        _635 = _634;
                    }
                    else
                    {
                        uint4 _625;
                        if (_233 == 5u)
                        {
                            bool _613 = !(_570 != 0);
                            uint4 _618 = uint4(0u);
                            _618.w = 0u | (_566 >> (uint(_613) & 31u));
                            uint4 _624;
                            if (_613)
                            {
                                uint4 _623 = _618;
                                _623.z = 0u | (_566 << 31u);
                                _624 = _623;
                            }
                            else
                            {
                                _624 = _618;
                            }
                            _625 = _624;
                        }
                        else
                        {
                            uint4 _611;
                            if (_233 < 9u)
                            {
                                uint4 _610 = uint4(0u);
                                _610.w = 0u | (_566 << (((2u + ((_233 - 6u) * 3u)) + uint(_570)) & 31u));
                                _611 = _610;
                            }
                            else
                            {
                                uint4 _601 = uint4(0u);
                                _601.w = 0u | (_566 << (((11u + ((_233 - 9u) * 3u)) + uint(int(_204[_301] == 15u))) & 31u));
                                _611 = _601;
                            }
                            _625 = _611;
                        }
                        _635 = _625;
                    }
                    _643 = _635;
                }
                _647 = _643;
            }
            _685 = _647;
        }
        float2 _687 = as_type<float2>(_685.zw);
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(_687.x, _687.y, shared_temp[gl_LocalInvocationIndex].pixel_hr.z);
        _691 = _685;
    }
    else
    {
        _691 = uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_233 < 8u)
    {
        float2 _705 = as_type<float2>(as_type<uint2>(shared_temp[gl_LocalInvocationIndex].pixel_hr.xy) | as_type<uint2>(shared_temp[gl_LocalInvocationIndex + 8u].pixel_hr.xy));
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(_705.x, _705.y, shared_temp[gl_LocalInvocationIndex].pixel_hr.z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_233 < 4u)
    {
        float2 _721 = as_type<float2>(as_type<uint2>(shared_temp[gl_LocalInvocationIndex].pixel_hr.xy) | as_type<uint2>(shared_temp[gl_LocalInvocationIndex + 4u].pixel_hr.xy));
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(_721.x, _721.y, shared_temp[gl_LocalInvocationIndex].pixel_hr.z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_468)
    {
        float2 _736 = as_type<float2>(as_type<uint2>(shared_temp[gl_LocalInvocationIndex].pixel_hr.xy) | as_type<uint2>(shared_temp[gl_LocalInvocationIndex + 2u].pixel_hr.xy));
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(_736.x, _736.y, shared_temp[gl_LocalInvocationIndex].pixel_hr.z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 _759;
    if (_233 < 1u)
    {
        float2 _752 = as_type<float2>(as_type<uint2>(shared_temp[gl_LocalInvocationIndex].pixel_hr.xy) | as_type<uint2>(shared_temp[gl_LocalInvocationIndex + 1u].pixel_hr.xy));
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(_752.x, _752.y, shared_temp[gl_LocalInvocationIndex].pixel_hr.z);
        uint2 _757 = as_type<uint2>(shared_temp[gl_LocalInvocationIndex].pixel_hr.xy);
        _759 = uint4(_691.x, _691.y, _757.x, _757.y);
    }
    else
    {
        _759 = _691;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _760 = _299 - 1u;
    bool _762 = _187[_760];
    uint4 _764 = _202[_760];
    if (_233 == 2u)
    {
        spvUnsafeArray<int3, 2> _220;
        _220[0u] = shared_temp[_232].endPoint_low;
        _220[1u] = shared_temp[_232].endPoint_high;
        int _775 = int(_764.x);
        for (int _777 = 0; _777 < 2; _777++)
        {
            uint _783 = uint(_777);
            int3 _839;
            if (spvDescriptorSet0.cbCS.g_format == 95u)
            {
                _839 = select(select((_220[_783] << (int3(_775) & int3(31))) >> int3(16), int3((1 << (_775 & 31)) - 1), _220[_783] == int3(65535)), _220[_783], (int3(int(_775 >= 15)) | int3(_220[_783] == int3(0))) != int3(0));
            }
            else
            {
                int _801 = _775 - 1;
                int _803 = 1 << (_801 & 31);
                int3 _807 = int3(_801) & int3(31);
                int3 _811 = -_220[_783];
                _839 = select(select(select(-((_811 << _807) >> int3(15)), int3(1 - _803), _811 == int3(32767)), select((_220[_783] << _807) >> int3(15), int3(_803 - 1), _220[_783] == int3(32767)), _220[_783] >= int3(0)), _220[_783], (int3(int(_775 >= 16)) | int3(_220[_783] == int3(0))) != int3(0));
            }
            _220[_783] = _839;
        }
        if (_762)
        {
            _220[1u] -= _220[0u];
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _220[0u];
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _220[1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_233 == 3u)
    {
        uint _852 = _232 + 2u;
        uint _855 = _232 + 1u;
        spvUnsafeArray<int3, 2> _221;
        _221[0u] = shared_temp[_855].endPoint_low;
        _221[1u] = shared_temp[_855].endPoint_high;
        if (_299 <= 10u)
        {
            int _866 = int(_764.x);
            for (int _868 = 0; _868 < 2; _868++)
            {
                uint _874 = uint(_868);
                int3 _930;
                if (spvDescriptorSet0.cbCS.g_format == 95u)
                {
                    _930 = select(select((_221[_874] << (int3(_866) & int3(31))) >> int3(16), int3((1 << (_866 & 31)) - 1), _221[_874] == int3(65535)), _221[_874], (int3(int(_866 >= 15)) | int3(_221[_874] == int3(0))) != int3(0));
                }
                else
                {
                    int _892 = _866 - 1;
                    int _894 = 1 << (_892 & 31);
                    int3 _898 = int3(_892) & int3(31);
                    int3 _902 = -_221[_874];
                    _930 = select(select(select(-((_902 << _898) >> int3(15)), int3(1 - _894), _902 == int3(32767)), select((_221[_874] << _898) >> int3(15), int3(_894 - 1), _221[_874] == int3(32767)), _221[_874] >= int3(0)), _221[_874], (int3(int(_866 >= 16)) | int3(_221[_874] == int3(0))) != int3(0));
                }
                _221[_874] = _930;
            }
            if (_762)
            {
                _221[0u] -= shared_temp[_852].endPoint_low;
                _221[1u] -= shared_temp[_852].endPoint_low;
            }
            shared_temp[gl_LocalInvocationIndex].endPoint_low = _221[0u];
            shared_temp[gl_LocalInvocationIndex].endPoint_high = _221[1u];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_468)
    {
        uint _943 = gl_LocalInvocationIndex + 2u;
        spvUnsafeArray<int3, 2> _222;
        _222[0u] = shared_temp[_943].endPoint_low;
        _222[1u] = shared_temp[_943].endPoint_high;
        if (_233 == 0u)
        {
            if (_299 > 10u)
            {
                for (int _1082 = 0; _1082 < 2; _1082++)
                {
                    uint _1088 = uint(_1082);
                    int3 _1090 = _222[_1088];
                    int3 _1136;
                    if (_762)
                    {
                        int3 _1110 = _1090;
                        _1110.x = int(uint(_1090.x) & ((1u << (_764.x & 31u)) - 1u));
                        uint3 _1115 = uint3(uint(_1090.y));
                        uint3 _1119 = uint3(1u) << ((_764.yzw - uint3(1u)) & uint3(31u));
                        _1110.y = int(select(select(_1115 & ((uint3(1u) << (_764.yzw & uint3(31u))) - uint3(1u)), _1119, uint3(uint(-_1090.y)) > _1119), select(_1115, _1119 - uint3(1u), _1115 >= _1119), bool3(_1090.y >= 0)).x);
                        _1136 = _1110;
                    }
                    else
                    {
                        _1136 = int3(uint3(_1090) & uint3((1u << (_764.x & 31u)) - 1u));
                    }
                    _222[_1088] = _1136;
                }
            }
            else
            {
                for (int _1026 = 0; _1026 < 2; _1026++)
                {
                    uint _1032 = uint(_1026);
                    int3 _1034 = _222[_1032];
                    int3 _1080;
                    if (_762)
                    {
                        int3 _1054 = _1034;
                        _1054.x = int(uint(_1034.x) & ((1u << (_764.x & 31u)) - 1u));
                        uint3 _1059 = uint3(uint(_1034.y));
                        uint3 _1063 = uint3(1u) << ((_764.yzw - uint3(1u)) & uint3(31u));
                        _1054.y = int(select(select(_1059 & ((uint3(1u) << (_764.yzw & uint3(31u))) - uint3(1u)), _1063, uint3(uint(-_1034.y)) > _1063), select(_1059, _1063 - uint3(1u), _1059 >= _1063), bool3(_1034.y >= 0)).x);
                        _1080 = _1054;
                    }
                    else
                    {
                        _1080 = int3(uint3(_1034) & uint3((1u << (_764.x & 31u)) - 1u));
                    }
                    _222[_1032] = _1080;
                }
            }
        }
        else
        {
            if (_299 <= 10u)
            {
                for (int _958 = 0; _958 < 2; _958++)
                {
                    uint _964 = uint(_958);
                    int3 _966 = _222[_964];
                    int3 _1020;
                    if (_762)
                    {
                        uint3 _982 = uint3(uint(_966.x));
                        uint3 _986 = uint3(1u) << ((_764.yzw - uint3(1u)) & uint3(31u));
                        uint3 _988 = _986 - uint3(1u);
                        uint3 _996 = (uint3(1u) << (_764.yzw & uint3(31u))) - uint3(1u);
                        int3 _1002 = _966;
                        _1002.x = int(select(select(_982 & _996, _986, uint3(uint(-_966.x)) > _986), select(_982, _988, _982 >= _986), bool3(_966.x >= 0)).x);
                        uint3 _1007 = uint3(uint(_966.y));
                        _1002.y = int(select(select(_1007 & _996, _986, uint3(uint(-_966.y)) > _986), select(_1007, _988, _1007 >= _986), bool3(_966.y >= 0)).x);
                        _1020 = _1002;
                    }
                    else
                    {
                        _1020 = int3(uint3(_966) & uint3((1u << (_764.x & 31u)) - 1u));
                    }
                    _222[_964] = _1020;
                }
            }
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _222[0u];
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _222[1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_233 == 0u)
    {
        uint _1148 = _232 + 1u;
        uint4 _6449;
        if (_299 > 10u)
        {
            uint4 _5190 = uint4(uint2(0u).x, uint2(0u).y, _759.z, _759.w);
            uint _5192 = _759.z & 4294967294u;
            _5190.z = _5192;
            uint4 _6448;
            if (_299 == 11u)
            {
                uint _6205 = ((((((((((((3u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                uint _6271 = ((((((((((((_6205 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                uint4 _6277 = _5190;
                _6277.x = _6271 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                uint _6343 = ((((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_232].endPoint_high.x >> 6) & 1) << 9)) | uint(((shared_temp[_232].endPoint_high.x >> 7) & 1) << 10)) | uint(((shared_temp[_232].endPoint_high.x >> 8) & 1) << 11)) | uint(((shared_temp[_232].endPoint_high.x >> 9) & 1) << 12);
                uint _6410 = ((((((((((((_6343 | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_232].endPoint_high.y >> 6) & 1) << 19)) | uint(((shared_temp[_232].endPoint_high.y >> 7) & 1) << 20)) | uint(((shared_temp[_232].endPoint_high.y >> 8) & 1) << 21)) | uint(((shared_temp[_232].endPoint_high.y >> 9) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25);
                _6277.y = (((((_6410 | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_232].endPoint_high.z >> 6) & 1) << 29)) | uint(((shared_temp[_232].endPoint_high.z >> 7) & 1) << 30)) | uint(((shared_temp[_232].endPoint_high.z >> 8) & 1) << 31);
                _6277.z = _5192 | uint(((shared_temp[_232].endPoint_high.z >> 9) & 1) << 0);
                _6448 = _6277;
            }
            else
            {
                uint4 _6138;
                if (_299 == 12u)
                {
                    uint _5895 = ((((((((((((7u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                    uint _5961 = ((((((((((((_5895 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                    uint4 _5967 = _5190;
                    _5967.x = _5961 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                    uint _6033 = ((((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_232].endPoint_high.x >> 6) & 1) << 9)) | uint(((shared_temp[_232].endPoint_high.x >> 7) & 1) << 10)) | uint(((shared_temp[_232].endPoint_high.x >> 8) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 12);
                    uint _6100 = ((((((((((((_6033 | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_232].endPoint_high.y >> 6) & 1) << 19)) | uint(((shared_temp[_232].endPoint_high.y >> 7) & 1) << 20)) | uint(((shared_temp[_232].endPoint_high.y >> 8) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25);
                    _5967.y = (((((_6100 | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_232].endPoint_high.z >> 6) & 1) << 29)) | uint(((shared_temp[_232].endPoint_high.z >> 7) & 1) << 30)) | uint(((shared_temp[_232].endPoint_high.z >> 8) & 1) << 31);
                    _5967.z = _5192 | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 0);
                    _6138 = _5967;
                }
                else
                {
                    uint4 _5828;
                    if (_299 == 13u)
                    {
                        uint _5585 = ((((((((((((11u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                        uint _5651 = ((((((((((((_5585 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                        uint4 _5657 = _5190;
                        _5657.x = _5651 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                        uint _5723 = ((((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_232].endPoint_high.x >> 6) & 1) << 9)) | uint(((shared_temp[_232].endPoint_high.x >> 7) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 11) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 12);
                        uint _5790 = ((((((((((((_5723 | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_232].endPoint_high.y >> 6) & 1) << 19)) | uint(((shared_temp[_232].endPoint_high.y >> 7) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 11) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25);
                        _5657.y = (((((_5790 | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_232].endPoint_high.z >> 6) & 1) << 29)) | uint(((shared_temp[_232].endPoint_high.z >> 7) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 11) & 1) << 31);
                        _5657.z = _5192 | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 0);
                        _5828 = _5657;
                    }
                    else
                    {
                        uint4 _5518;
                        if (_299 == 14u)
                        {
                            uint _5275 = ((((((((((((15u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                            uint _5341 = ((((((((((((_5275 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                            uint4 _5347 = _5190;
                            _5347.x = _5341 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                            uint _5413 = ((((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 15) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 14) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 13) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 12) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 11) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 12);
                            uint _5480 = ((((((((((((_5413 | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 15) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 14) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 13) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 12) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 11) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25);
                            _5347.y = (((((_5480 | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 15) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 14) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 13) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 12) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 11) & 1) << 31);
                            _5347.z = _5192 | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 0);
                            _5518 = _5347;
                        }
                        else
                        {
                            _5518 = _5190;
                        }
                        _5828 = _5518;
                    }
                    _6138 = _5828;
                }
                _6448 = _6138;
            }
            _6449 = _6448;
        }
        else
        {
            uint4 _1157 = uint4(uint2(0u).x, uint2(0u).y, _759.z, _759.w);
            uint _1159 = _759.z & 4294705152u;
            _1157.z = _1159;
            uint4 _5189;
            if (_299 == 1u)
            {
                uint _4842 = (((((((((((0u | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 4)) | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13);
                uint _4909 = ((((((((((((_4842 | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26);
                uint4 _4935 = _1157;
                _4935.x = ((((_4909 | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                uint _4997 = (((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                uint _5059 = (((((((((((_4997 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23);
                _4935.y = (((((((_5059 | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                uint _5162 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11);
                _4935.z = (((((_5162 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                _5189 = _4935;
            }
            else
            {
                uint4 _4778;
                if (_299 == 2u)
                {
                    uint _4431 = (((((((((((1u | uint(((shared_temp[_1148].endPoint_low.y >> 5) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_high.y >> 5) & 1) << 4)) | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 12)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 13);
                    uint _4494 = (((((((((((_4431 | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_low.z >> 5) & 1) << 22)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25);
                    uint4 _4525 = _1157;
                    _4525.x = (((((_4494 | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                    uint _4586 = (((((((((((0u | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_high.z >> 5) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                    uint _4648 = (((((((((((_4586 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23);
                    _4525.y = (((((((_4648 | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                    uint _4756 = ((((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_low.x >> 5) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_high.x >> 5) & 1) << 12);
                    _4525.z = ((((_4756 | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                    _4778 = _4525;
                }
                else
                {
                    uint4 _4367;
                    if (_299 == 3u)
                    {
                        uint _4038 = ((((((((((((2u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                        uint _4104 = ((((((((((((_4038 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                        uint4 _4110 = _1157;
                        _4110.x = _4104 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                        uint _4172 = (((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                        uint _4230 = ((((((((((_4172 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22);
                        _4110.y = ((((((((_4230 | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                        uint _4340 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11);
                        _4110.z = (((((_4340 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                        _4367 = _4110;
                    }
                    else
                    {
                        uint4 _3971;
                        if (_299 == 4u)
                        {
                            uint _3642 = ((((((((((((6u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                            uint _3708 = ((((((((((((_3642 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                            uint4 _3714 = _1157;
                            _3714.x = _3708 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                            uint _3772 = ((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10);
                            uint _3828 = ((((((((((_3772 | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21);
                            _3714.y = (((((((((_3828 | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                            uint _3944 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 11);
                            _3714.z = (((((_3944 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                            _3971 = _3714;
                        }
                        else
                        {
                            uint4 _3575;
                            if (_299 == 5u)
                            {
                                uint _3246 = ((((((((((((10u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_232].endPoint_low.x >> 9) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                                uint _3312 = ((((((((((((_3246 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_232].endPoint_low.y >> 9) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30);
                                uint4 _3318 = _1157;
                                _3318.x = _3312 | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                                uint _3381 = (((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_232].endPoint_low.z >> 9) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 10) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                                uint _3439 = ((((((((((_3381 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 10) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22);
                                _3318.y = ((((((((_3439 | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 10) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                uint _3548 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 11);
                                _3318.z = (((((_3548 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                _3575 = _3318;
                            }
                            else
                            {
                                uint4 _3179;
                                if (_299 == 6u)
                                {
                                    uint _2851 = ((((((((((((14u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_232].endPoint_low.x >> 8) & 1) << 13)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16)) | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17);
                                    uint _2913 = (((((((((((_2851 | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_232].endPoint_low.y >> 8) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28)) | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29);
                                    uint4 _2924 = _1157;
                                    _2924.x = (_2913 | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                                    uint _2982 = ((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_232].endPoint_low.z >> 8) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10);
                                    uint _3043 = (((((((((((_2982 | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22);
                                    _2924.y = ((((((((_3043 | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                    uint _3152 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11);
                                    _2924.z = (((((_3152 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                    _3179 = _2924;
                                }
                                else
                                {
                                    uint4 _2783;
                                    if (_299 == 7u)
                                    {
                                        uint _2451 = (((((((((((18u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 13)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16);
                                        uint _2514 = (((((((((((_2451 | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28);
                                        uint4 _2530 = _1157;
                                        _2530.x = ((_2514 | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                                        uint _2591 = (((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                                        uint _2653 = (((((((((((_2591 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23);
                                        _2530.y = (((((((_2653 | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                        uint _2761 = ((((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_low.x >> 5) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_high.x >> 5) & 1) << 12);
                                        _2530.z = ((((_2761 | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                        _2783 = _2530;
                                    }
                                    else
                                    {
                                        uint4 _2387;
                                        if (_299 == 8u)
                                        {
                                            uint _2055 = (((((((((((22u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 13)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16);
                                            uint _2117 = (((((((((((_2055 | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_1148].endPoint_low.y >> 5) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28);
                                            uint4 _2133 = _1157;
                                            _2133.x = ((_2117 | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                                            uint _2190 = ((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_high.y >> 5) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10);
                                            uint _2251 = (((((((((((_2190 | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22);
                                            _2133.y = ((((((((_2251 | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                            uint _2360 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11);
                                            _2133.z = (((((_2360 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                            _2387 = _2133;
                                        }
                                        else
                                        {
                                            uint4 _1991;
                                            if (_299 == 9u)
                                            {
                                                uint _1659 = (((((((((((26u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_232].endPoint_low.x >> 6) & 1) << 11)) | uint(((shared_temp[_232].endPoint_low.x >> 7) & 1) << 12)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 13)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16);
                                                uint _1721 = (((((((((((_1659 | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_232].endPoint_low.y >> 6) & 1) << 21)) | uint(((shared_temp[_232].endPoint_low.y >> 7) & 1) << 22)) | uint(((shared_temp[_1148].endPoint_low.z >> 5) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28);
                                                uint4 _1737 = _1157;
                                                _1737.x = ((_1721 | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_232].endPoint_low.z >> 6) & 1) << 31);
                                                uint _1794 = ((((((((((0u | uint(((shared_temp[_232].endPoint_low.z >> 7) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_high.z >> 5) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10);
                                                uint _1855 = (((((((((((_1794 | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22);
                                                _1737.y = ((((((((_1855 | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23)) | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                                uint _1964 = (((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11);
                                                _1737.z = (((((_1964 | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 12)) | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                                _1991 = _1737;
                                            }
                                            else
                                            {
                                                uint4 _1595;
                                                if (_299 == 10u)
                                                {
                                                    uint _1264 = (((((((((((30u | uint(((shared_temp[_232].endPoint_low.x >> 0) & 1) << 5)) | uint(((shared_temp[_232].endPoint_low.x >> 1) & 1) << 6)) | uint(((shared_temp[_232].endPoint_low.x >> 2) & 1) << 7)) | uint(((shared_temp[_232].endPoint_low.x >> 3) & 1) << 8)) | uint(((shared_temp[_232].endPoint_low.x >> 4) & 1) << 9)) | uint(((shared_temp[_232].endPoint_low.x >> 5) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.y >> 4) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_high.z >> 0) & 1) << 12)) | uint(((shared_temp[_1148].endPoint_high.z >> 1) & 1) << 13)) | uint(((shared_temp[_1148].endPoint_low.z >> 4) & 1) << 14)) | uint(((shared_temp[_232].endPoint_low.y >> 0) & 1) << 15)) | uint(((shared_temp[_232].endPoint_low.y >> 1) & 1) << 16);
                                                    uint _1326 = (((((((((((_1264 | uint(((shared_temp[_232].endPoint_low.y >> 2) & 1) << 17)) | uint(((shared_temp[_232].endPoint_low.y >> 3) & 1) << 18)) | uint(((shared_temp[_232].endPoint_low.y >> 4) & 1) << 19)) | uint(((shared_temp[_232].endPoint_low.y >> 5) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_low.y >> 5) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_low.z >> 5) & 1) << 22)) | uint(((shared_temp[_1148].endPoint_high.z >> 2) & 1) << 23)) | uint(((shared_temp[_1148].endPoint_low.y >> 4) & 1) << 24)) | uint(((shared_temp[_232].endPoint_low.z >> 0) & 1) << 25)) | uint(((shared_temp[_232].endPoint_low.z >> 1) & 1) << 26)) | uint(((shared_temp[_232].endPoint_low.z >> 2) & 1) << 27)) | uint(((shared_temp[_232].endPoint_low.z >> 3) & 1) << 28);
                                                    uint4 _1342 = _1157;
                                                    _1342.x = ((_1326 | uint(((shared_temp[_232].endPoint_low.z >> 4) & 1) << 29)) | uint(((shared_temp[_232].endPoint_low.z >> 5) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_high.y >> 5) & 1) << 31);
                                                    uint _1403 = (((((((((((0u | uint(((shared_temp[_1148].endPoint_high.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_high.z >> 5) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_high.z >> 4) & 1) << 2)) | uint(((shared_temp[_232].endPoint_high.x >> 0) & 1) << 3)) | uint(((shared_temp[_232].endPoint_high.x >> 1) & 1) << 4)) | uint(((shared_temp[_232].endPoint_high.x >> 2) & 1) << 5)) | uint(((shared_temp[_232].endPoint_high.x >> 3) & 1) << 6)) | uint(((shared_temp[_232].endPoint_high.x >> 4) & 1) << 7)) | uint(((shared_temp[_232].endPoint_high.x >> 5) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_low.y >> 0) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_low.y >> 1) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_low.y >> 2) & 1) << 11);
                                                    uint _1465 = (((((((((((_1403 | uint(((shared_temp[_1148].endPoint_low.y >> 3) & 1) << 12)) | uint(((shared_temp[_232].endPoint_high.y >> 0) & 1) << 13)) | uint(((shared_temp[_232].endPoint_high.y >> 1) & 1) << 14)) | uint(((shared_temp[_232].endPoint_high.y >> 2) & 1) << 15)) | uint(((shared_temp[_232].endPoint_high.y >> 3) & 1) << 16)) | uint(((shared_temp[_232].endPoint_high.y >> 4) & 1) << 17)) | uint(((shared_temp[_232].endPoint_high.y >> 5) & 1) << 18)) | uint(((shared_temp[_1148].endPoint_high.y >> 0) & 1) << 19)) | uint(((shared_temp[_1148].endPoint_high.y >> 1) & 1) << 20)) | uint(((shared_temp[_1148].endPoint_high.y >> 2) & 1) << 21)) | uint(((shared_temp[_1148].endPoint_high.y >> 3) & 1) << 22)) | uint(((shared_temp[_232].endPoint_high.z >> 0) & 1) << 23);
                                                    _1342.y = (((((((_1465 | uint(((shared_temp[_232].endPoint_high.z >> 1) & 1) << 24)) | uint(((shared_temp[_232].endPoint_high.z >> 2) & 1) << 25)) | uint(((shared_temp[_232].endPoint_high.z >> 3) & 1) << 26)) | uint(((shared_temp[_232].endPoint_high.z >> 4) & 1) << 27)) | uint(((shared_temp[_232].endPoint_high.z >> 5) & 1) << 28)) | uint(((shared_temp[_1148].endPoint_low.z >> 0) & 1) << 29)) | uint(((shared_temp[_1148].endPoint_low.z >> 1) & 1) << 30)) | uint(((shared_temp[_1148].endPoint_low.z >> 2) & 1) << 31);
                                                    uint _1573 = ((((((((((((_1159 | uint(((shared_temp[_1148].endPoint_low.z >> 3) & 1) << 0)) | uint(((shared_temp[_1148].endPoint_low.x >> 0) & 1) << 1)) | uint(((shared_temp[_1148].endPoint_low.x >> 1) & 1) << 2)) | uint(((shared_temp[_1148].endPoint_low.x >> 2) & 1) << 3)) | uint(((shared_temp[_1148].endPoint_low.x >> 3) & 1) << 4)) | uint(((shared_temp[_1148].endPoint_low.x >> 4) & 1) << 5)) | uint(((shared_temp[_1148].endPoint_low.x >> 5) & 1) << 6)) | uint(((shared_temp[_1148].endPoint_high.x >> 0) & 1) << 7)) | uint(((shared_temp[_1148].endPoint_high.x >> 1) & 1) << 8)) | uint(((shared_temp[_1148].endPoint_high.x >> 2) & 1) << 9)) | uint(((shared_temp[_1148].endPoint_high.x >> 3) & 1) << 10)) | uint(((shared_temp[_1148].endPoint_high.x >> 4) & 1) << 11)) | uint(((shared_temp[_1148].endPoint_high.x >> 5) & 1) << 12);
                                                    _1342.z = ((((_1573 | (((_301 >> 0u) & 1u) << 13u)) | (((_301 >> 1u) & 1u) << 14u)) | (((_301 >> 2u) & 1u) << 15u)) | (((_301 >> 3u) & 1u) << 16u)) | (((_301 >> 4u) & 1u) << 17u);
                                                    _1595 = _1342;
                                                }
                                                else
                                                {
                                                    _1595 = _1157;
                                                }
                                                _1991 = _1595;
                                            }
                                            _2387 = _1991;
                                        }
                                        _2783 = _2387;
                                    }
                                    _3179 = _2783;
                                }
                                _3575 = _3179;
                            }
                            _3971 = _3575;
                        }
                        _4367 = _3971;
                    }
                    _4778 = _4367;
                }
                _5189 = _4778;
            }
            _6449 = _5189;
        }
        (*spvDescriptorSet0.g_OutBuff)._m0[_231] = _6449;
    }
}

