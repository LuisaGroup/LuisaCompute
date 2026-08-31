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

constant bool3 _212 = {};

struct spvDescriptorSetBuffer0
{
    constant type_cbCS* cbCS [[id(0)]];
    texture2d<float> g_Input [[id(1)]];
    const device type_StructuredBuffer_v4uint* g_InBuff [[id(2)]];
    device type_RWStructuredBuffer_v4uint* g_OutBuff [[id(3)]];
};

constant spvUnsafeArray<uint, 14> _183 = spvUnsafeArray<uint, 14>({ 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u });
constant spvUnsafeArray<bool, 14> _184 = spvUnsafeArray<bool, 14>({ true, true, true, true, true, true, true, true, true, false, false, true, true, true });
constant spvUnsafeArray<uint4, 14> _199 = spvUnsafeArray<uint4, 14>({ uint4(10u, 5u, 5u, 5u), uint4(7u, 6u, 6u, 6u), uint4(11u, 5u, 4u, 4u), uint4(11u, 4u, 5u, 4u), uint4(11u, 4u, 4u, 5u), uint4(9u, 5u, 5u, 5u), uint4(8u, 6u, 5u, 5u), uint4(8u, 5u, 6u, 5u), uint4(8u, 5u, 5u, 6u), uint4(6u), uint4(10u), uint4(11u, 9u, 9u, 9u), uint4(12u, 8u, 8u, 8u), uint4(16u, 4u, 4u, 4u) });
constant spvUnsafeArray<uint, 32> _200 = spvUnsafeArray<uint, 32>({ 52428u, 34952u, 61166u, 60616u, 51328u, 65260u, 65224u, 60544u, 51200u, 65516u, 65152u, 59392u, 65512u, 65280u, 65520u, 61440u, 63248u, 142u, 28928u, 2254u, 140u, 29456u, 12544u, 36046u, 2188u, 12560u, 26214u, 13932u, 6120u, 4080u, 29070u, 14748u });
constant spvUnsafeArray<uint, 32> _201 = spvUnsafeArray<uint, 32>({ 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 2u, 8u, 2u, 2u, 8u, 8u, 15u, 2u, 8u, 2u, 2u, 8u, 8u, 2u, 2u });
constant spvUnsafeArray<uint, 64> _202 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<int, 8> _203 = spvUnsafeArray<int, 8>({ 0, 9, 18, 27, 37, 46, 55, 64 });

kernel void TryModeLE10CS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<SharedData, 64> shared_temp;
    uint _219 = gl_LocalInvocationIndex / 32u;
    uint _225 = ((*spvDescriptorSet0.cbCS).g_start_block_id + (gl_WorkGroupID.x * 2u)) + _219;
    uint _226 = _219 * 32u;
    uint _227 = gl_LocalInvocationIndex - _226;
    uint _230 = _225 / (*spvDescriptorSet0.cbCS).g_num_block_x;
    bool _235 = _227 < 16u;
    if (_235)
    {
        int3 _243 = int3(uint3(((_225 - (_230 * (*spvDescriptorSet0.cbCS).g_num_block_x)) * 4u) + (_227 % 4u), (_230 * 4u) + (_227 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = spvDescriptorSet0.g_Input.read(uint2(_243.xy), _243.z).xyz;
        shared_temp[gl_LocalInvocationIndex].pixel = precise::max(shared_temp[gl_LocalInvocationIndex].pixel, float3(0.0));
        float3 _252 = shared_temp[gl_LocalInvocationIndex].pixel;
        uint _255 = as_type<uint>(half2(float2(_252.x, 0.0)));
        uint _258 = as_type<uint>(half2(float2(_252.y, 0.0)));
        uint _261 = as_type<uint>(half2(float2(_252.z, 0.0)));
        uint3 _262 = uint3(_255, _258, _261);
        shared_temp[gl_LocalInvocationIndex].pixel_hr = float3(float2(as_type<half2>(_255)).x, float2(as_type<half2>(_258)).x, float2(as_type<half2>(_261)).x);
        shared_temp[gl_LocalInvocationIndex].pixel_lum = dot(shared_temp[gl_LocalInvocationIndex].pixel_hr, float3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875));
        int3 _298;
        do
        {
            if ((*spvDescriptorSet0.cbCS).g_format == 95u)
            {
                _298 = int3((_262 << uint3(6u)) / uint3(31u));
                break;
            }
            else
            {
                bool3 _283 = _262 == uint3(31743u);
                _298 = select(select(-int3(((uint3(32767u) & _262) << uint3(5u)) / uint3(31u)), int3(-32767), _283), select(int3((_262 << uint3(5u)) / uint3(31u)), int3(32767), _283), _262 < uint3(32768u));
                break;
            }
            break; // unreachable workaround
        } while(false);
        shared_temp[gl_LocalInvocationIndex].pixel_ph = _298;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_227 < 32u)
    {
        float2 _306;
        float2 _309;
        int3 _311;
        int3 _313;
        int3 _315;
        int3 _317;
        _306 = float2(3.4028234663852885981170418348452e+38, -3.4028234663852885981170418348452e+38);
        _309 = float2(3.4028234663852885981170418348452e+38, -3.4028234663852885981170418348452e+38);
        _311 = int3(2147483647);
        _313 = int3(int(0x80000000));
        _315 = int3(2147483647);
        _317 = int3(int(0x80000000));
        float2 _307;
        float2 _310;
        int3 _312;
        int3 _314;
        int3 _316;
        int3 _318;
        for (uint _319 = 0u; _319 < 16u; _306 = _307, _309 = _310, _311 = _312, _313 = _314, _315 = _316, _317 = _318, _319++)
        {
            uint _324 = _226 + _319;
            if (((_200[_227] >> (_319 & 31u)) & 1u) != 0u)
            {
                bool _353 = _306.x > shared_temp[_324].pixel_lum;
                float2 _357;
                if (_353)
                {
                    float2 _356 = _306;
                    _356.x = shared_temp[_324].pixel_lum;
                    _357 = _356;
                }
                else
                {
                    _357 = _306;
                }
                bool _361 = _357.y < shared_temp[_324].pixel_lum;
                float2 _365;
                if (_361)
                {
                    float2 _364 = _357;
                    _364.y = shared_temp[_324].pixel_lum;
                    _365 = _364;
                }
                else
                {
                    _365 = _357;
                }
                _307 = _365;
                _310 = _309;
                _312 = select(_311, shared_temp[_324].pixel_ph, bool3(_353));
                _314 = select(_313, shared_temp[_324].pixel_ph, bool3(_361));
                _316 = _315;
                _318 = _317;
            }
            else
            {
                bool _337 = _309.x > shared_temp[_324].pixel_lum;
                float2 _341;
                if (_337)
                {
                    float2 _340 = _309;
                    _340.x = shared_temp[_324].pixel_lum;
                    _341 = _340;
                }
                else
                {
                    _341 = _309;
                }
                bool _345 = _341.y < shared_temp[_324].pixel_lum;
                float2 _349;
                if (_345)
                {
                    float2 _348 = _341;
                    _348.y = shared_temp[_324].pixel_lum;
                    _349 = _348;
                }
                else
                {
                    _349 = _341;
                }
                _307 = _306;
                _310 = _349;
                _312 = _311;
                _314 = _313;
                _316 = select(_315, shared_temp[_324].pixel_ph, bool3(_337));
                _318 = select(_317, shared_temp[_324].pixel_ph, bool3(_345));
            }
        }
        float3 _369 = float3(_317 - _315);
        float _370 = dot(_369, _369);
        float _375 = dot(_369, float3(shared_temp[_226].pixel_ph - _315));
        bool _383 = ((_370 > 0.0) && (_375 >= 0.0)) && (uint((_375 * 63.499988555908203125) / _370) > 32u);
        float3 _387;
        if (_383)
        {
            _387 = -_369;
        }
        else
        {
            _387 = _369;
        }
        bool3 _388 = bool3(_383);
        int3 _389 = select(_315, _317, _388);
        float3 _392 = float3(_313 - _311);
        float _393 = dot(_392, _392);
        float _401 = dot(_392, float3(shared_temp[_226 + _201[_227]].pixel_ph - _311));
        bool _409 = ((_393 > 0.0) && (_401 >= 0.0)) && (uint((_401 * 63.499988555908203125) / _393) > 32u);
        float3 _413;
        if (_409)
        {
            _413 = -_392;
        }
        else
        {
            _413 = _392;
        }
        bool3 _414 = bool3(_409);
        int3 _415 = select(_311, _313, _414);
        spvUnsafeArray<int3, 2> _421 = spvUnsafeArray<int3, 2>({ _415, select(_313, _311, _414) });
        spvUnsafeArray<int3, 2> _422 = spvUnsafeArray<int3, 2>({ _389, select(_317, _315, _388) });
        spvUnsafeArray<int3, 2> _216 = _422;
        spvUnsafeArray<int3, 2> _215 = _421;
        int _424 = int(_199[(*spvDescriptorSet0.cbCS).g_mode_id].x);
        for (int _426 = 0; _426 < 2; _426++)
        {
            uint _432 = uint(_426);
            int3 _488;
            if ((*spvDescriptorSet0.cbCS).g_format == 95u)
            {
                _488 = select(select((_216[_432] << (int3(_424) & int3(31))) >> int3(16), int3((1 << (_424 & 31)) - 1), _216[_432] == int3(65535)), _216[_432], (int3(int(_424 >= 15)) | int3(_216[_432] == int3(0))) != int3(0));
            }
            else
            {
                int _450 = _424 - 1;
                int _452 = 1 << (_450 & 31);
                int3 _456 = int3(_450) & int3(31);
                int3 _460 = -_216[_432];
                _488 = select(select(select(-((_460 << _456) >> int3(15)), int3(1 - _452), _460 == int3(32767)), select((_216[_432] << _456) >> int3(15), int3(_452 - 1), _216[_432] == int3(32767)), _216[_432] >= int3(0)), _216[_432], (int3(int(_424 >= 16)) | int3(_216[_432] == int3(0))) != int3(0));
            }
            _216[_432] = _488;
        }
        for (int _490 = 0; _490 < 2; _490++)
        {
            uint _496 = uint(_490);
            int3 _552;
            if ((*spvDescriptorSet0.cbCS).g_format == 95u)
            {
                _552 = select(select((_215[_496] << (int3(_424) & int3(31))) >> int3(16), int3((1 << (_424 & 31)) - 1), _215[_496] == int3(65535)), _215[_496], (int3(int(_424 >= 15)) | int3(_215[_496] == int3(0))) != int3(0));
            }
            else
            {
                int _514 = _424 - 1;
                int _516 = 1 << (_514 & 31);
                int3 _520 = int3(_514) & int3(31);
                int3 _524 = -_215[_496];
                _552 = select(select(select(-((_524 << _520) >> int3(15)), int3(1 - _516), _524 == int3(32767)), select((_215[_496] << _520) >> int3(15), int3(_516 - 1), _215[_496] == int3(32767)), _215[_496] >= int3(0)), _215[_496], (int3(int(_424 >= 16)) | int3(_215[_496] == int3(0))) != int3(0));
            }
            _215[_496] = _552;
        }
        if (_184[(*spvDescriptorSet0.cbCS).g_mode_id])
        {
            _216[1u] -= _216[0u];
            _215[0u] -= _216[0u];
            _215[1u] -= _216[0u];
        }
        int _571;
        _571 = 0;
        int _572;
        for (int _574 = 0; _574 < 2; _571 = _572, _574++)
        {
            uint _579 = uint(_574);
            int3 _581 = _216[_579];
            int3 _629;
            if (_184[(*spvDescriptorSet0.cbCS).g_mode_id])
            {
                bool3 _594 = bool3(_581.y >= 0);
                uint3 _596 = uint3(uint(_581.y));
                uint3 _600 = uint3(1u) << ((_199[(*spvDescriptorSet0.cbCS).g_mode_id].yzw - uint3(1u)) & uint3(31u));
                bool3 _601 = _596 >= _600;
                bool3 _605 = uint3(uint(-_581.y)) > _600;
                int3 _617 = _581;
                _617.x = int(uint(_581.x) & ((1u << (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x & 31u)) - 1u));
                _617.y = int(select(select(_596 & ((uint3(1u) << (_199[(*spvDescriptorSet0.cbCS).g_mode_id].yzw & uint3(31u))) - uint3(1u)), _600, _605), select(_596, _600 - uint3(1u), _601), _594).x);
                _629 = _617;
                _572 = _571 | int(any(select(_605, _601, _594)));
            }
            else
            {
                _629 = int3(uint3(_581) & uint3((1u << (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x & 31u)) - 1u));
                _572 = _571;
            }
            _216[_579] = _629;
        }
        bool3 _631;
        int _634;
        _631 = _212;
        _634 = _571;
        bool3 _632;
        int _635;
        for (int _636 = 0; _636 < 2; _631 = _632, _634 = _635, _636++)
        {
            uint _641 = uint(_636);
            int3 _643 = _215[_641];
            int3 _705;
            if (_184[(*spvDescriptorSet0.cbCS).g_mode_id])
            {
                bool3 _656 = bool3(_643.x >= 0);
                uint3 _658 = uint3(uint(_643.x));
                uint3 _662 = uint3(1u) << ((_199[(*spvDescriptorSet0.cbCS).g_mode_id].yzw - uint3(1u)) & uint3(31u));
                bool3 _663 = _658 >= _662;
                bool3 _667 = uint3(uint(-_643.x)) > _662;
                bool3 _670 = _631;
                _670.x = select(_667, _663, _656).x;
                bool3 _673 = bool3(_643.y >= 0);
                uint3 _675 = uint3(uint(_643.y));
                bool3 _676 = _675 >= _662;
                bool3 _680 = uint3(uint(-_643.y)) > _662;
                _670.y = select(_680, _676, _673).x;
                uint3 _687 = _662 - uint3(1u);
                uint3 _691 = (uint3(1u) << (_199[(*spvDescriptorSet0.cbCS).g_mode_id].yzw & uint3(31u))) - uint3(1u);
                int3 _697 = _643;
                _697.x = int(select(select(_658 & _691, _662, _667), select(_658, _687, _663), _656).x);
                _697.y = int(select(select(_675 & _691, _662, _680), select(_675, _687, _676), _673).x);
                _632 = _670;
                _705 = _697;
                _635 = _634 | int(any(_670));
            }
            else
            {
                _632 = _631;
                _705 = int3(uint3(_643) & uint3((1u << (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x & 31u)) - 1u));
                _635 = _634;
            }
            _215[_641] = _705;
        }
        bool _708 = (*spvDescriptorSet0.cbCS).g_format == 96u;
        if (_708)
        {
            uint3 _715 = uint3(1u) << ((uint3(_199[(*spvDescriptorSet0.cbCS).g_mode_id].x) - uint3(1u)) & uint3(31u));
            _216[0u] = int3(select(uint3(_216[0u]), (uint3(_216[0u]) & (_715 - uint3(1u))) - _715, (uint3(_216[0u]) & _715) != uint3(0u)));
        }
        if (_708 || _184[(*spvDescriptorSet0.cbCS).g_mode_id])
        {
            uint3 _736 = uint3(1u) << ((_199[(*spvDescriptorSet0.cbCS).g_mode_id].yzw - uint3(1u)) & uint3(31u));
            uint3 _743 = _736 - uint3(1u);
            _216[1u] = int3(select(uint3(_216[1u]), (uint3(_216[1u]) & _743) - _736, (uint3(_216[1u]) & _736) != uint3(0u)));
            _215[0u] = int3(select(uint3(_215[0u]), (uint3(_215[0u]) & _743) - _736, (uint3(_215[0u]) & _736) != uint3(0u)));
            _215[1u] = int3(select(uint3(_215[1u]), (uint3(_215[1u]) & _743) - _736, (uint3(_215[1u]) & _736) != uint3(0u)));
        }
        if (_184[(*spvDescriptorSet0.cbCS).g_mode_id])
        {
            _216[1u] += _216[0u];
            _215[0u] += _216[0u];
            _215[1u] += _216[0u];
        }
        for (int _792 = 0; _792 < 2; _792++)
        {
            uint _798 = uint(_792);
            int3 _846;
            if ((*spvDescriptorSet0.cbCS).g_format == 95u)
            {
                int3 _845;
                if (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x < 15u)
                {
                    _845 = select(_216[_798], select(((_216[_798] << int3(16)) + int3(32768)) >> (int3(_424) & int3(31)), int3(65535), _216[_798] == int3((1 << (_424 & 31)) - 1)), _216[_798] != int3(0));
                }
                else
                {
                    _845 = _216[_798];
                }
                _846 = _845;
            }
            else
            {
                int3 _828;
                if (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x < 16u)
                {
                    int3 _810 = abs(_216[_798]);
                    int _812 = _424 - 1;
                    int3 _824 = select(_810, select(((_810 << int3(15)) + int3(16384)) >> (int3(_812) & int3(31)), int3(32767), _810 >= int3((1 << (_812 & 31)) - 1)), _810 != int3(0));
                    _828 = select(_824, -_824, select(uint3(1u), uint3(0u), _216[_798] >= int3(0)) > uint3(0u));
                }
                else
                {
                    _828 = _216[_798];
                }
                _846 = _828;
            }
            _216[_798] = _846;
        }
        for (int _848 = 0; _848 < 2; _848++)
        {
            uint _854 = uint(_848);
            int3 _902;
            if ((*spvDescriptorSet0.cbCS).g_format == 95u)
            {
                int3 _901;
                if (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x < 15u)
                {
                    _901 = select(_215[_854], select(((_215[_854] << int3(16)) + int3(32768)) >> (int3(_424) & int3(31)), int3(65535), _215[_854] == int3((1 << (_424 & 31)) - 1)), _215[_854] != int3(0));
                }
                else
                {
                    _901 = _215[_854];
                }
                _902 = _901;
            }
            else
            {
                int3 _884;
                if (_199[(*spvDescriptorSet0.cbCS).g_mode_id].x < 16u)
                {
                    int3 _866 = abs(_215[_854]);
                    int _868 = _424 - 1;
                    int3 _880 = select(_866, select(((_866 << int3(15)) + int3(16384)) >> (int3(_868) & int3(31)), int3(32767), _866 >= int3((1 << (_868 & 31)) - 1)), _866 != int3(0));
                    _884 = select(_880, -_880, select(uint3(1u), uint3(0u), _215[_854] >= int3(0)) > uint3(0u));
                }
                else
                {
                    _884 = _215[_854];
                }
                _902 = _884;
            }
            _215[_854] = _902;
        }
        float _907;
        _907 = 0.0;
        float _908;
        spvUnsafeArray<int, 8> aWeight3;
        bool _905;
        bool _904 = false;
        uint _909 = 0u;
        for (; _909 < 16u; _904 = _905, _907 = _908, _909++)
        {
            uint3 _1035;
            if (((_200[_227] >> (_909 & 31u)) & 1u) != 0u)
            {
                float _983 = dot(_413, float3(shared_temp[_226 + _909].pixel_ph - _415));
                int _1001 = int(((_393 <= 0.0) || (_983 <= 0.0)) ? 0u : ((_983 < _393) ? _202[uint((_983 * 63.499988555908203125) / _393)] : _202[63]));
                if (!_904)
                {
                    aWeight3 = _203;
                }
                int3 _1015 = (((_215[0u] * int3(64 - aWeight3[_1001])) + (_215[1u] * int3(aWeight3[_1001]))) + int3(32)) >> int3(6);
                int3 _1033;
                if ((*spvDescriptorSet0.cbCS).g_format == 95u)
                {
                    _1033 = (_1015 * int3(31)) >> int3(6);
                }
                else
                {
                    int3 _1026 = select((_1015 * int3(31)) >> int3(5), -((_1015 * int3(-31)) >> int3(5)), _1015 < int3(0));
                    _1033 = select(_1026, (-_1026) | int3(32768), _1026 < int3(0));
                }
                _905 = _904 ? _904 : true;
                _1035 = uint3(_1033);
            }
            else
            {
                float _926 = dot(_387, float3(shared_temp[_226 + _909].pixel_ph - _389));
                int _944 = int(((_370 <= 0.0) || (_926 <= 0.0)) ? 0u : ((_926 < _370) ? _202[uint((_926 * 63.499988555908203125) / _370)] : _202[63]));
                if (!_904)
                {
                    aWeight3 = _203;
                }
                int3 _958 = (((_216[0u] * int3(64 - aWeight3[_944])) + (_216[1u] * int3(aWeight3[_944]))) + int3(32)) >> int3(6);
                int3 _976;
                if ((*spvDescriptorSet0.cbCS).g_format == 95u)
                {
                    _976 = (_958 * int3(31)) >> int3(6);
                }
                else
                {
                    int3 _969 = select((_958 * int3(31)) >> int3(5), -((_958 * int3(-31)) >> int3(5)), _958 < int3(0));
                    _976 = select(_969, (-_969) | int3(32768), _969 < int3(0));
                }
                _905 = _904 ? _904 : true;
                _1035 = uint3(_976);
            }
            float3 _1049 = float3(float2(as_type<half2>(_1035.x)).x, float2(as_type<half2>(_1035.y)).x, float2(as_type<half2>(_1035.z)).x) - shared_temp[_226 + _909].pixel_hr;
            _908 = _907 + dot(_1049, _1049);
        }
        shared_temp[gl_LocalInvocationIndex].error = (_634 != 0) ? 100000002004087734272.0 : _907;
        shared_temp[gl_LocalInvocationIndex].best_mode = _183[(*spvDescriptorSet0.cbCS).g_mode_id];
        shared_temp[gl_LocalInvocationIndex].best_partition = _227;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_235)
    {
        uint _1062 = gl_LocalInvocationIndex + 16u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_1062].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_1062].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_1062].best_mode;
            shared_temp[gl_LocalInvocationIndex].best_partition = shared_temp[_1062].best_partition;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_227 < 8u)
    {
        uint _1080 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_1080].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_1080].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_1080].best_mode;
            shared_temp[gl_LocalInvocationIndex].best_partition = shared_temp[_1080].best_partition;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_227 < 4u)
    {
        uint _1098 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_1098].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_1098].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_1098].best_mode;
            shared_temp[gl_LocalInvocationIndex].best_partition = shared_temp[_1098].best_partition;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_227 < 2u)
    {
        uint _1116 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_1116].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_1116].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_1116].best_mode;
            shared_temp[gl_LocalInvocationIndex].best_partition = shared_temp[_1116].best_partition;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_227 < 1u)
    {
        uint _1134 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_1134].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_1134].error;
            shared_temp[gl_LocalInvocationIndex].best_mode = shared_temp[_1134].best_mode;
            shared_temp[gl_LocalInvocationIndex].best_partition = shared_temp[_1134].best_partition;
        }
        if (as_type<float>(((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_225])[0]) > shared_temp[gl_LocalInvocationIndex].error)
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_225] = uint4(as_type<uint>(shared_temp[gl_LocalInvocationIndex].error), shared_temp[gl_LocalInvocationIndex].best_mode, shared_temp[gl_LocalInvocationIndex].best_partition, 0u);
        }
        else
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_225] = (*spvDescriptorSet0.g_InBuff)._m0[_225];
        }
    }
}

