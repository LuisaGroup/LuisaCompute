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

struct type_StructuredBuffer_v4uint
{
    uint4 _m0[1];
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

struct spvDescriptorSetBuffer0
{
    constant type_cbCS* cbCS [[id(0)]];
    texture2d<float> g_Input [[id(1)]];
    const device type_StructuredBuffer_v4uint* g_InBuff [[id(2)]];
    device type_RWStructuredBuffer_v4uint* g_OutBuff [[id(3)]];
};

constant spvUnsafeArray<uint, 64> _192 = spvUnsafeArray<uint, 64>({ 52428u, 34952u, 61166u, 60616u, 51328u, 65260u, 65224u, 60544u, 51200u, 65516u, 65152u, 59392u, 65512u, 65280u, 65520u, 61440u, 63248u, 142u, 28928u, 2254u, 140u, 29456u, 12544u, 36046u, 2188u, 12560u, 26214u, 13932u, 6120u, 4080u, 29070u, 14748u, 43690u, 61680u, 23130u, 13260u, 15420u, 21930u, 38550u, 42330u, 29646u, 5064u, 12876u, 15324u, 27030u, 49980u, 39270u, 1632u, 626u, 1252u, 20032u, 10016u, 51510u, 37740u, 14790u, 25500u, 37686u, 40134u, 33150u, 59160u, 52464u, 4044u, 30532u, 60962u });
constant spvUnsafeArray<uint2, 128> _215 = spvUnsafeArray<uint2, 128>({ uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(6u, 0u), uint2(2u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(3u, 15u), uint2(3u, 8u), uint2(15u, 8u), uint2(15u, 3u), uint2(8u, 15u), uint2(3u, 15u), uint2(15u, 3u), uint2(15u, 8u), uint2(8u, 15u), uint2(8u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(5u, 15u), uint2(3u, 15u), uint2(3u, 8u), uint2(3u, 15u), uint2(3u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(3u, 8u), uint2(6u, 15u), uint2(10u, 8u), uint2(5u, 3u), uint2(8u, 15u), uint2(8u, 6u), uint2(6u, 10u), uint2(8u, 15u), uint2(5u, 15u), uint2(15u, 10u), uint2(15u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(5u, 10u), uint2(6u, 10u), uint2(10u, 8u), uint2(8u, 9u), uint2(15u, 10u), uint2(15u, 6u), uint2(3u, 15u), uint2(15u, 8u), uint2(5u, 15u), uint2(15u, 3u), uint2(15u, 6u), uint2(15u, 6u), uint2(15u, 8u), uint2(3u, 15u), uint2(15u, 3u), uint2(5u, 15u), uint2(5u, 15u), uint2(5u, 15u), uint2(8u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(8u, 15u), uint2(13u, 15u), uint2(15u, 3u), uint2(12u, 15u), uint2(3u, 15u), uint2(3u, 8u) });
constant spvUnsafeArray<uint, 16> _216 = spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u });
constant spvUnsafeArray<uint, 16> _217 = spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<uint, 16> _218 = spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<spvUnsafeArray<uint, 16>, 3> _219 = spvUnsafeArray<spvUnsafeArray<uint, 16>, 3>({ spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u }), spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }), spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }) });
constant spvUnsafeArray<uint, 64> _220 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });
constant spvUnsafeArray<uint, 64> _221 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<uint, 64> _222 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u });
constant spvUnsafeArray<spvUnsafeArray<uint, 64>, 3> _223 = spvUnsafeArray<spvUnsafeArray<uint, 64>, 3>({ spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u }) });
constant spvUnsafeArray<uint, 2> _224 = spvUnsafeArray<uint, 2>({ 0u, 0u });
constant spvUnsafeArray<uint, 2> _225 = spvUnsafeArray<uint, 2>({ 4294967295u, 4294967295u });

kernel void TryMode137CS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<BufferShared, 64> shared_temp;
    uint _242 = gl_LocalInvocationIndex / 64u;
    uint _247 = ((*spvDescriptorSet0.cbCS).g_start_block_id + gl_WorkGroupID.x) + _242;
    uint _248 = _242 * 64u;
    uint _249 = gl_LocalInvocationIndex - _248;
    uint _252 = _247 / (*spvDescriptorSet0.cbCS).g_num_block_x;
    bool _257 = _249 < 16u;
    if (_257)
    {
        int3 _265 = int3(uint3(((_247 - (_252 * (*spvDescriptorSet0.cbCS).g_num_block_x)) * 4u) + (_249 % 4u), (_252 * 4u) + (_249 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = clamp(uint4(spvDescriptorSet0.g_Input.read(uint2(_265.xy), _265.z) * 255.0), uint4(0u), uint4(255u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared_temp[gl_LocalInvocationIndex].error = 4294967295u;
    if (_249 < 64u)
    {
        spvUnsafeArray<spvUnsafeArray<uint4, 2>, 2> _235;
        _235[0][0u] = uint4(4294967295u);
        _235[0][1u] = uint4(0u);
        _235[1][0u] = uint4(4294967295u);
        _235[1][1u] = uint4(0u);
        for (uint _285 = 0u; _285 < 16u; _285++)
        {
            uint _291 = _248 + _285;
            if (((_192[_249] >> (_285 & 31u)) & 1u) == 1u)
            {
                _235[1][0u] = min(_235[1][0u], shared_temp[_291].pixel);
                _235[1][1u] = max(_235[1][1u], shared_temp[_291].pixel);
            }
            else
            {
                _235[0][0u] = min(_235[0][0u], shared_temp[_291].pixel);
                _235[0][1u] = max(_235[0][1u], shared_temp[_291].pixel);
            }
        }
        spvUnsafeArray<uint4, 2> _312 = _235[1];
        uint _316 = (1u == (*spvDescriptorSet0.cbCS).g_mode_id) ? 2u : 4u;
        spvUnsafeArray<uint, 2> _236 = spvUnsafeArray<uint, 2>({ 0u, 0u });
        spvUnsafeArray<uint, 2> _237 = spvUnsafeArray<uint, 2>({ 4294967295u, 4294967295u });
        spvUnsafeArray<uint4, 2> _234;
        for (uint _318 = 0u; _318 < _316; _318++)
        {
            _235[0] = _235[0];
            _235[1] = _312;
            uint _325;
            _325 = 0u;
            for (; _325 < 2u; _325++)
            {
                if ((*spvDescriptorSet0.cbCS).g_mode_id == 1u)
                {
                    uint3 _345 = uint3(_318);
                    uint4 _349 = (((((((_235[_325][0u].xyzz << uint4(8u)) + _235[_325][0u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | _345).xyzz << uint4(1u);
                    uint4 _351 = _349 | (_349 >> uint4(7u));
                    _235[_325][0u] = uint4(_351.x, _351.y, _351.z, _235[_325][0u].w);
                    _235[_325][0u].w = 255u;
                    uint4 _368 = (((((((_235[_325][1u].xyzz << uint4(8u)) + _235[_325][1u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | _345).xyzz << uint4(1u);
                    uint4 _370 = _368 | (_368 >> uint4(7u));
                    _235[_325][1u] = uint4(_370.x, _370.y, _370.z, _235[_325][1u].w);
                    _235[_325][1u].w = 255u;
                }
                else
                {
                    if ((*spvDescriptorSet0.cbCS).g_mode_id == 3u)
                    {
                        uint2 _238 = uint2(_318, _318 >> 1u) & uint2(1u);
                        for (uint _382 = 0u; _382 < 2u; )
                        {
                            uint3 _390 = _235[_325][_382].xyz & uint3(4294967294u);
                            _234[_382] = uint4(_390.x, _390.y, _390.z, _234[_382].w);
                            uint3 _399 = _234[_382].xyz | uint3(_238[_382]);
                            _234[_382] = uint4(_399.x, _399.y, _399.z, _234[_382].w);
                            _234[_382].w = 255u;
                            _235[_325][_382] = uint4(_234[_382].x, _234[_382].y, _234[_382].z, _235[_325][_382].w);
                            _235[_325][_382].w = 255u;
                            _382++;
                            continue;
                        }
                    }
                    else
                    {
                        if ((*spvDescriptorSet0.cbCS).g_mode_id == 7u)
                        {
                            uint2 _412 = uint2(_318, _318 >> 1u) & uint2(1u);
                            uint4 _424 = (((((((_235[_325][0u] << uint4(8u)) + _235[_325][0u]) * uint4(63u)) + uint4(32768u)) >> uint4(16u)) & uint4(4294967294u)) | uint4(_412.x)) << uint4(2u);
                            _235[_325][0u] = _424 | (_424 >> uint4(6u));
                            uint4 _438 = (((((((_235[_325][1u] << uint4(8u)) + _235[_325][1u]) * uint4(63u)) + uint4(32768u)) >> uint4(16u)) & uint4(4294967294u)) | uint4(_412.y)) << uint4(2u);
                            _235[_325][1u] = _438 | (_438 >> uint4(6u));
                        }
                    }
                }
            }
            int4 _444 = int4(_235[0][1u] - _235[0][0u]);
            int4 _448 = int4(_235[1][1u] - _235[1][0u]);
            bool _449 = (*spvDescriptorSet0.cbCS).g_mode_id != 7u;
            int4 _454;
            int4 _455;
            if (_449)
            {
                int4 _452 = _448;
                _452.w = 0;
                int4 _453 = _444;
                _453.w = 0;
                _454 = _452;
                _455 = _453;
            }
            else
            {
                _454 = _448;
                _455 = _444;
            }
            int _466 = (((_455.x * _455.x) + (_455.y * _455.y)) + (_455.z * _455.z)) + (_455.w * _455.w);
            int _477 = (((_454.x * _454.x) + (_454.y * _454.y)) + (_454.z * _454.z)) + (_454.w * _454.w);
            uint4 _478 = uint4(_455);
            uint4 _482 = shared_temp[_248].pixel - _235[0][0u];
            int _498 = int((((_478.x * _482.x) + (_478.y * _482.y)) + (_478.z * _482.z)) + (_478.w * _482.w));
            int4 _514;
            if (((_466 > 0) && (_498 > 0)) && (uint(float(_498) * 63.499988555908203125) > uint(32 * _466)))
            {
                uint4 _512 = _235[0][0u];
                _235[0][0u] = _235[0][1u];
                _235[0][1u] = _512;
                _514 = -_455;
            }
            else
            {
                _514 = _455;
            }
            uint4 _515 = uint4(_454);
            uint4 _522 = shared_temp[_248 + _215[_249].x].pixel - _235[1][0u];
            int _538 = int((((_515.x * _522.x) + (_515.y * _522.y)) + (_515.z * _522.z)) + (_515.w * _522.w));
            int4 _554;
            if (((_477 > 0) && (_538 > 0)) && (uint(float(_538) * 63.499988555908203125) > uint(32 * _477)))
            {
                uint4 _552 = _235[1][0u];
                _235[1][0u] = _235[1][1u];
                _235[1][1u] = _552;
                _554 = -_454;
            }
            else
            {
                _554 = _454;
            }
            uint _556 = ((*spvDescriptorSet0.cbCS).g_mode_id != 1u) ? 2u : 1u;
            spvUnsafeArray<uint, 2> _239 = spvUnsafeArray<uint, 2>({ 0u, 0u });
            for (uint _558 = 0u; _558 < 16u; _558++)
            {
                uint _566 = (_192[_249] >> (_558 & 31u)) & 1u;
                bool _567 = _566 == 1u;
                uint _645;
                if (_567)
                {
                    uint4 _571 = uint4(_554);
                    uint4 _576 = shared_temp[_248 + _558].pixel - _235[1][0u];
                    int _592 = int((((_571.x * _576.x) + (_571.y * _576.y)) + (_571.z * _576.z)) + (_571.w * _576.w));
                    _645 = ((_477 <= 0) || (_592 <= 0)) ? 0u : ((_592 < _477) ? _223[_556][uint((float(_592) * 63.499988555908203125) / float(_477))] : _223[_556][63]);
                }
                else
                {
                    uint4 _608 = uint4(_514);
                    uint4 _613 = shared_temp[_248 + _558].pixel - _235[0][0u];
                    int _629 = int((((_608.x * _613.x) + (_608.y * _613.y)) + (_608.z * _613.z)) + (_608.w * _613.w));
                    _645 = ((_466 <= 0) || (_629 <= 0)) ? 0u : ((_629 < _466) ? _223[_556][uint((float(_629) * 63.499988555908203125) / float(_466))] : _223[_556][63]);
                }
                uint4 _660 = (((uint4(64u - _219[_556][_645]) * _235[_566][0u]) + (uint4(_219[_556][_645]) * _235[_566][1u])) + uint4(32u)) >> uint4(6u);
                uint4 _664;
                if (_449)
                {
                    uint4 _663 = _660;
                    _663.w = 255u;
                    _664 = _663;
                }
                else
                {
                    _664 = _660;
                }
                uint _665 = _248 + _558;
                uint4 _667 = shared_temp[_665].pixel;
                uint4 _675;
                uint4 _676;
                if (_664.x < _667.x)
                {
                    uint4 _673 = _664;
                    _673.x = _667.x;
                    uint4 _674 = _667;
                    _674.x = _664.x;
                    _675 = _674;
                    _676 = _673;
                }
                else
                {
                    _675 = _667;
                    _676 = _664;
                }
                uint4 _684;
                uint4 _685;
                if (_676.y < _675.y)
                {
                    uint4 _682 = _676;
                    _682.y = _675.y;
                    uint4 _683 = _675;
                    _683.y = _676.y;
                    _684 = _683;
                    _685 = _682;
                }
                else
                {
                    _684 = _675;
                    _685 = _676;
                }
                uint4 _693;
                uint4 _694;
                if (_685.z < _684.z)
                {
                    uint4 _691 = _685;
                    _691.z = _684.z;
                    uint4 _692 = _684;
                    _692.z = _685.z;
                    _693 = _692;
                    _694 = _691;
                }
                else
                {
                    _693 = _684;
                    _694 = _685;
                }
                uint4 _702;
                uint4 _703;
                if (_694.w < _693.w)
                {
                    uint4 _700 = _694;
                    _700.w = _693.w;
                    uint4 _701 = _693;
                    _701.w = _694.w;
                    _702 = _700;
                    _703 = _701;
                }
                else
                {
                    _702 = _694;
                    _703 = _693;
                }
                uint4 _704 = _702 - _703;
                uint _705 = _704.x;
                uint _707 = _704.y;
                uint _709 = _704.z;
                float _717 = float(_704.w);
                uint _721 = uint(float(((_705 * _705) + (_707 * _707)) + (_709 * _709)) + (((*spvDescriptorSet0.cbCS).g_alpha_weight * _717) * _717));
                if (_567)
                {
                    _239[1] += _721;
                }
                else
                {
                    _239[0] += _721;
                }
            }
            for (uint _732 = 0u; _732 < 2u; _732++)
            {
                if (_239[_732] < _237[_732])
                {
                    _237[_732] = _239[_732];
                    _236[_732] = _318;
                }
            }
        }
        shared_temp[gl_LocalInvocationIndex].error = _237[0] + _237[1];
        shared_temp[gl_LocalInvocationIndex].mode = (*spvDescriptorSet0.cbCS).g_mode_id;
        shared_temp[gl_LocalInvocationIndex]._partition = _249;
        if ((*spvDescriptorSet0.cbCS).g_mode_id == 1u)
        {
            shared_temp[gl_LocalInvocationIndex].rotation = (_236[1] << 1u) | _236[0];
        }
        else
        {
            shared_temp[gl_LocalInvocationIndex].rotation = (_236[1] << 2u) | _236[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_249 < 32u)
    {
        uint _776 = gl_LocalInvocationIndex + 32u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_776].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_776].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_776].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_776]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_776].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_257)
    {
        uint _795 = gl_LocalInvocationIndex + 16u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_795].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_795].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_795].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_795]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_795].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_249 < 8u)
    {
        uint _815 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_815].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_815].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_815].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_815]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_815].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_249 < 4u)
    {
        uint _835 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_835].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_835].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_835].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_835]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_835].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_249 < 2u)
    {
        uint _855 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_855].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_855].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_855].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_855]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_855].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_249 < 1u)
    {
        uint _875 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_875].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_875].error;
            shared_temp[gl_LocalInvocationIndex].mode = shared_temp[_875].mode;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_875]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_875].rotation;
        }
        if (((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_247])[0] > shared_temp[gl_LocalInvocationIndex].error)
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_247] = uint4(shared_temp[gl_LocalInvocationIndex].error, shared_temp[gl_LocalInvocationIndex].mode, shared_temp[gl_LocalInvocationIndex]._partition, shared_temp[gl_LocalInvocationIndex].rotation);
        }
        else
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_247] = (*spvDescriptorSet0.g_InBuff)._m0[_247];
        }
    }
}

