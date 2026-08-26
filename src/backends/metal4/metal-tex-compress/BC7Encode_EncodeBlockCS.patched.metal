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

constant uint4 _303 = {};

struct spvDescriptorSetBuffer0
{
    type_cbCS cbCS;
    texture2d<float> g_Input;
    const device type_StructuredBuffer_v4uint* g_InBuff;
    device type_RWStructuredBuffer_v4uint* g_OutBuff;
};

constant spvUnsafeArray<uint, 64> _262 = spvUnsafeArray<uint, 64>({ 52428u, 34952u, 61166u, 60616u, 51328u, 65260u, 65224u, 60544u, 51200u, 65516u, 65152u, 59392u, 65512u, 65280u, 65520u, 61440u, 63248u, 142u, 28928u, 2254u, 140u, 29456u, 12544u, 36046u, 2188u, 12560u, 26214u, 13932u, 6120u, 4080u, 29070u, 14748u, 43690u, 61680u, 23130u, 13260u, 15420u, 21930u, 38550u, 42330u, 29646u, 5064u, 12876u, 15324u, 27030u, 49980u, 39270u, 1632u, 626u, 1252u, 20032u, 10016u, 51510u, 37740u, 14790u, 25500u, 37686u, 40134u, 33150u, 59160u, 52464u, 4044u, 30532u, 60962u });
constant spvUnsafeArray<uint, 64> _263 = spvUnsafeArray<uint, 64>({ 2858963024u, 1784303680u, 1515864576u, 1414570152u, 2779054080u, 2694860880u, 1431675040u, 1515868240u, 2857697280u, 2857719040u, 2863289600u, 2425393296u, 2492765332u, 2762253476u, 2846200912u, 705315408u, 2777960512u, 172118100u, 2779096320u, 1436590240u, 2829603924u, 1785348160u, 2762231808u, 437912832u, 5285028u, 2862977168u, 342452500u, 1768494080u, 2693105056u, 2860651540u, 1352967248u, 1784283648u, 2846195712u, 1351655592u, 2829094992u, 606348324u, 11162880u, 613566756u, 608801316u, 1352993360u, 1342874960u, 2863285316u, 1717960704u, 2778768800u, 1352683680u, 1764256040u, 1152035396u, 1717986816u, 2856600644u, 1420317864u, 2508232064u, 2526451200u, 2824098984u, 2157286784u, 2853442580u, 2526412800u, 2863272980u, 2689618080u, 2695210400u, 2516582400u, 1082146944u, 2846402984u, 2863311428u, 709513812u });
constant spvUnsafeArray<uint2, 128> _286 = spvUnsafeArray<uint2, 128>({ uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(6u, 0u), uint2(2u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(3u, 15u), uint2(3u, 8u), uint2(15u, 8u), uint2(15u, 3u), uint2(8u, 15u), uint2(3u, 15u), uint2(15u, 3u), uint2(15u, 8u), uint2(8u, 15u), uint2(8u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(5u, 15u), uint2(3u, 15u), uint2(3u, 8u), uint2(3u, 15u), uint2(3u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(3u, 8u), uint2(6u, 15u), uint2(10u, 8u), uint2(5u, 3u), uint2(8u, 15u), uint2(8u, 6u), uint2(6u, 10u), uint2(8u, 15u), uint2(5u, 15u), uint2(15u, 10u), uint2(15u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(5u, 10u), uint2(6u, 10u), uint2(10u, 8u), uint2(8u, 9u), uint2(15u, 10u), uint2(15u, 6u), uint2(3u, 15u), uint2(15u, 8u), uint2(5u, 15u), uint2(15u, 3u), uint2(15u, 6u), uint2(15u, 6u), uint2(15u, 8u), uint2(3u, 15u), uint2(15u, 3u), uint2(5u, 15u), uint2(5u, 15u), uint2(5u, 15u), uint2(8u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(8u, 15u), uint2(13u, 15u), uint2(15u, 3u), uint2(12u, 15u), uint2(3u, 15u), uint2(3u, 8u) });
constant spvUnsafeArray<uint2, 128> _290 = spvUnsafeArray<uint2, 128>({ uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(6u, 0u), uint2(2u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(3u, 15u), uint2(3u, 8u), uint2(8u, 15u), uint2(3u, 15u), uint2(8u, 15u), uint2(3u, 15u), uint2(3u, 15u), uint2(8u, 15u), uint2(8u, 15u), uint2(8u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(5u, 15u), uint2(3u, 15u), uint2(3u, 8u), uint2(3u, 15u), uint2(3u, 8u), uint2(8u, 15u), uint2(3u, 15u), uint2(3u, 15u), uint2(3u, 8u), uint2(6u, 15u), uint2(8u, 10u), uint2(3u, 5u), uint2(8u, 15u), uint2(6u, 8u), uint2(6u, 10u), uint2(8u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(8u, 15u), uint2(8u, 15u), uint2(3u, 15u), uint2(3u, 15u), uint2(5u, 10u), uint2(6u, 10u), uint2(8u, 10u), uint2(8u, 9u), uint2(10u, 15u), uint2(6u, 15u), uint2(3u, 15u), uint2(8u, 15u), uint2(5u, 15u), uint2(3u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(8u, 15u), uint2(3u, 15u), uint2(3u, 15u), uint2(5u, 15u), uint2(5u, 15u), uint2(5u, 15u), uint2(8u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(8u, 15u), uint2(13u, 15u), uint2(3u, 15u), uint2(12u, 15u), uint2(3u, 15u), uint2(3u, 8u) });
constant spvUnsafeArray<uint, 64> _291 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });
constant spvUnsafeArray<uint, 64> _292 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<uint, 64> _293 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u });
constant spvUnsafeArray<spvUnsafeArray<uint, 64>, 3> _294 = spvUnsafeArray<spvUnsafeArray<uint, 64>, 3>({ spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u }) });

kernel void EncodeBlockCS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<BufferShared, 64> shared_temp;
    uint _315 = gl_LocalInvocationIndex / 16u;
    uint _321 = (spvDescriptorSet0.cbCS.g_start_block_id + (gl_WorkGroupID.x * 4u)) + _315;
    uint _322 = _315 * 16u;
    uint _323 = gl_LocalInvocationIndex - _322;
    uint _326 = _321 / spvDescriptorSet0.cbCS.g_num_block_x;
    uint _332 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_321])[1];
    uint _333 = _332 & 2147483647u;
    uint _335 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_321])[2];
    uint _336 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_321])[1];
    uint _338 = (_336 >> 31u) & 1u;
    uint _340 = ((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_321])[3];
    bool _341 = _323 < 16u;
    if (_341)
    {
        int3 _349 = int3(uint3(((_321 - (_326 * spvDescriptorSet0.cbCS.g_num_block_x)) * 4u) + (_323 % 4u), (_326 * 4u) + (_323 / 4u), 0u));
        uint4 _356 = clamp(uint4(spvDescriptorSet0.g_Input.read(uint2(_349.xy), _349.z) * 255.0), uint4(0u), uint4(255u));
        uint4 _379;
        if ((4u == _333) || (5u == _333))
        {
            uint4 _378;
            if (1u == _340)
            {
                _378 = uint4(_356.w, _356.y, _356.z, _356.x);
            }
            else
            {
                uint4 _376;
                if (2u == _340)
                {
                    _376 = uint4(_356.x, _356.w, _356.z, _356.y);
                }
                else
                {
                    uint4 _374;
                    if (3u == _340)
                    {
                        _374 = uint4(_356.x, _356.y, _356.w, _356.z);
                    }
                    else
                    {
                        _374 = _356;
                    }
                    _376 = _374;
                }
                _378 = _376;
            }
            _379 = _378;
        }
        else
        {
            _379 = _356;
        }
        shared_temp[gl_LocalInvocationIndex].pixel = _379;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _382 = _262[_335];
    uint _383 = _335 - 64u;
    uint _385 = _263[_383];
    spvUnsafeArray<uint4, 2> _310;
    _310[0u] = uint4(4294967295u);
    _310[1u] = uint4(0u);
    if (_341)
    {
        uint4 _405;
        uint4 _406;
        if ((0u == _333) || (2u == _333))
        {
            bool4 _402 = bool4(2u == ((_385 >> ((_323 * 2u) & 31u)) & 3u));
            _405 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _402);
            _406 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _402);
        }
        else
        {
            _405 = uint4(0u);
            _406 = uint4(4294967295u);
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _406;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _405;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _409 = _323 < 8u;
    if (_409)
    {
        uint _414 = gl_LocalInvocationIndex + 8u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_414].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_414].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _423 = _323 < 4u;
    if (_423)
    {
        uint _428 = gl_LocalInvocationIndex + 4u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_428].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_428].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _437 = _323 < 2u;
    if (_437)
    {
        uint _442 = gl_LocalInvocationIndex + 2u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_442].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_442].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _451 = _323 < 1u;
    if (_451)
    {
        uint _456 = gl_LocalInvocationIndex + 1u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_456].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_456].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int _465 = int(_323);
    if (2 == _465)
    {
        _310[0u] = shared_temp[_322].endPoint_low;
        _310[1u] = shared_temp[_322].endPoint_high;
    }
    if (_341)
    {
        uint4 _507;
        uint4 _508;
        if ((0u == _333) || (2u == _333))
        {
            bool4 _504 = bool4(1u == ((_385 >> ((_323 * 2u) & 31u)) & 3u));
            _507 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _504);
            _508 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _504);
        }
        else
        {
            uint4 _501;
            uint4 _502;
            if (((1u == _333) || (3u == _333)) || (7u == _333))
            {
                bool4 _498 = bool4(1u == ((_382 >> (_323 & 31u)) & 1u));
                _501 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _498);
                _502 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _498);
            }
            else
            {
                _501 = uint4(0u);
                _502 = uint4(4294967295u);
            }
            _507 = _501;
            _508 = _502;
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _508;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _507;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_409)
    {
        uint _515 = gl_LocalInvocationIndex + 8u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_515].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_515].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_423)
    {
        uint _528 = gl_LocalInvocationIndex + 4u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_528].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_528].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_437)
    {
        uint _541 = gl_LocalInvocationIndex + 2u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_541].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_541].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_451)
    {
        uint _554 = gl_LocalInvocationIndex + 1u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_554].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_554].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (1 == _465)
    {
        _310[0u] = shared_temp[_322].endPoint_low;
        _310[1u] = shared_temp[_322].endPoint_high;
    }
    if (_341)
    {
        uint4 _613;
        uint4 _614;
        if ((0u == _333) || (2u == _333))
        {
            bool4 _610 = bool4(0u == ((_385 >> ((_323 * 2u) & 31u)) & 3u));
            _613 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _610);
            _614 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _610);
        }
        else
        {
            uint4 _607;
            uint4 _608;
            if (((1u == _333) || (3u == _333)) || (7u == _333))
            {
                bool4 _604 = bool4(0u == ((_382 >> (_323 & 31u)) & 1u));
                _607 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _604);
                _608 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _604);
            }
            else
            {
                bool4 _600 = bool4(((4u == _333) || (5u == _333)) || (6u == _333));
                _607 = select(uint4(0u), shared_temp[gl_LocalInvocationIndex].pixel, _600);
                _608 = select(uint4(4294967295u), shared_temp[gl_LocalInvocationIndex].pixel, _600);
            }
            _613 = _607;
            _614 = _608;
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _614;
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _613;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_409)
    {
        uint _621 = gl_LocalInvocationIndex + 8u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_621].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_621].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_423)
    {
        uint _634 = gl_LocalInvocationIndex + 4u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_634].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_634].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_437)
    {
        uint _647 = gl_LocalInvocationIndex + 2u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_647].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_647].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_451)
    {
        uint _660 = gl_LocalInvocationIndex + 1u;
        shared_temp[gl_LocalInvocationIndex].endPoint_low = min(shared_temp[gl_LocalInvocationIndex].endPoint_low, shared_temp[_660].endPoint_low);
        shared_temp[gl_LocalInvocationIndex].endPoint_high = max(shared_temp[gl_LocalInvocationIndex].endPoint_high, shared_temp[_660].endPoint_high);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == _465)
    {
        _310[0u] = shared_temp[_322].endPoint_low;
        _310[1u] = shared_temp[_322].endPoint_high;
    }
    if (_323 < 3u)
    {
        bool _679 = 1u == _333;
        uint2 _695;
        if (_679)
        {
            _695 = uint2((_340 >> (_323 & 31u)) & 1u);
        }
        else
        {
            uint _683 = _323 * 2u;
            _695 = uint2(_340 >> (_683 & 31u), _340 >> ((_683 + 1u) & 31u)) & uint2(1u);
        }
        uint4 _1041;
        uint4 _1042;
        if (0u == _333)
        {
            uint3 _1007 = ((((((_310[0u].xyzz << uint4(8u)) + _310[0u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_695.x);
            uint4 _1008 = uint4(_1007.x, _1007.y, _1007.z, _303.w);
            _1008.w = 255u;
            uint4 _1011 = _1008.xyzz << uint4(3u);
            uint4 _1013 = _1011 | (_1011 >> uint4(5u));
            _310[0u] = uint4(_1013.x, _1013.y, _1013.z, _310[0u].w);
            _310[0u].w = 255u;
            uint3 _1030 = ((((((_310[1u].xyzz << uint4(8u)) + _310[1u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_695.y);
            uint4 _1031 = uint4(_1030.x, _1030.y, _1030.z, _303.w);
            _1031.w = 255u;
            uint4 _1034 = _1031.xyzz << uint4(3u);
            uint4 _1036 = _1034 | (_1034 >> uint4(5u));
            _310[1u] = uint4(_1036.x, _1036.y, _1036.z, _310[1u].w);
            _310[1u].w = 255u;
            _1041 = _1031 << uint4(3u);
            _1042 = _1008 << uint4(3u);
        }
        else
        {
            uint4 _993;
            uint4 _994;
            if (_679)
            {
                uint3 _959 = ((((((_310[0u].xyzz << uint4(8u)) + _310[0u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_695.x);
                uint4 _960 = uint4(_959.x, _959.y, _959.z, _303.w);
                _960.w = 255u;
                uint4 _963 = _960.xyzz << uint4(1u);
                uint4 _965 = _963 | (_963 >> uint4(7u));
                _310[0u] = uint4(_965.x, _965.y, _965.z, _310[0u].w);
                _310[0u].w = 255u;
                uint3 _982 = ((((((_310[1u].xyzz << uint4(8u)) + _310[1u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_695.y);
                uint4 _983 = uint4(_982.x, _982.y, _982.z, _303.w);
                _983.w = 255u;
                uint4 _986 = _983.xyzz << uint4(1u);
                uint4 _988 = _986 | (_986 >> uint4(7u));
                _310[1u] = uint4(_988.x, _988.y, _988.z, _310[1u].w);
                _310[1u].w = 255u;
                _993 = _983 << uint4(1u);
                _994 = _960 << uint4(1u);
            }
            else
            {
                uint4 _945;
                uint4 _946;
                if (2u == _333)
                {
                    uint4 _917 = ((((_310[0u].xyzz << uint4(8u)) + _310[0u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u);
                    uint4 _918 = uint4(_917.x, _917.y, _917.z, _303.w);
                    _918.w = 255u;
                    uint4 _921 = _918.xyzz << uint4(3u);
                    uint4 _923 = _921 | (_921 >> uint4(5u));
                    _310[0u] = uint4(_923.x, _923.y, _923.z, _310[0u].w);
                    _310[0u].w = 255u;
                    uint4 _934 = ((((_310[1u].xyzz << uint4(8u)) + _310[1u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u);
                    uint4 _935 = uint4(_934.x, _934.y, _934.z, _303.w);
                    _935.w = 255u;
                    uint4 _938 = _935.xyzz << uint4(3u);
                    uint4 _940 = _938 | (_938 >> uint4(5u));
                    _310[1u] = uint4(_940.x, _940.y, _940.z, _310[1u].w);
                    _310[1u].w = 255u;
                    _945 = _935 << uint4(3u);
                    _946 = _918 << uint4(3u);
                }
                else
                {
                    uint4 _909;
                    uint4 _910;
                    if (3u == _333)
                    {
                        uint2 _311 = _695;
                        spvUnsafeArray<uint4, 2> _309;
                        for (uint _881 = 0u; _881 < 2u; )
                        {
                            uint3 _889 = _310[_881].xyz & uint3(4294967294u);
                            _309[_881] = uint4(_889.x, _889.y, _889.z, _309[_881].w);
                            uint3 _898 = _309[_881].xyz | uint3(_311[_881]);
                            _309[_881] = uint4(_898.x, _898.y, _898.z, _309[_881].w);
                            _309[_881].w = 255u;
                            _310[_881] = uint4(_309[_881].x, _309[_881].y, _309[_881].z, _310[_881].w);
                            _310[_881].w = 255u;
                            _881++;
                            continue;
                        }
                        _909 = _309[1];
                        _910 = _309[0];
                    }
                    else
                    {
                        uint4 _878;
                        uint4 _879;
                        if (4u == _333)
                        {
                            uint4 _814 = _310[0u];
                            uint4 _820 = ((((_814.xyzz << uint4(8u)) + _814.xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u);
                            uint _822 = _310[0u].w;
                            uint4 _823 = uint4(_822);
                            uint4 _828 = ((((_823 << uint4(8u)) + _823) * uint4(63u)) + uint4(32768u)) >> uint4(16u);
                            uint _829 = _828.x;
                            uint4 _831 = _820.xyzz << uint4(3u);
                            uint4 _833 = _831 | (_831 >> uint4(5u));
                            _310[0u] = uint4(_833.x, _833.y, _833.z, _310[0u].w);
                            uint4 _837 = uint4(_829) << uint4(2u);
                            _310[0u].w = (_837 | (_837 >> uint4(6u))).x;
                            uint3 _842 = _820.xyz << uint3(3u);
                            uint4 _843 = uint4(_842.x, _842.y, _842.z, _303.w);
                            _843.w = _829 << 2u;
                            uint4 _846 = _310[1u];
                            uint4 _852 = ((((_846.xyzz << uint4(8u)) + _846.xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u);
                            uint _854 = _310[1u].w;
                            uint4 _855 = uint4(_854);
                            uint4 _860 = ((((_855 << uint4(8u)) + _855) * uint4(63u)) + uint4(32768u)) >> uint4(16u);
                            uint _861 = _860.x;
                            uint4 _863 = _852.xyzz << uint4(3u);
                            uint4 _865 = _863 | (_863 >> uint4(5u));
                            _310[1u] = uint4(_865.x, _865.y, _865.z, _310[1u].w);
                            uint4 _869 = uint4(_861) << uint4(2u);
                            _310[1u].w = (_869 | (_869 >> uint4(6u))).x;
                            uint3 _874 = _852.xyz << uint3(3u);
                            uint4 _875 = uint4(_874.x, _874.y, _874.z, _303.w);
                            _875.w = _861 << 2u;
                            _878 = _875;
                            _879 = _843;
                        }
                        else
                        {
                            uint4 _812;
                            uint4 _813;
                            if (5u == _333)
                            {
                                uint4 _778 = ((((_310[0u].xyzz << uint4(8u)) + _310[0u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u);
                                uint4 _779 = uint4(_778.x, _778.y, _778.z, _303.w);
                                _779.w = _310[0u].w;
                                uint4 _784 = _779.xyzz << uint4(1u);
                                uint4 _786 = _784 | (_784 >> uint4(7u));
                                _310[0u] = uint4(_786.x, _786.y, _786.z, _310[0u].w);
                                uint3 _790 = _779.xyz << uint3(1u);
                                uint4 _798 = ((((_310[1u].xyzz << uint4(8u)) + _310[1u].xyzz) * uint4(127u)) + uint4(32768u)) >> uint4(16u);
                                uint4 _799 = uint4(_798.x, _798.y, _798.z, _303.w);
                                _799.w = _310[1u].w;
                                uint4 _804 = _799.xyzz << uint4(1u);
                                uint4 _806 = _804 | (_804 >> uint4(7u));
                                _310[1u] = uint4(_806.x, _806.y, _806.z, _310[1u].w);
                                uint3 _810 = _799.xyz << uint3(1u);
                                _812 = uint4(_810.x, _810.y, _810.z, _799.w);
                                _813 = uint4(_790.x, _790.y, _790.z, _779.w);
                            }
                            else
                            {
                                uint4 _770;
                                uint4 _771;
                                if (6u == _333)
                                {
                                    uint2 _312 = _695;
                                    spvUnsafeArray<uint4, 2> _308;
                                    for (uint _752 = 0u; _752 < 2u; )
                                    {
                                        _308[_752] = _310[_752] & uint4(4294967294u);
                                        _308[_752] |= uint4(_312[_752]);
                                        _310[_752] = _308[_752];
                                        _752++;
                                        continue;
                                    }
                                    _770 = _308[1];
                                    _771 = _308[0];
                                }
                                else
                                {
                                    uint4 _723 = _310[0u];
                                    uint4 _732 = ((((((_723 << uint4(8u)) + _723) * uint4(63u)) + uint4(32768u)) >> uint4(16u)) & uint4(4294967294u)) | uint4(_695.x);
                                    uint4 _733 = _732 << uint4(2u);
                                    _310[0u] = _733 | (_733 >> uint4(6u));
                                    uint4 _736 = _310[1u];
                                    uint4 _745 = ((((((_736 << uint4(8u)) + _736) * uint4(63u)) + uint4(32768u)) >> uint4(16u)) & uint4(4294967294u)) | uint4(_695.y);
                                    uint4 _746 = _745 << uint4(2u);
                                    _310[1u] = _746 | (_746 >> uint4(6u));
                                    _770 = _745 * uint4(4u);
                                    _771 = _732 * uint4(4u);
                                }
                                _812 = _770;
                                _813 = _771;
                            }
                            _878 = _812;
                            _879 = _813;
                        }
                        _909 = _878;
                        _910 = _879;
                    }
                    _945 = _909;
                    _946 = _910;
                }
                _993 = _945;
                _994 = _946;
            }
            _1041 = _993;
            _1042 = _994;
        }
        int4 _1046 = int4(_310[1u] - _310[0u]);
        int4 _1051;
        if (_333 < 4u)
        {
            int4 _1050 = _1046;
            _1050.w = 0;
            _1051 = _1050;
        }
        else
        {
            _1051 = _1046;
        }
        uint4 _1217;
        uint4 _1218;
        if ((4u == _333) || (5u == _333))
        {
            uint4 _1215;
            uint4 _1216;
            if (0u == _323)
            {
                int2 _1141 = int2(uint2(uint(((_1051.x * _1051.x) + (_1051.y * _1051.y)) + (_1051.z * _1051.z)), uint(_1051.w * _1051.w)));
                uint3 _1143 = uint3(_1051.xyz);
                uint3 _1149 = shared_temp[_322].pixel.xyz - _310[0u].xyz;
                int _1161 = int(((_1143.x * _1149.x) + (_1143.y * _1149.y)) + (_1143.z * _1149.z));
                uint _1166 = _310[0u].w;
                int _1169 = int(uint(_1051.w) * (((threadgroup uint*)&shared_temp[_322].pixel)[3] - _1166));
                int _1170 = _1141.x;
                uint4 _1191;
                uint4 _1192;
                if (((_1170 > 0) && (_1161 > 0)) && (uint(float(_1161) * 63.499988555908203125) > uint(32 * _1170)))
                {
                    uint4 _1183 = _310[0u];
                    _310[0u] = uint4(_310[1u].x, _310[1u].y, _310[1u].z, _310[0u].w);
                    _310[1u] = uint4(_1183.x, _1183.y, _1183.z, _310[1u].w);
                    _1191 = uint4(_1042.x, _1042.y, _1042.z, _1041.w);
                    _1192 = uint4(_1041.x, _1041.y, _1041.z, _1042.w);
                }
                else
                {
                    _1191 = _1041;
                    _1192 = _1042;
                }
                int _1193 = _1141.y;
                uint4 _1213;
                uint4 _1214;
                if (((_1193 > 0) && (_1169 > 0)) && (uint(float(_1169) * 63.499988555908203125) > uint(32 * _1193)))
                {
                    uint _1207 = _310[0u].w;
                    _310[0u].w = _310[1u].w;
                    _310[1u].w = _1207;
                    uint4 _1211 = _1192;
                    _1211.w = _1191.w;
                    uint4 _1212 = _1191;
                    _1212.w = _1192.w;
                    _1213 = _1212;
                    _1214 = _1211;
                }
                else
                {
                    _1213 = _1191;
                    _1214 = _1192;
                }
                _1215 = _1213;
                _1216 = _1214;
            }
            else
            {
                _1215 = _1041;
                _1216 = _1042;
            }
            _1217 = _1215;
            _1218 = _1216;
        }
        else
        {
            int _1073;
            if (0u == _323)
            {
                _1073 = 0;
            }
            else
            {
                int _1072;
                if (1u == _323)
                {
                    _1072 = int(_286[_335].x);
                }
                else
                {
                    _1072 = int(_286[_335].y);
                }
                _1073 = _1072;
            }
            int _1084 = (((_1051.x * _1051.x) + (_1051.y * _1051.y)) + (_1051.z * _1051.z)) + (_1051.w * _1051.w);
            uint4 _1085 = uint4(_1051);
            uint4 _1090 = _310[0u];
            uint4 _1091 = shared_temp[_322 + uint(_1073)].pixel - _1090;
            int _1107 = int((((_1085.x * _1091.x) + (_1085.y * _1091.y)) + (_1085.z * _1091.z)) + (_1085.w * _1091.w));
            bool _1117 = ((_1084 > 0) && (_1107 > 0)) && (uint(float(_1107) * 63.499988555908203125) > uint(32 * _1084));
            if (_1117)
            {
                uint4 _1120 = _310[0u];
                _310[0u] = _310[1u];
                _310[1u] = _1120;
            }
            bool4 _1122 = bool4(_1117);
            _1217 = select(_1041, _1042, _1122);
            _1218 = select(_1042, _1041, _1122);
        }
        shared_temp[gl_LocalInvocationIndex].endPoint_low = _310[0u];
        shared_temp[gl_LocalInvocationIndex].endPoint_high = _310[1u];
        shared_temp[gl_LocalInvocationIndex].endPoint_low_quantized = _1218;
        shared_temp[gl_LocalInvocationIndex].endPoint_high_quantized = _1217;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_341)
    {
        bool _1227 = 0u == _333;
        bool _1228 = 1u == _333;
        uint2 _1246;
        if (_1227 || _1228)
        {
            _1246 = uint2(1u);
        }
        else
        {
            uint2 _1245;
            if (6u == _333)
            {
                _1245 = uint2(0u);
            }
            else
            {
                uint2 _1244;
                if (4u == _333)
                {
                    _1244 = select(uint2(1u, 2u), uint2(2u, 1u), bool2(0u == _338));
                }
                else
                {
                    _1244 = uint2(2u);
                }
                _1245 = _1244;
            }
            _1246 = _1245;
        }
        int _1269;
        if (_1227 || (2u == _333))
        {
            _1269 = int((_385 >> ((_323 * 2u) & 31u)) & 3u);
        }
        else
        {
            int _1263;
            if ((_1228 || (3u == _333)) || (7u == _333))
            {
                _1263 = int((_382 >> (_323 & 31u)) & 1u);
            }
            else
            {
                _1263 = 0;
            }
            _1269 = _1263;
        }
        uint _1271 = _322 + uint(_1269);
        int4 _1277 = int4(shared_temp[_1271].endPoint_high - shared_temp[_1271].endPoint_low);
        int4 _1282;
        if (_333 < 4u)
        {
            int4 _1281 = _1277;
            _1281.w = 0;
            _1282 = _1281;
        }
        else
        {
            _1282 = _1277;
        }
        uint _1407;
        uint _1408;
        if ((4u == _333) || (5u == _333))
        {
            int _1343 = ((_1282.x * _1282.x) + (_1282.y * _1282.y)) + (_1282.z * _1282.z);
            int _1345 = _1282.w * _1282.w;
            uint3 _1347 = uint3(_1282.xyz);
            uint3 _1352 = shared_temp[gl_LocalInvocationIndex].pixel.xyz - shared_temp[_1271].endPoint_low.xyz;
            int _1364 = int(((_1347.x * _1352.x) + (_1347.y * _1352.y)) + (_1347.z * _1352.z));
            uint _1380 = ((_1343 <= 0) || (_1364 <= 0)) ? 0u : ((_1364 < _1343) ? _294[_1246.x][uint((float(_1364) * 63.499988555908203125) / float(_1343))] : _294[_1246.x][63]);
            int _1387 = int(uint(_1282.w) * (((threadgroup uint*)&shared_temp[gl_LocalInvocationIndex].pixel)[3] - shared_temp[_1271].endPoint_low.w));
            uint _1403 = ((_1345 <= 0) || (_1387 <= 0)) ? 0u : ((_1387 < _1345) ? _294[_1246.y][uint((float(_1387) * 63.499988555908203125) / float(_1345))] : _294[_1246.y][63]);
            bool _1404 = _338 != 0u;
            _1407 = _1404 ? _1380 : _1403;
            _1408 = _1404 ? _1403 : _1380;
        }
        else
        {
            int _1299 = (((_1282.x * _1282.x) + (_1282.y * _1282.y)) + (_1282.z * _1282.z)) + (_1282.w * _1282.w);
            uint4 _1300 = uint4(_1282);
            uint4 _1303 = shared_temp[gl_LocalInvocationIndex].pixel - shared_temp[_1271].endPoint_low;
            int _1319 = int((((_1300.x * _1303.x) + (_1300.y * _1303.y)) + (_1300.z * _1303.z)) + (_1300.w * _1303.w));
            _1407 = 0u;
            _1408 = ((_1299 <= 0) || (_1319 <= 0)) ? 0u : ((_1319 < _1299) ? _294[_1246.x][uint((float(_1319) * 63.499988555908203125) / float(_1299))] : _294[_1246.x][63]);
        }
        shared_temp[gl_LocalInvocationIndex].error = _1408;
        shared_temp[gl_LocalInvocationIndex].mode = _1407;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0u == _323)
    {
        uint4 _3022;
        if (0u == _333)
        {
            uint _2808 = _322 + 1u;
            uint _2819 = _322 + 2u;
            uint4 _2926;
            uint _2929;
            _2926 = uint4((((((((1u | (_383 << 1u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 240u) << 1u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 240u) << 5u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_low_quantized)[0] & 240u) << 9u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_high_quantized)[0] & 240u) << 13u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_low_quantized)[0] & 240u) << 17u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_high_quantized)[0] & 240u) << 21u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 240u) << 25u), (((((((((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 240u) >> 7u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 240u) >> 3u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_low_quantized)[1] & 240u) << 1u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_high_quantized)[1] & 240u) << 5u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_low_quantized)[1] & 240u) << 9u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_high_quantized)[1] & 240u) << 13u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 240u) << 17u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 240u) << 21u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_low_quantized)[2] & 240u) << 25u), (((((((((((((threadgroup uint*)&shared_temp[_2808].endPoint_low_quantized)[2] & 240u) >> 7u) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_high_quantized)[2] & 240u) >> 3u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_low_quantized)[2] & 240u) << 1u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_high_quantized)[2] & 240u) << 5u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 8u) << 10u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 8u) << 11u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_low_quantized)[0] & 8u) << 12u)) | ((((threadgroup uint*)&shared_temp[_2808].endPoint_high_quantized)[0] & 8u) << 13u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_low_quantized)[0] & 8u) << 14u)) | ((((threadgroup uint*)&shared_temp[_2819].endPoint_high_quantized)[0] & 8u) << 15u)) | (shared_temp[_322].error << 19u), 0u);
            _2929 = 1u;
            for (; _2929 <= min(_290[_335].x, 4u); )
            {
                uint4 _2927 = _2926;
                _2927.z = _2926.z | (shared_temp[_322 + _2929].error << (((_2929 * 3u) + 18u) & 31u));
                _2926 = _2927;
                _2929++;
                continue;
            }
            uint4 _2984;
            uint _2985;
            if (_290[_335].x < 4u)
            {
                uint4 _2982 = _2926;
                _2982.z = _2926.z | (shared_temp[_322 + 4u].error << 29u);
                _2984 = _2982;
                _2985 = _2929 + 1u;
            }
            else
            {
                uint4 _2957 = _2926;
                _2957.w = _2926.w | ((shared_temp[_322 + 4u].error & 4u) >> 2u);
                uint4 _2959;
                uint _2962;
                _2959 = _2957;
                _2962 = _2929;
                for (; _2962 <= _290[_335].x; )
                {
                    uint4 _2960 = _2959;
                    _2960.w = _2959.w | (shared_temp[_322 + _2962].error << (((_2962 * 3u) - 14u) & 31u));
                    _2959 = _2960;
                    _2962++;
                    continue;
                }
                _2984 = _2959;
                _2985 = _2962;
            }
            uint4 _2987;
            uint _2990;
            _2987 = _2984;
            _2990 = _2985;
            for (; _2990 <= _290[_335].y; )
            {
                uint4 _2988 = _2987;
                _2988.w = _2987.w | (shared_temp[_322 + _2990].error << (((_2990 * 3u) - 15u) & 31u));
                _2987 = _2988;
                _2990++;
                continue;
            }
            uint4 _3006;
            _3006 = _2987;
            for (uint _3009 = _2990; _3009 < 16u; )
            {
                uint4 _3007 = _3006;
                _3007.w = _3006.w | (shared_temp[_322 + _3009].error << (((_3009 * 3u) - 16u) & 31u));
                _3006 = _3007;
                _3009++;
                continue;
            }
            _3022 = _3006;
        }
        else
        {
            uint4 _2795;
            if (1u == _333)
            {
                uint _2399 = _322 + 1u;
                uint _2409 = ((((2u | (_335 << 2u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 252u) << 6u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 252u) << 12u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_low_quantized)[0] & 252u) << 18u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_high_quantized)[0] & 252u) << 24u);
                uint _2438 = ((((((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 252u) >> 2u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 252u) << 4u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_low_quantized)[1] & 252u) << 10u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_high_quantized)[1] & 252u) << 16u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 252u) << 22u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 252u) << 28u);
                uint _2463 = ((((((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 252u) >> 4u) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_low_quantized)[2] & 252u) << 2u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_high_quantized)[2] & 252u) << 8u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 2u) << 15u)) | ((((threadgroup uint*)&shared_temp[_2399].endPoint_low_quantized)[0] & 2u) << 16u)) | (shared_temp[_322].error << 18u);
                uint4 _2794;
                if (_290[_335].x == 15u)
                {
                    uint4 _2770 = uint4(_2409, _2438, _2463, ((((((((((shared_temp[_322 + 15u].error << 30u) | (shared_temp[_322 + 14u].error << 27u)) | (shared_temp[_322 + 13u].error << 24u)) | (shared_temp[_322 + 12u].error << 21u)) | (shared_temp[_322 + 11u].error << 18u)) | (shared_temp[_322 + 10u].error << 15u)) | (shared_temp[_322 + 9u].error << 12u)) | (shared_temp[_322 + 8u].error << 9u)) | (shared_temp[_322 + 7u].error << 6u)) | (shared_temp[_322 + 6u].error << 3u)) | shared_temp[_322 + 5u].error);
                    _2770.z = _2463 | (((((shared_temp[_322 + 4u].error << 29u) | (shared_temp[_322 + 3u].error << 26u)) | (shared_temp[_322 + 2u].error << 23u)) | (shared_temp[_2399].error << 20u)) | (shared_temp[_322].error << 18u));
                    _2794 = _2770;
                }
                else
                {
                    uint4 _2716;
                    if (_290[_335].x == 2u)
                    {
                        uint _2684 = _322 + 5u;
                        uint4 _2689 = uint4(_2409, _2438, _2463, ((((((((((shared_temp[_322 + 15u].error << 29u) | (shared_temp[_322 + 14u].error << 26u)) | (shared_temp[_322 + 13u].error << 23u)) | (shared_temp[_322 + 12u].error << 20u)) | (shared_temp[_322 + 11u].error << 17u)) | (shared_temp[_322 + 10u].error << 14u)) | (shared_temp[_322 + 9u].error << 11u)) | (shared_temp[_322 + 8u].error << 8u)) | (shared_temp[_322 + 7u].error << 5u)) | (shared_temp[_322 + 6u].error << 2u)) | (shared_temp[_2684].error >> 1u));
                        _2689.z = _2463 | ((((((shared_temp[_2684].error << 31u) | (shared_temp[_322 + 4u].error << 28u)) | (shared_temp[_322 + 3u].error << 25u)) | (shared_temp[_322 + 2u].error << 23u)) | (shared_temp[_2399].error << 20u)) | (shared_temp[_322].error << 18u));
                        _2716 = _2689;
                    }
                    else
                    {
                        uint4 _2634;
                        if (_290[_335].x == 8u)
                        {
                            uint4 _2610 = uint4(_2409, _2438, _2463, ((((((((((shared_temp[_322 + 15u].error << 29u) | (shared_temp[_322 + 14u].error << 26u)) | (shared_temp[_322 + 13u].error << 23u)) | (shared_temp[_322 + 12u].error << 20u)) | (shared_temp[_322 + 11u].error << 17u)) | (shared_temp[_322 + 10u].error << 14u)) | (shared_temp[_322 + 9u].error << 11u)) | (shared_temp[_322 + 8u].error << 9u)) | (shared_temp[_322 + 7u].error << 6u)) | (shared_temp[_322 + 6u].error << 3u)) | shared_temp[_322 + 5u].error);
                            _2610.z = _2463 | (((((shared_temp[_322 + 4u].error << 29u) | (shared_temp[_322 + 3u].error << 26u)) | (shared_temp[_322 + 2u].error << 23u)) | (shared_temp[_2399].error << 20u)) | (shared_temp[_322].error << 18u));
                            _2634 = _2610;
                        }
                        else
                        {
                            uint4 _2533 = uint4(_2409, _2438, _2463, ((((((((((shared_temp[_322 + 15u].error << 29u) | (shared_temp[_322 + 14u].error << 26u)) | (shared_temp[_322 + 13u].error << 23u)) | (shared_temp[_322 + 12u].error << 20u)) | (shared_temp[_322 + 11u].error << 17u)) | (shared_temp[_322 + 10u].error << 14u)) | (shared_temp[_322 + 9u].error << 11u)) | (shared_temp[_322 + 8u].error << 8u)) | (shared_temp[_322 + 7u].error << 5u)) | (shared_temp[_322 + 6u].error << 3u)) | shared_temp[_322 + 5u].error);
                            _2533.z = _2463 | (((((shared_temp[_322 + 4u].error << 29u) | (shared_temp[_322 + 3u].error << 26u)) | (shared_temp[_322 + 2u].error << 23u)) | (shared_temp[_2399].error << 20u)) | (shared_temp[_322].error << 18u));
                            _2634 = _2533;
                        }
                        _2716 = _2634;
                    }
                    _2794 = _2716;
                }
                _2795 = _2794;
            }
            else
            {
                uint4 _2386;
                if (2u == _333)
                {
                    uint _2240 = _322 + 1u;
                    uint _2251 = _322 + 2u;
                    uint4 _2333;
                    uint _2336;
                    _2333 = uint4((((((4u | (_383 << 3u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 248u) << 6u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 248u) << 11u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_low_quantized)[0] & 248u) << 16u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_high_quantized)[0] & 248u) << 21u)) | ((((threadgroup uint*)&shared_temp[_2251].endPoint_low_quantized)[0] & 248u) << 26u), (((((((((threadgroup uint*)&shared_temp[_2251].endPoint_low_quantized)[0] & 248u) >> 6u) | ((((threadgroup uint*)&shared_temp[_2251].endPoint_high_quantized)[0] & 248u) >> 1u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 248u) << 4u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 248u) << 9u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_low_quantized)[1] & 248u) << 14u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_high_quantized)[1] & 248u) << 19u)) | ((((threadgroup uint*)&shared_temp[_2251].endPoint_low_quantized)[1] & 248u) << 24u), (((((((((threadgroup uint*)&shared_temp[_2251].endPoint_high_quantized)[1] & 248u) >> 3u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 248u) << 2u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 248u) << 7u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_low_quantized)[2] & 248u) << 12u)) | ((((threadgroup uint*)&shared_temp[_2240].endPoint_high_quantized)[2] & 248u) << 17u)) | ((((threadgroup uint*)&shared_temp[_2251].endPoint_low_quantized)[2] & 248u) << 22u)) | ((((threadgroup uint*)&shared_temp[_2251].endPoint_high_quantized)[2] & 248u) << 27u), ((((threadgroup uint*)&shared_temp[_2251].endPoint_high_quantized)[2] & 248u) >> 5u) | (shared_temp[_322].error << 3u));
                    _2336 = 1u;
                    for (; _2336 <= _290[_335].x; )
                    {
                        uint4 _2334 = _2333;
                        _2334.w = _2333.w | (shared_temp[_322 + _2336].error << (((_2336 * 2u) + 2u) & 31u));
                        _2333 = _2334;
                        _2336++;
                        continue;
                    }
                    uint4 _2352;
                    uint _2355;
                    _2352 = _2333;
                    _2355 = _2336;
                    for (; _2355 <= _290[_335].y; )
                    {
                        uint4 _2353 = _2352;
                        _2353.w = _2352.w | (shared_temp[_322 + _2355].error << (((_2355 * 2u) + 1u) & 31u));
                        _2352 = _2353;
                        _2355++;
                        continue;
                    }
                    uint4 _2371;
                    _2371 = _2352;
                    for (uint _2374 = _2355; _2374 < 16u; )
                    {
                        uint4 _2372 = _2371;
                        _2372.w = _2371.w | (shared_temp[_322 + _2374].error << ((_2374 * 2u) & 31u));
                        _2371 = _2372;
                        _2374++;
                        continue;
                    }
                    _2386 = _2371;
                }
                else
                {
                    uint4 _2227;
                    if (3u == _333)
                    {
                        uint _2115 = _322 + 1u;
                        uint4 _2193;
                        uint _2196;
                        _2193 = uint4(((((8u | (_335 << 4u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 254u) << 9u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 254u) << 16u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_low_quantized)[0] & 254u) << 23u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[0] & 254u) << 30u), (((((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[0] & 254u) >> 2u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 254u) << 5u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 254u) << 12u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_low_quantized)[1] & 254u) << 19u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[1] & 254u) << 26u), (((((((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[1] & 254u) >> 6u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 254u) << 1u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 254u) << 8u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_low_quantized)[2] & 254u) << 15u)) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[2] & 254u) << 22u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 1u) << 30u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 1u) << 31u), (((((threadgroup uint*)&shared_temp[_2115].endPoint_low_quantized)[0] & 1u) << 0u) | ((((threadgroup uint*)&shared_temp[_2115].endPoint_high_quantized)[0] & 1u) << 1u)) | (shared_temp[_322].error << 2u));
                        _2196 = 1u;
                        for (; _2196 <= _290[_335].x; )
                        {
                            uint4 _2194 = _2193;
                            _2194.w = _2193.w | (shared_temp[_322 + _2196].error << (((_2196 * 2u) + 1u) & 31u));
                            _2193 = _2194;
                            _2196++;
                            continue;
                        }
                        uint4 _2212;
                        _2212 = _2193;
                        for (uint _2215 = _2196; _2215 < 16u; )
                        {
                            uint4 _2213 = _2212;
                            _2213.w = _2212.w | (shared_temp[_322 + _2215].error << ((_2215 * 2u) & 31u));
                            _2212 = _2213;
                            _2215++;
                            continue;
                        }
                        _2227 = _2212;
                    }
                    else
                    {
                        uint4 _2102;
                        if (4u == _333)
                        {
                            uint _1957 = _322 + 1u;
                            uint _1962 = _322 + 2u;
                            uint _1967 = _322 + 3u;
                            uint _1972 = _322 + 4u;
                            uint _1977 = _322 + 5u;
                            uint _1982 = _322 + 6u;
                            uint _1987 = _322 + 7u;
                            uint _1994 = _322 + 8u;
                            uint _1999 = _322 + 9u;
                            uint _2004 = _322 + 10u;
                            uint _2009 = _322 + 11u;
                            uint _2014 = _322 + 12u;
                            uint _2019 = _322 + 13u;
                            uint _2024 = _322 + 14u;
                            uint _2029 = _322 + 15u;
                            uint _2054 = (((((((((((((shared_temp[_1987].error >> 1u) | (shared_temp[_1994].error << 1u)) | (shared_temp[_1999].error << 3u)) | (shared_temp[_2004].error << 5u)) | (shared_temp[_2009].error << 7u)) | (shared_temp[_2014].error << 9u)) | (shared_temp[_2019].error << 11u)) | (shared_temp[_2024].error << 13u)) | (shared_temp[_2029].error << 15u)) | ((shared_temp[_322].mode & 3u) << 17u)) | (shared_temp[_1957].mode << 19u)) | (shared_temp[_1962].mode << 22u)) | (shared_temp[_1967].mode << 25u)) | (shared_temp[_1972].mode << 28u);
                            _2102 = uint4(((((((16u | ((_340 & 3u) << 5u)) | ((_338 & 1u) << 7u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 248u) << 5u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 248u) << 10u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 248u) << 15u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 248u) << 20u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 248u) << 25u), ((((((((((((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 248u) >> 7u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 248u) >> 2u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[3] & 252u) << 4u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[3] & 252u) << 10u)) | ((shared_temp[_322].error & 1u) << 18u)) | (shared_temp[_1957].error << 19u)) | (shared_temp[_1962].error << 21u)) | (shared_temp[_1967].error << 23u)) | (shared_temp[_1972].error << 25u)) | (shared_temp[_1977].error << 27u)) | (shared_temp[_1982].error << 29u)) | (shared_temp[_1987].error << 31u), _2054 | (shared_temp[_1977].mode << 31u), ((((((((((shared_temp[_1977].mode >> 1u) | (shared_temp[_1982].mode << 2u)) | (shared_temp[_1987].mode << 5u)) | (shared_temp[_1994].mode << 8u)) | (shared_temp[_1999].mode << 11u)) | (shared_temp[_2004].mode << 14u)) | (shared_temp[_2009].mode << 17u)) | (shared_temp[_2014].mode << 20u)) | (shared_temp[_2019].mode << 23u)) | (shared_temp[_2024].mode << 26u)) | (shared_temp[_2029].mode << 29u));
                        }
                        else
                        {
                            uint4 _1902;
                            if (5u == _333)
                            {
                                uint _1760 = _322 + 1u;
                                uint _1765 = _322 + 2u;
                                uint _1770 = _322 + 3u;
                                uint _1775 = _322 + 4u;
                                uint _1780 = _322 + 5u;
                                uint _1785 = _322 + 6u;
                                uint _1790 = _322 + 7u;
                                uint _1795 = _322 + 8u;
                                uint _1800 = _322 + 9u;
                                uint _1805 = _322 + 10u;
                                uint _1810 = _322 + 11u;
                                uint _1815 = _322 + 12u;
                                uint _1820 = _322 + 13u;
                                uint _1824 = ((((((((((((((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[3] >> 6u) | (shared_temp[_322].error << 2u)) | (shared_temp[_1760].error << 3u)) | (shared_temp[_1765].error << 5u)) | (shared_temp[_1770].error << 7u)) | (shared_temp[_1775].error << 9u)) | (shared_temp[_1780].error << 11u)) | (shared_temp[_1785].error << 13u)) | (shared_temp[_1790].error << 15u)) | (shared_temp[_1795].error << 17u)) | (shared_temp[_1800].error << 19u)) | (shared_temp[_1805].error << 21u)) | (shared_temp[_1810].error << 23u)) | (shared_temp[_1815].error << 25u)) | (shared_temp[_1820].error << 27u);
                                uint _1825 = _322 + 14u;
                                uint _1830 = _322 + 15u;
                                uint _1892 = ((((((((((((((shared_temp[_1830].error >> 1u) | (shared_temp[_322].mode << 1u)) | (shared_temp[_1760].mode << 2u)) | (shared_temp[_1765].mode << 4u)) | (shared_temp[_1770].mode << 6u)) | (shared_temp[_1775].mode << 8u)) | (shared_temp[_1780].mode << 10u)) | (shared_temp[_1785].mode << 12u)) | (shared_temp[_1790].mode << 14u)) | (shared_temp[_1795].mode << 16u)) | (shared_temp[_1800].mode << 18u)) | (shared_temp[_1805].mode << 20u)) | (shared_temp[_1810].mode << 22u)) | (shared_temp[_1815].mode << 24u)) | (shared_temp[_1820].mode << 26u);
                                _1902 = uint4(((((32u | (_340 << 6u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 254u) << 7u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 254u) << 14u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 254u) << 21u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 254u) << 28u), (((((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 254u) >> 4u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 254u) << 3u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 254u) << 10u)) | (((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[3] << 18u)) | (((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[3] << 26u), (_1824 | (shared_temp[_1825].error << 29u)) | (shared_temp[_1830].error << 31u), (_1892 | (shared_temp[_1825].mode << 28u)) | (shared_temp[_1830].mode << 30u));
                            }
                            else
                            {
                                uint4 _1710;
                                if (6u == _333)
                                {
                                    _1710 = uint4((((64u | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 254u) << 6u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 254u) << 13u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 254u) << 20u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 254u) << 27u), ((((((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 254u) >> 5u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 254u) << 2u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 254u) << 9u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[3] & 254u) << 16u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[3] & 254u) << 23u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 1u) << 31u), ((((((((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 1u) | (shared_temp[_322].error << 1u)) | (shared_temp[_322 + 1u].error << 4u)) | (shared_temp[_322 + 2u].error << 8u)) | (shared_temp[_322 + 3u].error << 12u)) | (shared_temp[_322 + 4u].error << 16u)) | (shared_temp[_322 + 5u].error << 20u)) | (shared_temp[_322 + 6u].error << 24u)) | (shared_temp[_322 + 7u].error << 28u), (((((((shared_temp[_322 + 8u].error << 0u) | (shared_temp[_322 + 9u].error << 4u)) | (shared_temp[_322 + 10u].error << 8u)) | (shared_temp[_322 + 11u].error << 12u)) | (shared_temp[_322 + 12u].error << 16u)) | (shared_temp[_322 + 13u].error << 20u)) | (shared_temp[_322 + 14u].error << 24u)) | (shared_temp[_322 + 15u].error << 28u));
                                }
                                else
                                {
                                    uint _1454 = _322 + 1u;
                                    uint4 _1548;
                                    uint _1551;
                                    _1548 = uint4(((((128u | (_335 << 8u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 248u) << 11u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 248u) << 16u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_low_quantized)[0] & 248u) << 21u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[0] & 248u) << 26u), (((((((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[0] & 248u) >> 6u) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[1] & 248u) >> 1u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[1] & 248u) << 4u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_low_quantized)[1] & 248u) << 9u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[1] & 248u) << 14u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[2] & 248u) << 19u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[2] & 248u) << 24u), ((((((((((threadgroup uint*)&shared_temp[_1454].endPoint_low_quantized)[2] & 248u) >> 3u) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[2] & 248u) << 2u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[3] & 248u) << 7u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[3] & 248u) << 12u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_low_quantized)[3] & 248u) << 17u)) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[3] & 248u) << 22u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_low_quantized)[0] & 4u) << 28u)) | ((((threadgroup uint*)&shared_temp[_322].endPoint_high_quantized)[0] & 4u) << 29u), (((((threadgroup uint*)&shared_temp[_1454].endPoint_low_quantized)[0] & 4u) >> 2u) | ((((threadgroup uint*)&shared_temp[_1454].endPoint_high_quantized)[0] & 4u) >> 1u)) | (shared_temp[_322].error << 2u));
                                    _1551 = 1u;
                                    for (; _1551 <= _290[_335].x; )
                                    {
                                        uint4 _1549 = _1548;
                                        _1549.w = _1548.w | (shared_temp[_322 + _1551].error << (((_1551 * 2u) + 1u) & 31u));
                                        _1548 = _1549;
                                        _1551++;
                                        continue;
                                    }
                                    uint4 _1567;
                                    _1567 = _1548;
                                    for (uint _1570 = _1551; _1570 < 16u; )
                                    {
                                        uint4 _1568 = _1567;
                                        _1568.w = _1567.w | (shared_temp[_322 + _1570].error << ((_1570 * 2u) & 31u));
                                        _1567 = _1568;
                                        _1570++;
                                        continue;
                                    }
                                    _1710 = _1567;
                                }
                                _1902 = _1710;
                            }
                            _2102 = _1902;
                        }
                        _2227 = _2102;
                    }
                    _2386 = _2227;
                }
                _2795 = _2386;
            }
            _3022 = _2795;
        }
        (*spvDescriptorSet0.g_OutBuff)._m0[_321] = _3022;
    }
}

