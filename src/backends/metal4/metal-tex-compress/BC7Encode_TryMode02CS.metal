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

constant spvUnsafeArray<uint, 64> _198 = spvUnsafeArray<uint, 64>({ 2858963024u, 1784303680u, 1515864576u, 1414570152u, 2779054080u, 2694860880u, 1431675040u, 1515868240u, 2857697280u, 2857719040u, 2863289600u, 2425393296u, 2492765332u, 2762253476u, 2846200912u, 705315408u, 2777960512u, 172118100u, 2779096320u, 1436590240u, 2829603924u, 1785348160u, 2762231808u, 437912832u, 5285028u, 2862977168u, 342452500u, 1768494080u, 2693105056u, 2860651540u, 1352967248u, 1784283648u, 2846195712u, 1351655592u, 2829094992u, 606348324u, 11162880u, 613566756u, 608801316u, 1352993360u, 1342874960u, 2863285316u, 1717960704u, 2778768800u, 1352683680u, 1764256040u, 1152035396u, 1717986816u, 2856600644u, 1420317864u, 2508232064u, 2526451200u, 2824098984u, 2157286784u, 2853442580u, 2526412800u, 2863272980u, 2689618080u, 2695210400u, 2516582400u, 1082146944u, 2846402984u, 2863311428u, 709513812u });
constant spvUnsafeArray<uint2, 128> _221 = spvUnsafeArray<uint2, 128>({ uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(8u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(6u, 0u), uint2(6u, 0u), uint2(2u, 0u), uint2(6u, 0u), uint2(8u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(15u, 0u), uint2(2u, 0u), uint2(2u, 0u), uint2(15u, 0u), uint2(3u, 15u), uint2(3u, 8u), uint2(15u, 8u), uint2(15u, 3u), uint2(8u, 15u), uint2(3u, 15u), uint2(15u, 3u), uint2(15u, 8u), uint2(8u, 15u), uint2(8u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(6u, 15u), uint2(5u, 15u), uint2(3u, 15u), uint2(3u, 8u), uint2(3u, 15u), uint2(3u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(3u, 8u), uint2(6u, 15u), uint2(10u, 8u), uint2(5u, 3u), uint2(8u, 15u), uint2(8u, 6u), uint2(6u, 10u), uint2(8u, 15u), uint2(5u, 15u), uint2(15u, 10u), uint2(15u, 8u), uint2(8u, 15u), uint2(15u, 3u), uint2(3u, 15u), uint2(5u, 10u), uint2(6u, 10u), uint2(10u, 8u), uint2(8u, 9u), uint2(15u, 10u), uint2(15u, 6u), uint2(3u, 15u), uint2(15u, 8u), uint2(5u, 15u), uint2(15u, 3u), uint2(15u, 6u), uint2(15u, 6u), uint2(15u, 8u), uint2(3u, 15u), uint2(15u, 3u), uint2(5u, 15u), uint2(5u, 15u), uint2(5u, 15u), uint2(8u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(5u, 15u), uint2(10u, 15u), uint2(8u, 15u), uint2(13u, 15u), uint2(15u, 3u), uint2(12u, 15u), uint2(3u, 15u), uint2(3u, 8u) });
constant spvUnsafeArray<uint, 16> _222 = spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u });
constant spvUnsafeArray<uint, 16> _223 = spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<uint, 16> _224 = spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u });
constant spvUnsafeArray<spvUnsafeArray<uint, 16>, 3> _225 = spvUnsafeArray<spvUnsafeArray<uint, 16>, 3>({ spvUnsafeArray<uint, 16>({ 0u, 4u, 9u, 13u, 17u, 21u, 26u, 30u, 34u, 38u, 43u, 47u, 51u, 55u, 60u, 64u }), spvUnsafeArray<uint, 16>({ 0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }), spvUnsafeArray<uint, 16>({ 0u, 21u, 43u, 64u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }) });
constant spvUnsafeArray<uint, 64> _226 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u });
constant spvUnsafeArray<uint, 64> _227 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u });
constant spvUnsafeArray<uint, 64> _228 = spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u });
constant spvUnsafeArray<spvUnsafeArray<uint, 64>, 3> _229 = spvUnsafeArray<spvUnsafeArray<uint, 64>, 3>({ spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u }), spvUnsafeArray<uint, 64>({ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u }) });
constant spvUnsafeArray<uint, 3> _230 = spvUnsafeArray<uint, 3>({ 0u, 0u, 0u });
constant spvUnsafeArray<uint, 3> _231 = spvUnsafeArray<uint, 3>({ 4294967295u, 4294967295u, 4294967295u });

kernel void TryMode02CS(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]])
{
    threadgroup spvUnsafeArray<BufferShared, 64> shared_temp;
    uint _245 = gl_LocalInvocationIndex / 64u;
    uint _250 = ((*spvDescriptorSet0.cbCS).g_start_block_id + gl_WorkGroupID.x) + _245;
    uint _251 = _245 * 64u;
    uint _252 = gl_LocalInvocationIndex - _251;
    uint _255 = _250 / (*spvDescriptorSet0.cbCS).g_num_block_x;
    bool _260 = _252 < 16u;
    if (_260)
    {
        int3 _268 = int3(uint3(((_250 - (_255 * (*spvDescriptorSet0.cbCS).g_num_block_x)) * 4u) + (_252 % 4u), (_255 * 4u) + (_252 / 4u), 0u));
        shared_temp[gl_LocalInvocationIndex].pixel = clamp(uint4(spvDescriptorSet0.g_Input.read(uint2(_268.xy), _268.z) * 255.0), uint4(0u), uint4(255u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shared_temp[gl_LocalInvocationIndex].error = 4294967295u;
    bool _280 = 0u == (*spvDescriptorSet0.cbCS).g_mode_id;
    if (_252 < (_280 ? 16u : 64u))
    {
        uint _285 = _252 + 64u;
        spvUnsafeArray<spvUnsafeArray<uint4, 2>, 3> _235;
        _235[0][0u] = uint4(4294967295u);
        _235[0][1u] = uint4(0u);
        _235[1][0u] = uint4(4294967295u);
        _235[1][1u] = uint4(0u);
        _235[2][0u] = uint4(4294967295u);
        _235[2][1u] = uint4(0u);
        for (uint _295 = 0u; _295 < 16u; _295++)
        {
            uint _301 = _251 + _295;
            uint _307 = (_198[_252] >> ((_295 * 2u) & 31u)) & 3u;
            if (_307 == 2u)
            {
                _235[2][0u] = min(_235[2][0u], shared_temp[_301].pixel);
                _235[2][1u] = max(_235[2][1u], shared_temp[_301].pixel);
            }
            else
            {
                if (_307 == 1u)
                {
                    _235[1][0u] = min(_235[1][0u], shared_temp[_301].pixel);
                    _235[1][1u] = max(_235[1][1u], shared_temp[_301].pixel);
                }
                else
                {
                    _235[0][0u] = min(_235[0][0u], shared_temp[_301].pixel);
                    _235[0][1u] = max(_235[0][1u], shared_temp[_301].pixel);
                }
            }
        }
        spvUnsafeArray<uint4, 2> _331 = _235[1];
        spvUnsafeArray<uint4, 2> _333 = _235[2];
        uint _334 = _280 ? 4u : 1u;
        spvUnsafeArray<uint, 3> _237 = spvUnsafeArray<uint, 3>({ 0u, 0u, 0u });
        spvUnsafeArray<uint, 3> _238 = spvUnsafeArray<uint, 3>({ 4294967295u, 4294967295u, 4294967295u });
        spvUnsafeArray<uint, 16> _236;
        spvUnsafeArray<int4, 3> _239;
        spvUnsafeArray<int, 3> _240;
        for (uint _336 = 0u; _336 < _334; _336++)
        {
            _235[0] = _235[0];
            _235[1] = _331;
            _235[2] = _333;
            for (uint _343 = 0u; _343 < 3u; _343++)
            {
                if (_280)
                {
                    uint2 _354 = uint2(_336, _336 >> 1u) & uint2(1u);
                    uint4 _370 = (((((((_235[_343][0u].xyzz << uint4(8u)) + _235[_343][0u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_354.x)).xyzz << uint4(3u);
                    uint4 _372 = _370 | (_370 >> uint4(5u));
                    _235[_343][0u] = uint4(_372.x, _372.y, _372.z, _235[_343][0u].w);
                    _235[_343][0u].w = 255u;
                    uint4 _391 = (((((((_235[_343][1u].xyzz << uint4(8u)) + _235[_343][1u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyz & uint3(4294967294u)).xyz | uint3(_354.y)).xyzz << uint4(3u);
                    uint4 _393 = _391 | (_391 >> uint4(5u));
                    _235[_343][1u] = uint4(_393.x, _393.y, _393.z, _235[_343][1u].w);
                    _235[_343][1u].w = 255u;
                }
                else
                {
                    uint4 _406 = (((((_235[_343][0u].xyzz << uint4(8u)) + _235[_343][0u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(3u);
                    uint4 _408 = _406 | (_406 >> uint4(5u));
                    _235[_343][0u] = uint4(_408.x, _408.y, _408.z, _235[_343][0u].w);
                    _235[_343][0u].w = 255u;
                    uint4 _421 = (((((_235[_343][1u].xyzz << uint4(8u)) + _235[_343][1u].xyzz) * uint4(31u)) + uint4(32768u)) >> uint4(16u)).xyzz << uint4(3u);
                    uint4 _423 = _421 | (_421 >> uint4(5u));
                    _235[_343][1u] = uint4(_423.x, _423.y, _423.z, _235[_343][1u].w);
                    _235[_343][1u].w = 255u;
                }
            }
            uint _430 = uint(1 + int(2u == (*spvDescriptorSet0.cbCS).g_mode_id));
            _239[0] = int4(_235[0][1u] - _235[0][0u]);
            _239[1] = int4(_235[1][1u] - _235[1][0u]);
            _239[2] = int4(_235[2][1u] - _235[2][0u]);
            _239[2].w = 0;
            _239[1].w = 0;
            _239[0].w = 0;
            _240[0] = (((_239[0].x * _239[0].x) + (_239[0].y * _239[0].y)) + (_239[0].z * _239[0].z)) + (_239[0].w * _239[0].w);
            _240[1] = (((_239[1].x * _239[1].x) + (_239[1].y * _239[1].y)) + (_239[1].z * _239[1].z)) + (_239[1].w * _239[1].w);
            _240[2] = (((_239[2].x * _239[2].x) + (_239[2].y * _239[2].y)) + (_239[2].z * _239[2].z)) + (_239[2].w * _239[2].w);
            spvUnsafeArray<uint, 3> _507 = spvUnsafeArray<uint, 3>({ 0u, _221[_285].x, _221[_285].y });
            spvUnsafeArray<uint, 3> _241 = _507;
            for (uint _509 = 0u; _509 < 3u; _509++)
            {
                uint4 _517 = uint4(_239[_509]);
                uint4 _525 = shared_temp[_251 + _241[_509]].pixel - _235[_509][0u];
                int _541 = int((((_517.x * _525.x) + (_517.y * _525.y)) + (_517.z * _525.z)) + (_517.w * _525.w));
                if (((_240[_509] > 0) && (_541 > 0)) && (uint(float(_541) * 63.499988555908203125) > uint(32 * _240[_509])))
                {
                    _239[_509] = -_239[_509];
                    uint4 _560 = _235[_509][0u];
                    _235[_509][0u] = _235[_509][1u];
                    _235[_509][1u] = _560;
                }
            }
            spvUnsafeArray<uint, 3> _242 = spvUnsafeArray<uint, 3>({ 0u, 0u, 0u });
            for (uint _563 = 0u; _563 < 16u; _563++)
            {
                uint _572 = (_198[_252] >> ((_563 * 2u) & 31u)) & 3u;
                bool _573 = _572 == 2u;
                if (_573)
                {
                    uint4 _578 = uint4(_239[2]);
                    uint4 _583 = shared_temp[_251 + _563].pixel - _235[2][0u];
                    int _599 = int((((_578.x * _583.x) + (_578.y * _583.y)) + (_578.z * _583.z)) + (_578.w * _583.w));
                    _236[_563] = ((_240[2] <= 0) || (_599 <= 0)) ? 0u : ((_599 < _240[2]) ? _229[_430][uint((float(_599) * 63.499988555908203125) / float(_240[2]))] : _229[_430][63]);
                }
                else
                {
                    if (_572 == 1u)
                    {
                        uint4 _624 = uint4(_239[1]);
                        uint4 _629 = shared_temp[_251 + _563].pixel - _235[1][0u];
                        int _645 = int((((_624.x * _629.x) + (_624.y * _629.y)) + (_624.z * _629.z)) + (_624.w * _629.w));
                        _236[_563] = ((_240[1] <= 0) || (_645 <= 0)) ? 0u : ((_645 < _240[1]) ? _229[_430][uint((float(_645) * 63.499988555908203125) / float(_240[1]))] : _229[_430][63]);
                    }
                    else
                    {
                        uint4 _666 = uint4(_239[0]);
                        uint4 _671 = shared_temp[_251 + _563].pixel - _235[0][0u];
                        int _687 = int((((_666.x * _671.x) + (_666.y * _671.y)) + (_666.z * _671.z)) + (_666.w * _671.w));
                        _236[_563] = ((_240[0] <= 0) || (_687 <= 0)) ? 0u : ((_687 < _240[0]) ? _229[_430][uint((float(_687) * 63.499988555908203125) / float(_240[0]))] : _229[_430][63]);
                    }
                }
                uint4 _725 = (((uint4(64u - _225[_430][_236[_563]]) * _235[_572][0u]) + (uint4(_225[_430][_236[_563]]) * _235[_572][1u])) + uint4(32u)) >> uint4(6u);
                uint4 _726 = _725;
                _726.w = 255u;
                uint _727 = _251 + _563;
                uint4 _729 = shared_temp[_727].pixel;
                uint _730 = _725.x;
                uint4 _737;
                uint4 _738;
                if (_730 < _729.x)
                {
                    uint4 _735 = _726;
                    _735.x = _729.x;
                    uint4 _736 = _729;
                    _736.x = _730;
                    _737 = _736;
                    _738 = _735;
                }
                else
                {
                    _737 = _729;
                    _738 = _726;
                }
                uint4 _746;
                uint4 _747;
                if (_738.y < _737.y)
                {
                    uint4 _744 = _738;
                    _744.y = _737.y;
                    uint4 _745 = _737;
                    _745.y = _738.y;
                    _746 = _745;
                    _747 = _744;
                }
                else
                {
                    _746 = _737;
                    _747 = _738;
                }
                uint4 _755;
                uint4 _756;
                if (_747.z < _746.z)
                {
                    uint4 _753 = _747;
                    _753.z = _746.z;
                    uint4 _754 = _746;
                    _754.z = _747.z;
                    _755 = _754;
                    _756 = _753;
                }
                else
                {
                    _755 = _746;
                    _756 = _747;
                }
                uint4 _764;
                uint4 _765;
                if (_756.w < _755.w)
                {
                    uint4 _762 = _756;
                    _762.w = _755.w;
                    uint4 _763 = _755;
                    _763.w = _756.w;
                    _764 = _762;
                    _765 = _763;
                }
                else
                {
                    _764 = _756;
                    _765 = _755;
                }
                uint4 _766 = _764 - _765;
                uint _767 = _766.x;
                uint _769 = _766.y;
                uint _771 = _766.z;
                float _779 = float(_766.w);
                uint _783 = uint(float(((_767 * _767) + (_769 * _769)) + (_771 * _771)) + (((*spvDescriptorSet0.cbCS).g_alpha_weight * _779) * _779));
                if (_573)
                {
                    _242[2] += _783;
                }
                else
                {
                    if (_572 == 1u)
                    {
                        _242[1] += _783;
                    }
                    else
                    {
                        _242[0] += _783;
                    }
                }
            }
            for (uint _801 = 0u; _801 < 3u; _801++)
            {
                if (_242[_801] < _238[_801])
                {
                    _238[_801] = _242[_801];
                    _237[_801] = _336;
                }
            }
        }
        shared_temp[gl_LocalInvocationIndex].error = (_238[0] + _238[1]) + _238[2];
        shared_temp[gl_LocalInvocationIndex]._partition = _285;
        shared_temp[gl_LocalInvocationIndex].rotation = ((_237[2] << 4u) | (_237[1] << 2u)) | _237[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_252 < 32u)
    {
        uint _840 = gl_LocalInvocationIndex + 32u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_840].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_840].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_840]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_840].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_260)
    {
        uint _856 = gl_LocalInvocationIndex + 16u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_856].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_856].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_856]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_856].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_252 < 8u)
    {
        uint _873 = gl_LocalInvocationIndex + 8u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_873].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_873].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_873]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_873].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_252 < 4u)
    {
        uint _890 = gl_LocalInvocationIndex + 4u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_890].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_890].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_890]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_890].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_252 < 2u)
    {
        uint _907 = gl_LocalInvocationIndex + 2u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_907].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_907].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_907]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_907].rotation;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (_252 < 1u)
    {
        uint _924 = gl_LocalInvocationIndex + 1u;
        if (shared_temp[gl_LocalInvocationIndex].error > shared_temp[_924].error)
        {
            shared_temp[gl_LocalInvocationIndex].error = shared_temp[_924].error;
            shared_temp[gl_LocalInvocationIndex]._partition = shared_temp[_924]._partition;
            shared_temp[gl_LocalInvocationIndex].rotation = shared_temp[_924].rotation;
        }
        if (((device uint*)&(*spvDescriptorSet0.g_InBuff)._m0[_250])[0] > shared_temp[gl_LocalInvocationIndex].error)
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_250] = uint4(shared_temp[gl_LocalInvocationIndex].error, (*spvDescriptorSet0.cbCS).g_mode_id, shared_temp[gl_LocalInvocationIndex]._partition, shared_temp[gl_LocalInvocationIndex].rotation);
        }
        else
        {
            (*spvDescriptorSet0.g_OutBuff)._m0[_250] = (*spvDescriptorSet0.g_InBuff)._m0[_250];
        }
    }
}

