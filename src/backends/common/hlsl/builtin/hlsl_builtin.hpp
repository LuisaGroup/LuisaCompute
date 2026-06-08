#pragma once
#include <luisa/core/stl/string.h>
#include <cstdint>

#ifdef LUISA_BIN_2_OBJ
#define LC_HLSL_DECL_VARNAME(VAR_NAME) \
    extern const uint8_t _binary_##VAR_NAME##_start[];   \
    extern const uint8_t _binary_##VAR_NAME##_end[];

#define LC_HLSL_INSERT_VARNAME(VAR_NAME, KEY_NAME) \
    dict.try_emplace(KEY_NAME, HLSLCompressedHeader{_binary_##VAR_NAME##_start, (size_t)(_binary_##VAR_NAME##_end - _binary_##VAR_NAME##_start)});

#else

#define LC_HLSL_DECL_VARNAME(VAR_NAME) \
    extern const unsigned char VAR_NAME[];   \
    extern const unsigned long long VAR_NAME##_size;

#define LC_HLSL_INSERT_VARNAME(VAR_NAME, KEY_NAME) \
    dict.try_emplace(KEY_NAME, HLSLCompressedHeader{VAR_NAME, VAR_NAME##_size});

#endif

extern "C" {
LC_HLSL_DECL_VARNAME(accel_process_vk_motion_bytes)
LC_HLSL_DECL_VARNAME(accel_process_vk_bytes)
LC_HLSL_DECL_VARNAME(bindless_upload_bytes)
LC_HLSL_DECL_VARNAME(hlsl_header_bytes)
LC_HLSL_DECL_VARNAME(dx_linalg_bytes)
LC_HLSL_DECL_VARNAME(hlsl_header_fallback_bytes)
LC_HLSL_DECL_VARNAME(raytracing_header_bytes)
LC_HLSL_DECL_VARNAME(raytracing_motion_header_bytes)
LC_HLSL_DECL_VARNAME(tex2d_bindless_bytes)
LC_HLSL_DECL_VARNAME(tex3d_bindless_bytes)
LC_HLSL_DECL_VARNAME(compute_quad_bytes)
LC_HLSL_DECL_VARNAME(determinant_bytes)
LC_HLSL_DECL_VARNAME(inverse_bytes)
LC_HLSL_DECL_VARNAME(indirect_bytes)
LC_HLSL_DECL_VARNAME(resource_size_bytes)
LC_HLSL_DECL_VARNAME(accel_header_bytes)
LC_HLSL_DECL_VARNAME(accel_process_bytes)
LC_HLSL_DECL_VARNAME(copy_sign_bytes)
LC_HLSL_DECL_VARNAME(bindless_common_bytes)
LC_HLSL_DECL_VARNAME(auto_diff_bytes)
LC_HLSL_DECL_VARNAME(reduce_bytes)
LC_HLSL_DECL_VARNAME(accel_process_vk_dxil)
LC_HLSL_DECL_VARNAME(load_bdls_dxil)
LC_HLSL_DECL_VARNAME(load_bdls_vk_dxil)
LC_HLSL_DECL_VARNAME(set_accel4_dxil)
LC_HLSL_DECL_VARNAME(bc6_encodeblock_dxil)
LC_HLSL_DECL_VARNAME(bc6_trymodeg10_dxil)
LC_HLSL_DECL_VARNAME(bc6_trymodele10_dxil)
LC_HLSL_DECL_VARNAME(bc7_encodeblock_dxil)
LC_HLSL_DECL_VARNAME(bc7_trymode02_dxil)
LC_HLSL_DECL_VARNAME(bc7_trymode137_dxil)
LC_HLSL_DECL_VARNAME(bc7_trymode456_dxil)
LC_HLSL_DECL_VARNAME(spv_alias_bytes)
LC_HLSL_DECL_VARNAME(bindless_upload_vk_bytes)
}

namespace lc_hlsl {
struct HLSLCompressedHeader {
    void const *ptr{};
    size_t size{};
};
static HLSLCompressedHeader get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, HLSLCompressedHeader> dict;
        Dict() {
            LC_HLSL_INSERT_VARNAME(bindless_upload_vk_bytes, "bindless_upload_vk.bytes")
            LC_HLSL_INSERT_VARNAME(accel_process_vk_bytes, "accel_process_vk.bytes")
            LC_HLSL_INSERT_VARNAME(hlsl_header_bytes, "hlsl_header")
            LC_HLSL_INSERT_VARNAME(spv_alias_bytes, "spv_alias")
            LC_HLSL_INSERT_VARNAME(dx_linalg_bytes, "dx_linalg")
            LC_HLSL_INSERT_VARNAME(hlsl_header_fallback_bytes, "hlsl_header_fallback")
            LC_HLSL_INSERT_VARNAME(raytracing_header_bytes, "raytracing_header")
            LC_HLSL_INSERT_VARNAME(raytracing_motion_header_bytes, "raytracing_motion_header")
            LC_HLSL_INSERT_VARNAME(tex2d_bindless_bytes, "tex2d_bindless")
            LC_HLSL_INSERT_VARNAME(tex3d_bindless_bytes, "tex3d_bindless")
            LC_HLSL_INSERT_VARNAME(compute_quad_bytes, "compute_quad")
            LC_HLSL_INSERT_VARNAME(bindless_upload_bytes, "bindless_upload.bytes")
            LC_HLSL_INSERT_VARNAME(determinant_bytes, "determinant")
            LC_HLSL_INSERT_VARNAME(inverse_bytes, "inverse")
            LC_HLSL_INSERT_VARNAME(indirect_bytes, "indirect")
            LC_HLSL_INSERT_VARNAME(resource_size_bytes, "resource_size")
            LC_HLSL_INSERT_VARNAME(accel_header_bytes, "accel_header")
            LC_HLSL_INSERT_VARNAME(copy_sign_bytes, "copy_sign")
            LC_HLSL_INSERT_VARNAME(bindless_common_bytes, "bindless_common")
            LC_HLSL_INSERT_VARNAME(auto_diff_bytes, "auto_diff")
            LC_HLSL_INSERT_VARNAME(reduce_bytes, "reduce")
            LC_HLSL_INSERT_VARNAME(accel_process_bytes, "accel_process.bytes")
            LC_HLSL_INSERT_VARNAME(accel_process_vk_dxil, "accel_process_vk.dxil")
            LC_HLSL_INSERT_VARNAME(load_bdls_dxil, "load_bdls.dxil")
            LC_HLSL_INSERT_VARNAME(load_bdls_vk_dxil, "load_bdls_vk.dxil")
            LC_HLSL_INSERT_VARNAME(set_accel4_dxil, "set_accel4.dxil")
            LC_HLSL_INSERT_VARNAME(bc6_encodeblock_dxil, "bc6_encodeblock.dxil")
            LC_HLSL_INSERT_VARNAME(bc6_trymodeg10_dxil, "bc6_trymodeg10.dxil")
            LC_HLSL_INSERT_VARNAME(bc6_trymodele10_dxil, "bc6_trymodele10.dxil")
            LC_HLSL_INSERT_VARNAME(bc7_encodeblock_dxil, "bc7_encodeblock.dxil")
            LC_HLSL_INSERT_VARNAME(bc7_trymode02_dxil, "bc7_trymode02.dxil")
            LC_HLSL_INSERT_VARNAME(bc7_trymode137_dxil, "bc7_trymode137.dxil")
            LC_HLSL_INSERT_VARNAME(bc7_trymode456_dxil, "bc7_trymode456.dxil")
            LC_HLSL_INSERT_VARNAME(accel_process_vk_motion_bytes, "accel_process_vk_motion.bytes")
        }
    };
    static Dict dict;
    auto iter = dict.dict.find(ss);
    if (iter == dict.dict.end()) return {};
    return iter->second;
}
}// namespace lc_hlsl