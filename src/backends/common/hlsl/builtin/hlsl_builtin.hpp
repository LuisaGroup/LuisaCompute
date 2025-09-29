#pragma once
#include <luisa/core/stl/string.h>
#define LC_HLSL_DECL_VARNAME(VAR_NAME) \
    extern const unsigned char VAR_NAME[];   \
    extern const unsigned long long VAR_NAME##_size;

#define LC_HLSL_INSERT_VARNAME(VAR_NAME, KEY_NAME) \
    dict.try_emplace(KEY_NAME, HLSLCompressedHeader{VAR_NAME, VAR_NAME##_size});
extern "C" {
LC_HLSL_DECL_VARNAME(hlsl_header)
LC_HLSL_DECL_VARNAME(dx_linalg)
LC_HLSL_DECL_VARNAME(hlsl_header_fallback)
LC_HLSL_DECL_VARNAME(raytracing_header)
LC_HLSL_DECL_VARNAME(tex2d_bindless)
LC_HLSL_DECL_VARNAME(tex3d_bindless)
LC_HLSL_DECL_VARNAME(compute_quad)
LC_HLSL_DECL_VARNAME(determinant)
LC_HLSL_DECL_VARNAME(inverse)
LC_HLSL_DECL_VARNAME(indirect)
LC_HLSL_DECL_VARNAME(resource_size)
LC_HLSL_DECL_VARNAME(accel_header)
LC_HLSL_DECL_VARNAME(copy_sign)
LC_HLSL_DECL_VARNAME(bindless_common)
LC_HLSL_DECL_VARNAME(auto_diff)
LC_HLSL_DECL_VARNAME(reduce)
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
LC_HLSL_DECL_VARNAME(spv_alias)
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
            LC_HLSL_INSERT_VARNAME(hlsl_header, "hlsl_header")
            LC_HLSL_INSERT_VARNAME(spv_alias, "spv_alias")
            LC_HLSL_INSERT_VARNAME(dx_linalg, "dx_linalg")
            LC_HLSL_INSERT_VARNAME(hlsl_header_fallback, "hlsl_header_fallback")
            LC_HLSL_INSERT_VARNAME(raytracing_header, "raytracing_header")
            LC_HLSL_INSERT_VARNAME(tex2d_bindless, "tex2d_bindless")
            LC_HLSL_INSERT_VARNAME(tex3d_bindless, "tex3d_bindless")
            LC_HLSL_INSERT_VARNAME(compute_quad, "compute_quad")
            LC_HLSL_INSERT_VARNAME(determinant, "determinant")
            LC_HLSL_INSERT_VARNAME(inverse, "inverse")
            LC_HLSL_INSERT_VARNAME(indirect, "indirect")
            LC_HLSL_INSERT_VARNAME(resource_size, "resource_size")
            LC_HLSL_INSERT_VARNAME(accel_header, "accel_header")
            LC_HLSL_INSERT_VARNAME(copy_sign, "copy_sign")
            LC_HLSL_INSERT_VARNAME(bindless_common, "bindless_common")
            LC_HLSL_INSERT_VARNAME(auto_diff, "auto_diff")
            LC_HLSL_INSERT_VARNAME(reduce, "reduce")
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
        }
    };
    static Dict dict;
    auto iter = dict.dict.find(ss);
    if (iter == dict.dict.end()) return {};
    return iter->second;
}
}// namespace lc_hlsl