#pragma once
#include <luisa/core/stl/string.h>
namespace lc_hlsl{
extern unsigned char hlsl_header[];
extern unsigned char hlsl_header_fallback[];
extern unsigned char raytracing_header[];
extern unsigned char tex2d_bindless[];
extern unsigned char tex3d_bindless[];
extern unsigned char compute_quad[];
extern unsigned char determinant[];
extern unsigned char inverse[];
extern unsigned char indirect[];
extern unsigned char resource_size[];
extern unsigned char accel_header[];
extern unsigned char copy_sign[];
extern unsigned char bindless_common[];
extern unsigned char auto_diff[];
extern unsigned char reduce[];
extern unsigned char accel_process_vk_dxil[];
extern unsigned char load_bdls_dxil[];
extern unsigned char load_bdls_vk_dxil[];
extern unsigned char set_accel4_dxil[];
extern unsigned char bc6_encodeblock_dxil[];
extern unsigned char bc6_trymodeg10_dxil[];
extern unsigned char bc6_trymodele10_dxil[];
extern unsigned char bc7_encodeblock_dxil[];
extern unsigned char bc7_trymode02_dxil[];
extern unsigned char bc7_trymode137_dxil[];
extern unsigned char bc7_trymode456_dxil[];
struct HLSLCompressedHeader {
    void const* ptr{};
    size_t compressed_size{};
    size_t uncompressed_size{};
};
static HLSLCompressedHeader get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, HLSLCompressedHeader> dict;
        Dict(){
			dict.try_emplace("hlsl_header", HLSLCompressedHeader{hlsl_header, 1452, 5648});
			dict.try_emplace("hlsl_header_fallback", HLSLCompressedHeader{hlsl_header_fallback, 1568, 6256});
			dict.try_emplace("raytracing_header", HLSLCompressedHeader{raytracing_header, 889, 3045});
			dict.try_emplace("tex2d_bindless", HLSLCompressedHeader{tex2d_bindless, 738, 8883});
			dict.try_emplace("tex3d_bindless", HLSLCompressedHeader{tex3d_bindless, 676, 7738});
			dict.try_emplace("compute_quad", HLSLCompressedHeader{compute_quad, 87, 138});
			dict.try_emplace("determinant", HLSLCompressedHeader{determinant, 525, 2019});
			dict.try_emplace("inverse", HLSLCompressedHeader{inverse, 681, 2907});
			dict.try_emplace("indirect", HLSLCompressedHeader{indirect, 238, 494});
			dict.try_emplace("resource_size", HLSLCompressedHeader{resource_size, 227, 1374});
			dict.try_emplace("accel_header", HLSLCompressedHeader{accel_header, 391, 1160});
			dict.try_emplace("copy_sign", HLSLCompressedHeader{copy_sign, 175, 765});
			dict.try_emplace("bindless_common", HLSLCompressedHeader{bindless_common, 375, 1615});
			dict.try_emplace("auto_diff", HLSLCompressedHeader{auto_diff, 345, 1795});
			dict.try_emplace("reduce", HLSLCompressedHeader{reduce, 743, 6372});
			dict.try_emplace("accel_process_vk.dxil", HLSLCompressedHeader{accel_process_vk_dxil, 1229, 3604});
			dict.try_emplace("load_bdls.dxil", HLSLCompressedHeader{load_bdls_dxil, 2432, 4168});
			dict.try_emplace("load_bdls_vk.dxil", HLSLCompressedHeader{load_bdls_vk_dxil, 610, 1556});
			dict.try_emplace("set_accel4.dxil", HLSLCompressedHeader{set_accel4_dxil, 3397, 5512});
			dict.try_emplace("bc6_encodeblock.dxil", HLSLCompressedHeader{bc6_encodeblock_dxil, 16807, 24120});
			dict.try_emplace("bc6_trymodeg10.dxil", HLSLCompressedHeader{bc6_trymodeg10_dxil, 9862, 14072});
			dict.try_emplace("bc6_trymodele10.dxil", HLSLCompressedHeader{bc6_trymodele10_dxil, 12813, 17396});
			dict.try_emplace("bc7_encodeblock.dxil", HLSLCompressedHeader{bc7_encodeblock_dxil, 23505, 33400});
			dict.try_emplace("bc7_trymode02.dxil", HLSLCompressedHeader{bc7_trymode02_dxil, 9953, 14032});
			dict.try_emplace("bc7_trymode137.dxil", HLSLCompressedHeader{bc7_trymode137_dxil, 10722, 15088});
			dict.try_emplace("bc7_trymode456.dxil", HLSLCompressedHeader{bc7_trymode456_dxil, 9338, 13692});
		}
	};
	static Dict dict;
	auto iter = dict.dict.find(ss);
	if (iter == dict.dict.end()) return {};
	return iter->second;
}
}