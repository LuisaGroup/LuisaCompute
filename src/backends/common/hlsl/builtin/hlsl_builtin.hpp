#pragma once
#include <luisa/core/stl/string.h>
namespace lc_hlsl{
extern unsigned char accel_process[];
extern unsigned char bindless_upload[];
extern unsigned char bc6_encode_block[];
extern unsigned char bc6_header[];
extern unsigned char bc6_trymode_g10cs[];
extern unsigned char bc6_trymode_le10cs[];
extern unsigned char bc7_encode_block[];
extern unsigned char bc7_header[];
extern unsigned char bc7_trymode_02cs[];
extern unsigned char bc7_trymode_137cs[];
extern unsigned char bc7_trymode_456cs[];
extern unsigned char hlsl_header[];
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
struct HLSLCompressedHeader {
    void const* ptr;
    size_t compressed_size;
    size_t uncompressed_size;
};
static HLSLCompressedHeader get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, HLSLCompressedHeader> dict;
        Dict(){
			dict.try_emplace("accel_process", HLSLCompressedHeader{accel_process, 525, 1491});
			dict.try_emplace("bindless_upload", HLSLCompressedHeader{bindless_upload, 225, 334});
			dict.try_emplace("bc6_encode_block", HLSLCompressedHeader{bc6_encode_block, 1805, 11021});
			dict.try_emplace("bc6_header", HLSLCompressedHeader{bc6_header, 6209, 80770});
			dict.try_emplace("bc6_trymode_g10cs", HLSLCompressedHeader{bc6_trymode_g10cs, 1311, 6513});
			dict.try_emplace("bc6_trymode_le10cs", HLSLCompressedHeader{bc6_trymode_le10cs, 1581, 6872});
			dict.try_emplace("bc7_encode_block", HLSLCompressedHeader{bc7_encode_block, 1809, 9429});
			dict.try_emplace("bc7_header", HLSLCompressedHeader{bc7_header, 4517, 25981});
			dict.try_emplace("bc7_trymode_02cs", HLSLCompressedHeader{bc7_trymode_02cs, 1782, 7820});
			dict.try_emplace("bc7_trymode_137cs", HLSLCompressedHeader{bc7_trymode_137cs, 1814, 7852});
			dict.try_emplace("bc7_trymode_456cs", HLSLCompressedHeader{bc7_trymode_456cs, 2391, 11171});
			dict.try_emplace("hlsl_header", HLSLCompressedHeader{hlsl_header, 1411, 5596});
			dict.try_emplace("raytracing_header", HLSLCompressedHeader{raytracing_header, 868, 3614});
			dict.try_emplace("tex2d_bindless", HLSLCompressedHeader{tex2d_bindless, 551, 4136});
			dict.try_emplace("tex3d_bindless", HLSLCompressedHeader{tex3d_bindless, 507, 3535});
			dict.try_emplace("compute_quad", HLSLCompressedHeader{compute_quad, 87, 138});
			dict.try_emplace("determinant", HLSLCompressedHeader{determinant, 525, 2019});
			dict.try_emplace("inverse", HLSLCompressedHeader{inverse, 681, 2907});
			dict.try_emplace("indirect", HLSLCompressedHeader{indirect, 238, 494});
			dict.try_emplace("resource_size", HLSLCompressedHeader{resource_size, 227, 1374});
			dict.try_emplace("accel_header", HLSLCompressedHeader{accel_header, 391, 1160});
			dict.try_emplace("copy_sign", HLSLCompressedHeader{copy_sign, 175, 765});
			dict.try_emplace("bindless_common", HLSLCompressedHeader{bindless_common, 308, 724});
			dict.try_emplace("auto_diff", HLSLCompressedHeader{auto_diff, 345, 1795});
			dict.try_emplace("reduce", HLSLCompressedHeader{reduce, 743, 6372});
		}
	};
	static Dict dict;
	return dict.dict.find(ss)->second;
}
}