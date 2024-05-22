#pragma once
#include <luisa/core/stl/string.h>
namespace lc_hlsl{
extern char accel_process[];
extern char bindless_upload[];
extern char bc6_encode_block[];
extern char bc6_header[];
extern char bc6_trymode_g10cs[];
extern char bc6_trymode_le10cs[];
extern char bc7_encode_block[];
extern char bc7_header[];
extern char bc7_trymode_02cs[];
extern char bc7_trymode_137cs[];
extern char bc7_trymode_456cs[];
extern char hlsl_header[];
extern char raytracing_header[];
extern char tex2d_bindless[];
extern char tex3d_bindless[];
extern char compute_quad[];
extern char determinant[];
extern char inverse[];
extern char indirect[];
extern char resource_size[];
extern char accel_header[];
extern char copy_sign[];
extern char bindless_common[];
extern char auto_diff[];
extern char reduce[];
static luisa::string_view get_hlsl_builtin(luisa::string_view ss) {
    struct Dict {
        luisa::unordered_map<luisa::string_view, luisa::string_view> dict;
        Dict(){
			dict.try_emplace("accel_process", luisa::string_view{accel_process, 1431});
			dict.try_emplace("bindless_upload", luisa::string_view{bindless_upload, 315});
			dict.try_emplace("bc6_encode_block", luisa::string_view{bc6_encode_block, 10621});
			dict.try_emplace("bc6_header", luisa::string_view{bc6_header, 79069});
			dict.try_emplace("bc6_trymode_g10cs", luisa::string_view{bc6_trymode_g10cs, 6314});
			dict.try_emplace("bc6_trymode_le10cs", luisa::string_view{bc6_trymode_le10cs, 6643});
			dict.try_emplace("bc7_encode_block", luisa::string_view{bc7_encode_block, 9030});
			dict.try_emplace("bc7_header", luisa::string_view{bc7_header, 25349});
			dict.try_emplace("bc7_trymode_02cs", luisa::string_view{bc7_trymode_02cs, 7558});
			dict.try_emplace("bc7_trymode_137cs", luisa::string_view{bc7_trymode_137cs, 7584});
			dict.try_emplace("bc7_trymode_456cs", luisa::string_view{bc7_trymode_456cs, 10817});
			dict.try_emplace("hlsl_header", luisa::string_view{hlsl_header, 4820});
			dict.try_emplace("raytracing_header", luisa::string_view{raytracing_header, 3495});
			dict.try_emplace("tex2d_bindless", luisa::string_view{tex2d_bindless, 2505});
			dict.try_emplace("tex3d_bindless", luisa::string_view{tex3d_bindless, 2529});
			dict.try_emplace("compute_quad", luisa::string_view{compute_quad, 130});
			dict.try_emplace("determinant", luisa::string_view{determinant, 1971});
			dict.try_emplace("inverse", luisa::string_view{inverse, 2834});
			dict.try_emplace("indirect", luisa::string_view{indirect, 478});
			dict.try_emplace("resource_size", luisa::string_view{resource_size, 1296});
			dict.try_emplace("accel_header", luisa::string_view{accel_header, 1126});
			dict.try_emplace("copy_sign", luisa::string_view{copy_sign, 757});
			dict.try_emplace("bindless_common", luisa::string_view{bindless_common, 704});
			dict.try_emplace("auto_diff", luisa::string_view{auto_diff, 1771});
			dict.try_emplace("reduce", luisa::string_view{reduce, 6276});
		}
	};
	static Dict dict;
	return dict.dict.find(ss)->second;
}
}