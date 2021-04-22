#pragma once
#include <Common/GFXUtil.h>
namespace luisa::compute {

class ShaderCompiler {
public:
	struct ConstBufferData {
		HashMap<uint, size_t> offsets;
		size_t cbufferSize;
		ConstBufferData() {}
	};
	static void TryCompileCompute(uint32_t uid);
	static ConstBufferData const& GetCBufferData(uint kernel_uid);

private:
};
}// namespace luisa::compute