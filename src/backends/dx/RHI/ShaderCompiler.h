#pragma once
#include <Common/GFXUtil.h>
#include <ast/function.h>
namespace luisa::compute {

class ShaderCompiler {
public:
	struct ConstBufferData {
		HashMap<uint, size_t> offsets;
		size_t cbufferSize;
		ConstBufferData() {}
	};
	static void TryCompileCompute(Function func);
	static ConstBufferData const& GetCBufferData(Function func);

private:
};
}// namespace luisa::compute