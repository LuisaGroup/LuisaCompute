#include <iostream>
#include "ShaderCompile/ShaderVarUtility.h"
#include <optional>
#include <Common/DLL.h>
#include <Common/DynamicDLL.h>
namespace luisa::compute {
class ScopeStmt;
}
int main() {
	vengine::vengine_init_malloc();
	using namespace luisa::compute;
	static std::optional<DynamicDLL> dll;
	static funcPtr_t<void(ScopeStmt const*)> codegenFunc;
	std::cout << "Fuck!" << std::endl;
	if (!dll.has_value()) {
		dll.emplace("LC_DXBackend.dll");
		codegenFunc = dll->GetDLLFunc<void(ScopeStmt const*)>("CodegenBody");
		std::cout << "Fuck Start!" << std::endl;
	}
	std::cout << codegenFunc << std::endl;
	system("pause");
	codegenFunc(nullptr);
}