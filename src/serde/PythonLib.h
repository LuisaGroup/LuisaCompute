#pragma once
#include <vstl/Common.h>
namespace toolhub::py {
class PythonLibImpl {
public:
	PythonLibImpl();
	~PythonLibImpl();
	void Initialize();
	void Finalize();
	bool ExecutePythonString(char const* c_str);
	bool ExecuteFunc(
		char const* moduleName,
		char const* funcName);
	static PythonLibImpl* Current();
};
}// namespace toolhub::py