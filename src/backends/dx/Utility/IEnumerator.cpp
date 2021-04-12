#include <Utility/IEnumerator.h>
bool IEnumerator::ExecuteOne() {
	if (startIndex >= executors.size()) return false;
	if (executors[startIndex]()) {
		startIndex++;
	}
	return true;
}
IEnumerator::Executor::Executor(const Executor& exe) : disposeFunc(exe.disposeFunc),
													   funcPtr(exe.funcPtr),
													   constructPtr(exe.constructPtr) {
	constructPtr(c, (void*)exe.c);
}
IEnumerator::Executor::~Executor() {
	if (disposeFunc) disposeFunc(c);
}
bool IEnumerator::Executor::operator()() {
	return funcPtr(c);
}
void IEnumerator::Executor::operator=(const Executor& exe) {
	if (disposeFunc) disposeFunc(c);
	disposeFunc = exe.disposeFunc;
	funcPtr = exe.funcPtr;
	constructPtr = exe.constructPtr;
	constructPtr(c, (void*)exe.c);
}
