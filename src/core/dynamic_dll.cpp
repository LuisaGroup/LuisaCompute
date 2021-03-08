#include "dynamic_dll.h"
#ifdef _WINDOWS
namespace luisa::compute {

DynamicDLL::DynamicDLL(char const *name) {
    inst = LoadLibraryA(name);
}
DynamicDLL::~DynamicDLL() {
    if (inst)
        FreeLibrary(inst);
}
}// namespace luisa::compute
#endif