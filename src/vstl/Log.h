#pragma once

#include <vstl/config.h>
#include <initializer_list>
#include <vstl/MetaLib.h>

namespace vstd {

LUISA_DLL void vstl_log(char const *chunk);
LUISA_DLL void vstl_log(char const *const *chunk, size_t chunkCount);
LUISA_DLL void vstl_log(std::initializer_list<char const *> initList);
LUISA_DLL void vstl_log_error_pure_virtual(Type tarType);

//#define VENGINE_PURE_VIRTUAL                    \
//    [&] {                                       \
//        vstl_log_error_pure_virtual(typeid(*this)); \
//    }()
//
//#define NOT_IMPLEMENT_EXCEPTION(T)                        \
//    [&] {                                                 \
//        vstl_log({#T##_sv, " not implemented!\n"_sv}); \
//        VSTL_ABORT();                                     \
//    }()

}// namespace vstd
