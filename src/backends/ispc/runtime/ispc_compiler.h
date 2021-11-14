#pragma once
#include <vstl/Common.h>
#include <vstl/MD5.h>
#include <vstl/StringUtility.h>
#include <vstl/file_system.h>
namespace lc::ispc {
class Compiler {
private:
public:
    luisa::string CompileCode(
        std::string_view code) const;
};
}// namespace lc::ispc