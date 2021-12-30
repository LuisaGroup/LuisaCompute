#pragma once
#include <vstl/Common.h>
#include <runtime/command.h>
#include <ast/variable.h>
using namespace luisa;
using namespace luisa::compute;
namespace toolhub::directx {
class ArgumentPacker {
    static void PackData(
        vstd::Iterator<std::pair<Variable const *, vstd::span<vbyte const>>> const &allVars,
        uint3 dispatchSize,
        vstd::vector<vbyte> &data);
};
}// namespace toolhub::directx
