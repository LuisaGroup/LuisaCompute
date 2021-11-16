#pragma once

#include <vstl/Common.h>
#include <vstl/MD5.h>
#include <vstl/StringUtility.h>
#include <vstl/file_system.h>

#include <runtime/context.h>
#include <backends/ispc/runtime/ispc_jit_module.h>

namespace lc::ispc {

using luisa::compute::Context;

struct Compiler {
     JITModule CompileCode(
        const Context &ctx,
        std::string_view code) const;
};

}// namespace lc::ispc
