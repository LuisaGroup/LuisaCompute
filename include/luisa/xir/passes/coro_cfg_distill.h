#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

class BasicBlock;
class Function;
class Module;

struct CoroCfgDistillResult {

    struct Scope {
        luisa::vector<BasicBlock *> blocks;
        int scope_id{0};
        luisa::optional<uint32_t> suspend_token;
        luisa::optional<luisa::string> suspend_name;
        bool is_terminal{false};
    };

    luisa::vector<Scope> scopes;
    luisa::vector<luisa::vector<size_t>> edges;
};

[[nodiscard]] LUISA_XIR_API CoroCfgDistillResult coro_cfg_distill_pass_run_on_function(Function *f) noexcept;
[[nodiscard]] LUISA_XIR_API size_t coro_cfg_distill_pass_run_on_module(Module *m) noexcept;

}// namespace luisa::compute::xir
