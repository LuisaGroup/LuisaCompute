#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ir/ir.h>

namespace luisa::compute::ir_v2 {

class Ref2Ret {

private:
    struct Context {
        NodeRef ret;
        luisa::vector<NodeRef> ref_args;
        luisa::vector<uint> ref_arg_indices;
    };

    struct Metadata {
        CArc<raw::CallableModule> module;
        luisa::vector<uint> ref_arg_indices;
    };

private:
    luisa::unordered_map<const CallableModule *, Metadata> _processed;
    Context *_current{nullptr};

private:
    [[nodiscard]] static bool _needs_transform(CallableModule *m) noexcept;
    static Context _prepare_context(CallableModule *m) noexcept;

private:
    void _transform(const CArc<KernelModule> &m) noexcept;
    void _transform(const CArc<CallableModule> &m) noexcept;
    void _transform(BasicBlock *bb) noexcept;
    void _transform_return(Instruction::Return *ret) noexcept;
    void _transform_call(Instruction::Call *call) noexcept;

public:
    // TODO
};

}// namespace luisa::compute::ir_v2
