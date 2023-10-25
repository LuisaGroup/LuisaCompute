#pragma once
#include <luisa/ir_v2/ir_v2_fwd.h>

namespace luisa::compute::ir_v2 {
template<class T>
struct Slice {
    T *data = nullptr;
    size_t len = 0;
    constexpr Slice() noexcept = default;
    constexpr Slice(T *data, size_t len) noexcept : data(data), len(len) {}
#ifndef BINDGEN
    // construct from array
    template<size_t N>
    constexpr Slice(T (&arr)[N]) noexcept : data(arr), len(N) {}
    // construct from std::array
    template<size_t N>
    constexpr Slice(std::array<T, N> &arr) noexcept : data(arr.data()), len(N) {}
    // construct from luisa::vector
    constexpr Slice(luisa::vector<T> &vec) noexcept : data(vec.data()), len(vec.size()) {}
    // construct from luisa::span
    constexpr Slice(luisa::span<T> &span) noexcept : data(span.data()), len(span.size()) {}
    // construct from luisa::string
    constexpr Slice(luisa::string &str) noexcept : data(str.data()), len(str.size()) {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
    }
    luisa::vector<T> to_vector() const noexcept {
        return luisa::vector<T>(data, data + len);
    }
    luisa::string to_string() const noexcept {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
        return luisa::string(data, len);
    }
#endif
};
}// namespace luisa::compute::ir_v2

namespace luisa::compute::ir_v2 {
/**
* <div rustbindgen nocopy></div>
*/
struct CFunc {
    void *data;
    FuncTag tag;
};
static_assert(sizeof(CFunc) == 16);
/**
* <div rustbindgen nocopy></div>
*/
struct CInstruction {
    void *data;
    InstructionTag tag;
};
static_assert(sizeof(CInstruction) == 16);
/**
* <div rustbindgen nocopy></div>
*/
struct CBinding {
    void *data;
    BindingTag tag;
};
static_assert(sizeof(CBinding) == 16);
struct IrV2BindingTable {
    AssumeFn *(*Func_as_AssumeFn)(CFunc *self);
    UnreachableFn *(*Func_as_UnreachableFn)(CFunc *self);
    BindlessAtomicExchangeFn *(*Func_as_BindlessAtomicExchangeFn)(CFunc *self);
    BindlessAtomicCompareExchangeFn *(*Func_as_BindlessAtomicCompareExchangeFn)(CFunc *self);
    BindlessAtomicFetchAddFn *(*Func_as_BindlessAtomicFetchAddFn)(CFunc *self);
    BindlessAtomicFetchSubFn *(*Func_as_BindlessAtomicFetchSubFn)(CFunc *self);
    BindlessAtomicFetchAndFn *(*Func_as_BindlessAtomicFetchAndFn)(CFunc *self);
    BindlessAtomicFetchOrFn *(*Func_as_BindlessAtomicFetchOrFn)(CFunc *self);
    BindlessAtomicFetchXorFn *(*Func_as_BindlessAtomicFetchXorFn)(CFunc *self);
    BindlessAtomicFetchMinFn *(*Func_as_BindlessAtomicFetchMinFn)(CFunc *self);
    BindlessAtomicFetchMaxFn *(*Func_as_BindlessAtomicFetchMaxFn)(CFunc *self);
    CallableFn *(*Func_as_CallableFn)(CFunc *self);
    CpuExtFn *(*Func_as_CpuExtFn)(CFunc *self);
    FuncTag (*Func_tag)(Func *self);
    Slice<const char> (*AssumeFn_msg)(AssumeFn *self);
    void (*AssumeFn_set_msg)(AssumeFn *self, Slice<const char> value);
    CFunc (*AssumeFn_new)(Pool *pool, Slice<const char> msg);
    Slice<const char> (*UnreachableFn_msg)(UnreachableFn *self);
    void (*UnreachableFn_set_msg)(UnreachableFn *self, Slice<const char> value);
    CFunc (*UnreachableFn_new)(Pool *pool, Slice<const char> msg);
    const Type *(*BindlessAtomicExchangeFn_ty)(BindlessAtomicExchangeFn *self);
    void (*BindlessAtomicExchangeFn_set_ty)(BindlessAtomicExchangeFn *self, const Type *value);
    CFunc (*BindlessAtomicExchangeFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicCompareExchangeFn_ty)(BindlessAtomicCompareExchangeFn *self);
    void (*BindlessAtomicCompareExchangeFn_set_ty)(BindlessAtomicCompareExchangeFn *self, const Type *value);
    CFunc (*BindlessAtomicCompareExchangeFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchAddFn_ty)(BindlessAtomicFetchAddFn *self);
    void (*BindlessAtomicFetchAddFn_set_ty)(BindlessAtomicFetchAddFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchAddFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchSubFn_ty)(BindlessAtomicFetchSubFn *self);
    void (*BindlessAtomicFetchSubFn_set_ty)(BindlessAtomicFetchSubFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchSubFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchAndFn_ty)(BindlessAtomicFetchAndFn *self);
    void (*BindlessAtomicFetchAndFn_set_ty)(BindlessAtomicFetchAndFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchAndFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchOrFn_ty)(BindlessAtomicFetchOrFn *self);
    void (*BindlessAtomicFetchOrFn_set_ty)(BindlessAtomicFetchOrFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchOrFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchXorFn_ty)(BindlessAtomicFetchXorFn *self);
    void (*BindlessAtomicFetchXorFn_set_ty)(BindlessAtomicFetchXorFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchXorFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchMinFn_ty)(BindlessAtomicFetchMinFn *self);
    void (*BindlessAtomicFetchMinFn_set_ty)(BindlessAtomicFetchMinFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchMinFn_new)(Pool *pool, const Type *ty);
    const Type *(*BindlessAtomicFetchMaxFn_ty)(BindlessAtomicFetchMaxFn *self);
    void (*BindlessAtomicFetchMaxFn_set_ty)(BindlessAtomicFetchMaxFn *self, const Type *value);
    CFunc (*BindlessAtomicFetchMaxFn_new)(Pool *pool, const Type *ty);
    CallableModule *(*CallableFn_module)(CallableFn *self);
    void (*CallableFn_set_module)(CallableFn *self, CallableModule *value);
    CFunc (*CallableFn_new)(Pool *pool, CallableModule *module);
    CpuExternFn (*CpuExtFn_f)(CpuExtFn *self);
    void (*CpuExtFn_set_f)(CpuExtFn *self, CpuExternFn value);
    CFunc (*CpuExtFn_new)(Pool *pool, CpuExternFn f);
    CFunc (*Func_new)(Pool *pool, FuncTag tag);
    ArgumentInst *(*Instruction_as_ArgumentInst)(CInstruction *self);
    ConstantInst *(*Instruction_as_ConstantInst)(CInstruction *self);
    CallInst *(*Instruction_as_CallInst)(CInstruction *self);
    PhiInst *(*Instruction_as_PhiInst)(CInstruction *self);
    IfInst *(*Instruction_as_IfInst)(CInstruction *self);
    GenericLoopInst *(*Instruction_as_GenericLoopInst)(CInstruction *self);
    SwitchInst *(*Instruction_as_SwitchInst)(CInstruction *self);
    LocalInst *(*Instruction_as_LocalInst)(CInstruction *self);
    ReturnInst *(*Instruction_as_ReturnInst)(CInstruction *self);
    PrintInst *(*Instruction_as_PrintInst)(CInstruction *self);
    UpdateInst *(*Instruction_as_UpdateInst)(CInstruction *self);
    RayQueryInst *(*Instruction_as_RayQueryInst)(CInstruction *self);
    RevAutodiffInst *(*Instruction_as_RevAutodiffInst)(CInstruction *self);
    FwdAutodiffInst *(*Instruction_as_FwdAutodiffInst)(CInstruction *self);
    InstructionTag (*Instruction_tag)(Instruction *self);
    bool (*ArgumentInst_by_value)(ArgumentInst *self);
    void (*ArgumentInst_set_by_value)(ArgumentInst *self, bool value);
    CInstruction (*ArgumentInst_new)(Pool *pool, bool by_value);
    const Type *(*ConstantInst_ty)(ConstantInst *self);
    Slice<uint8_t> (*ConstantInst_value)(ConstantInst *self);
    void (*ConstantInst_set_ty)(ConstantInst *self, const Type *value);
    void (*ConstantInst_set_value)(ConstantInst *self, Slice<uint8_t> value);
    CInstruction (*ConstantInst_new)(Pool *pool, const Type *ty, Slice<uint8_t> value);
    const CFunc *(*CallInst_func)(CallInst *self);
    Slice<Node *> (*CallInst_args)(CallInst *self);
    void (*CallInst_set_func)(CallInst *self, CFunc value);
    void (*CallInst_set_args)(CallInst *self, Slice<Node *> value);
    CInstruction (*CallInst_new)(Pool *pool, CFunc func, Slice<Node *> args);
    Slice<PhiIncoming> (*PhiInst_incomings)(PhiInst *self);
    void (*PhiInst_set_incomings)(PhiInst *self, Slice<PhiIncoming> value);
    CInstruction (*PhiInst_new)(Pool *pool, Slice<PhiIncoming> incomings);
    Node *(*IfInst_cond)(IfInst *self);
    BasicBlock *(*IfInst_true_branch)(IfInst *self);
    BasicBlock *(*IfInst_false_branch)(IfInst *self);
    void (*IfInst_set_cond)(IfInst *self, Node *value);
    void (*IfInst_set_true_branch)(IfInst *self, BasicBlock *value);
    void (*IfInst_set_false_branch)(IfInst *self, BasicBlock *value);
    CInstruction (*IfInst_new)(Pool *pool, Node *cond, BasicBlock *true_branch, BasicBlock *false_branch);
    BasicBlock *(*GenericLoopInst_prepare)(GenericLoopInst *self);
    Node *(*GenericLoopInst_cond)(GenericLoopInst *self);
    BasicBlock *(*GenericLoopInst_body)(GenericLoopInst *self);
    BasicBlock *(*GenericLoopInst_update)(GenericLoopInst *self);
    void (*GenericLoopInst_set_prepare)(GenericLoopInst *self, BasicBlock *value);
    void (*GenericLoopInst_set_cond)(GenericLoopInst *self, Node *value);
    void (*GenericLoopInst_set_body)(GenericLoopInst *self, BasicBlock *value);
    void (*GenericLoopInst_set_update)(GenericLoopInst *self, BasicBlock *value);
    CInstruction (*GenericLoopInst_new)(Pool *pool, BasicBlock *prepare, Node *cond, BasicBlock *body, BasicBlock *update);
    Node *(*SwitchInst_value)(SwitchInst *self);
    Slice<SwitchCase> (*SwitchInst_cases)(SwitchInst *self);
    BasicBlock *(*SwitchInst_default_)(SwitchInst *self);
    void (*SwitchInst_set_value)(SwitchInst *self, Node *value);
    void (*SwitchInst_set_cases)(SwitchInst *self, Slice<SwitchCase> value);
    void (*SwitchInst_set_default_)(SwitchInst *self, BasicBlock *value);
    CInstruction (*SwitchInst_new)(Pool *pool, Node *value, Slice<SwitchCase> cases, BasicBlock *default_);
    Node *(*LocalInst_init)(LocalInst *self);
    void (*LocalInst_set_init)(LocalInst *self, Node *value);
    CInstruction (*LocalInst_new)(Pool *pool, Node *init);
    Node *(*ReturnInst_value)(ReturnInst *self);
    void (*ReturnInst_set_value)(ReturnInst *self, Node *value);
    CInstruction (*ReturnInst_new)(Pool *pool, Node *value);
    Slice<const char> (*PrintInst_fmt)(PrintInst *self);
    Slice<Node *> (*PrintInst_args)(PrintInst *self);
    void (*PrintInst_set_fmt)(PrintInst *self, Slice<const char> value);
    void (*PrintInst_set_args)(PrintInst *self, Slice<Node *> value);
    CInstruction (*PrintInst_new)(Pool *pool, Slice<const char> fmt, Slice<Node *> args);
    Node *(*UpdateInst_var)(UpdateInst *self);
    Node *(*UpdateInst_value)(UpdateInst *self);
    void (*UpdateInst_set_var)(UpdateInst *self, Node *value);
    void (*UpdateInst_set_value)(UpdateInst *self, Node *value);
    CInstruction (*UpdateInst_new)(Pool *pool, Node *var, Node *value);
    Node *(*RayQueryInst_query)(RayQueryInst *self);
    BasicBlock *(*RayQueryInst_on_triangle_hit)(RayQueryInst *self);
    BasicBlock *(*RayQueryInst_on_procedural_hit)(RayQueryInst *self);
    void (*RayQueryInst_set_query)(RayQueryInst *self, Node *value);
    void (*RayQueryInst_set_on_triangle_hit)(RayQueryInst *self, BasicBlock *value);
    void (*RayQueryInst_set_on_procedural_hit)(RayQueryInst *self, BasicBlock *value);
    CInstruction (*RayQueryInst_new)(Pool *pool, Node *query, BasicBlock *on_triangle_hit, BasicBlock *on_procedural_hit);
    BasicBlock *(*RevAutodiffInst_body)(RevAutodiffInst *self);
    void (*RevAutodiffInst_set_body)(RevAutodiffInst *self, BasicBlock *value);
    CInstruction (*RevAutodiffInst_new)(Pool *pool, BasicBlock *body);
    BasicBlock *(*FwdAutodiffInst_body)(FwdAutodiffInst *self);
    void (*FwdAutodiffInst_set_body)(FwdAutodiffInst *self, BasicBlock *value);
    CInstruction (*FwdAutodiffInst_new)(Pool *pool, BasicBlock *body);
    CInstruction (*Instruction_new)(Pool *pool, InstructionTag tag);
    const FuncMetadata *(*func_metadata)();
    BufferBinding *(*Binding_as_BufferBinding)(CBinding *self);
    TextureBinding *(*Binding_as_TextureBinding)(CBinding *self);
    BindlessArrayBinding *(*Binding_as_BindlessArrayBinding)(CBinding *self);
    AccelBinding *(*Binding_as_AccelBinding)(CBinding *self);
    BindingTag (*Binding_tag)(Binding *self);
    uint64_t (*BufferBinding_handle)(BufferBinding *self);
    uint64_t (*BufferBinding_offset)(BufferBinding *self);
    uint64_t (*BufferBinding_size)(BufferBinding *self);
    void (*BufferBinding_set_handle)(BufferBinding *self, uint64_t value);
    void (*BufferBinding_set_offset)(BufferBinding *self, uint64_t value);
    void (*BufferBinding_set_size)(BufferBinding *self, uint64_t value);
    CBinding (*BufferBinding_new)(Pool *pool, uint64_t handle, uint64_t offset, uint64_t size);
    uint64_t (*TextureBinding_handle)(TextureBinding *self);
    uint64_t (*TextureBinding_level)(TextureBinding *self);
    void (*TextureBinding_set_handle)(TextureBinding *self, uint64_t value);
    void (*TextureBinding_set_level)(TextureBinding *self, uint64_t value);
    CBinding (*TextureBinding_new)(Pool *pool, uint64_t handle, uint64_t level);
    uint64_t (*BindlessArrayBinding_handle)(BindlessArrayBinding *self);
    void (*BindlessArrayBinding_set_handle)(BindlessArrayBinding *self, uint64_t value);
    CBinding (*BindlessArrayBinding_new)(Pool *pool, uint64_t handle);
    uint64_t (*AccelBinding_handle)(AccelBinding *self);
    void (*AccelBinding_set_handle)(AccelBinding *self, uint64_t value);
    CBinding (*AccelBinding_new)(Pool *pool, uint64_t handle);
    CBinding (*Binding_new)(Pool *pool, BindingTag tag);
};
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table();
}// namespace luisa::compute::ir_v2
