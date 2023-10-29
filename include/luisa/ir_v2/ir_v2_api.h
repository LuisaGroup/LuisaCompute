#pragma once
#include <luisa/ir_v2/ir_v2_fwd.h>

namespace luisa::compute::ir_v2 {
/** 
* <div rustbindgen nodebug></div>
*/
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
    AssertFn *(*Func_as_AssertFn)(CFunc *self);
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
    RustyFuncTag (*Func_tag)(const CFunc *self);
    Slice<const char> (*AssumeFn_msg)(AssumeFn *self);
    void (*AssumeFn_set_msg)(AssumeFn *self, Slice<const char> value);
    CFunc (*AssumeFn_new)(Pool *pool, Slice<const char> msg);
    Slice<const char> (*UnreachableFn_msg)(UnreachableFn *self);
    void (*UnreachableFn_set_msg)(UnreachableFn *self, Slice<const char> value);
    CFunc (*UnreachableFn_new)(Pool *pool, Slice<const char> msg);
    Slice<const char> (*AssertFn_msg)(AssertFn *self);
    void (*AssertFn_set_msg)(AssertFn *self, Slice<const char> value);
    CFunc (*AssertFn_new)(Pool *pool, Slice<const char> msg);
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
    CpuExternFn *(*CpuExtFn_f)(CpuExtFn *self);
    void (*CpuExtFn_set_f)(CpuExtFn *self, CpuExternFn *value);
    CFunc (*CpuExtFn_new)(Pool *pool, CpuExternFn *f);
    CFunc (*Func_new)(Pool *pool, RustyFuncTag tag);
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
    CommentInst *(*Instruction_as_CommentInst)(CInstruction *self);
    UpdateInst *(*Instruction_as_UpdateInst)(CInstruction *self);
    RayQueryInst *(*Instruction_as_RayQueryInst)(CInstruction *self);
    RevAutodiffInst *(*Instruction_as_RevAutodiffInst)(CInstruction *self);
    FwdAutodiffInst *(*Instruction_as_FwdAutodiffInst)(CInstruction *self);
    RustyInstructionTag (*Instruction_tag)(const CInstruction *self);
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
    const BasicBlock *(*IfInst_true_branch)(IfInst *self);
    const BasicBlock *(*IfInst_false_branch)(IfInst *self);
    void (*IfInst_set_cond)(IfInst *self, Node *value);
    void (*IfInst_set_true_branch)(IfInst *self, const BasicBlock *value);
    void (*IfInst_set_false_branch)(IfInst *self, const BasicBlock *value);
    CInstruction (*IfInst_new)(Pool *pool, Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch);
    const BasicBlock *(*GenericLoopInst_prepare)(GenericLoopInst *self);
    Node *(*GenericLoopInst_cond)(GenericLoopInst *self);
    const BasicBlock *(*GenericLoopInst_body)(GenericLoopInst *self);
    const BasicBlock *(*GenericLoopInst_update)(GenericLoopInst *self);
    void (*GenericLoopInst_set_prepare)(GenericLoopInst *self, const BasicBlock *value);
    void (*GenericLoopInst_set_cond)(GenericLoopInst *self, Node *value);
    void (*GenericLoopInst_set_body)(GenericLoopInst *self, const BasicBlock *value);
    void (*GenericLoopInst_set_update)(GenericLoopInst *self, const BasicBlock *value);
    CInstruction (*GenericLoopInst_new)(Pool *pool, const BasicBlock *prepare, Node *cond, const BasicBlock *body, const BasicBlock *update);
    Node *(*SwitchInst_value)(SwitchInst *self);
    Slice<SwitchCase> (*SwitchInst_cases)(SwitchInst *self);
    const BasicBlock *(*SwitchInst_default_)(SwitchInst *self);
    void (*SwitchInst_set_value)(SwitchInst *self, Node *value);
    void (*SwitchInst_set_cases)(SwitchInst *self, Slice<SwitchCase> value);
    void (*SwitchInst_set_default_)(SwitchInst *self, const BasicBlock *value);
    CInstruction (*SwitchInst_new)(Pool *pool, Node *value, Slice<SwitchCase> cases, const BasicBlock *default_);
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
    Slice<const char> (*CommentInst_comment)(CommentInst *self);
    void (*CommentInst_set_comment)(CommentInst *self, Slice<const char> value);
    CInstruction (*CommentInst_new)(Pool *pool, Slice<const char> comment);
    Node *(*UpdateInst_var)(UpdateInst *self);
    Node *(*UpdateInst_value)(UpdateInst *self);
    void (*UpdateInst_set_var)(UpdateInst *self, Node *value);
    void (*UpdateInst_set_value)(UpdateInst *self, Node *value);
    CInstruction (*UpdateInst_new)(Pool *pool, Node *var, Node *value);
    Node *(*RayQueryInst_query)(RayQueryInst *self);
    const BasicBlock *(*RayQueryInst_on_triangle_hit)(RayQueryInst *self);
    const BasicBlock *(*RayQueryInst_on_procedural_hit)(RayQueryInst *self);
    void (*RayQueryInst_set_query)(RayQueryInst *self, Node *value);
    void (*RayQueryInst_set_on_triangle_hit)(RayQueryInst *self, const BasicBlock *value);
    void (*RayQueryInst_set_on_procedural_hit)(RayQueryInst *self, const BasicBlock *value);
    CInstruction (*RayQueryInst_new)(Pool *pool, Node *query, const BasicBlock *on_triangle_hit, const BasicBlock *on_procedural_hit);
    const BasicBlock *(*RevAutodiffInst_body)(RevAutodiffInst *self);
    void (*RevAutodiffInst_set_body)(RevAutodiffInst *self, const BasicBlock *value);
    CInstruction (*RevAutodiffInst_new)(Pool *pool, const BasicBlock *body);
    const BasicBlock *(*FwdAutodiffInst_body)(FwdAutodiffInst *self);
    void (*FwdAutodiffInst_set_body)(FwdAutodiffInst *self, const BasicBlock *value);
    CInstruction (*FwdAutodiffInst_new)(Pool *pool, const BasicBlock *body);
    CInstruction (*Instruction_new)(Pool *pool, RustyInstructionTag tag);
    const FuncMetadata *(*func_metadata)();
    BufferBinding *(*Binding_as_BufferBinding)(CBinding *self);
    TextureBinding *(*Binding_as_TextureBinding)(CBinding *self);
    BindlessArrayBinding *(*Binding_as_BindlessArrayBinding)(CBinding *self);
    AccelBinding *(*Binding_as_AccelBinding)(CBinding *self);
    RustyBindingTag (*Binding_tag)(const CBinding *self);
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
    CBinding (*Binding_new)(Pool *pool, RustyBindingTag tag);
    const Type *(*type_extract)(const Type *ty, uint32_t index);
    size_t (*type_size)(const Type *ty);
    size_t (*type_alignment)(const Type *ty);
    RustyTypeTag (*type_tag)(const Type *ty);
    bool (*type_is_scalar)(const Type *ty);
    bool (*type_is_bool)(const Type *ty);
    bool (*type_is_int16)(const Type *ty);
    bool (*type_is_int32)(const Type *ty);
    bool (*type_is_int64)(const Type *ty);
    bool (*type_is_uint16)(const Type *ty);
    bool (*type_is_uint32)(const Type *ty);
    bool (*type_is_uint64)(const Type *ty);
    bool (*type_is_float16)(const Type *ty);
    bool (*type_is_float32)(const Type *ty);
    bool (*type_is_array)(const Type *ty);
    bool (*type_is_vector)(const Type *ty);
    bool (*type_is_struct)(const Type *ty);
    bool (*type_is_custom)(const Type *ty);
    bool (*type_is_matrix)(const Type *ty);
    const Type *(*type_element)(const Type *ty);
    Slice<const char> (*type_description)(const Type *ty);
    size_t (*type_dimension)(const Type *ty);
    Slice<const Type *const> (*type_members)(const Type *ty);
    const Type *(*make_struct)(size_t alignment, const Type **tys, uint32_t count);
    const Type *(*make_array)(const Type *ty, uint32_t count);
    const Type *(*make_vector)(const Type *ty, uint32_t count);
    const Type *(*make_matrix)(uint32_t dim);
    const Type *(*make_custom)(Slice<const char> name);
    const Type *(*from_desc)(Slice<const char> desc);
    const Type *(*type_bool)();
    const Type *(*type_int16)();
    const Type *(*type_int32)();
    const Type *(*type_int64)();
    const Type *(*type_uint16)();
    const Type *(*type_uint32)();
    const Type *(*type_uint64)();
    const Type *(*type_float16)();
    const Type *(*type_float32)();
    const Node *(*node_prev)(const Node *node);
    const Node *(*node_next)(const Node *node);
    const CInstruction *(*node_inst)(const Node *node);
    const Type *(*node_type)(const Node *node);
    int32_t (*node_get_index)(const Node *node);
    const Node *(*basic_block_first)(const BasicBlock *block);
    const Node *(*basic_block_last)(const BasicBlock *block);
    void (*node_unlink)(Node *node);
    void (*node_set_next)(Node *node, Node *next);
    void (*node_set_prev)(Node *node, Node *prev);
    void (*node_replace)(Node *node, Node *new_node);
    Pool *(*pool_new)();
    void (*pool_drop)(Pool *pool);
    Pool *(*pool_clone)(Pool *pool);
    IrBuilder *(*ir_builder_new)(Pool *pool);
    IrBuilder *(*ir_builder_new_without_bb)(Pool *pool);
    void (*ir_builder_drop)(IrBuilder *builder);
    void (*ir_builder_set_insert_point)(IrBuilder *builder, Node *node);
    Node *(*ir_builder_insert_point)(IrBuilder *builder);
    Node *(*ir_build_call)(IrBuilder *builder, CFunc &&func, Slice<const Node *const> args, const Type *ty);
    Node *(*ir_build_call_tag)(IrBuilder *builder, RustyFuncTag tag, Slice<const Node *const> args, const Type *ty);
    Node *(*ir_build_if)(IrBuilder *builder, const Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch);
    Node *(*ir_build_generic_loop)(IrBuilder *builder, const BasicBlock *prepare, const Node *cond, const BasicBlock *body, const BasicBlock *update);
    Node *(*ir_build_switch)(IrBuilder *builder, const Node *value, Slice<const SwitchCase> cases, const BasicBlock *default_);
    Node *(*ir_build_local)(IrBuilder *builder, const Node *init);
    Node *(*ir_build_break)(IrBuilder *builder);
    Node *(*ir_build_continue)(IrBuilder *builder);
    Node *(*ir_build_return)(IrBuilder *builder, const Node *value);
    const BasicBlock *(*ir_builder_finish)(IrBuilder &&builder);
    const CpuExternFnData *(*cpu_ext_fn_data)(const CpuExternFn *f);
    const CpuExternFn *(*cpu_ext_fn_new)(CpuExternFnData);
    const CpuExternFn *(*cpu_ext_fn_clone)(const CpuExternFn *f);
    void (*cpu_ext_fn_drop)(const CpuExternFn *f);
};
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table();
}// namespace luisa::compute::ir_v2
