#include <luisa/ir_v2/ir_v2_api.h>
#include <luisa/ir_v2/ir_v2.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/core/forget.h>

namespace luisa::compute::ir_v2 {
const Type *ir_v2_binding_type_extract(const Type *ty, uint32_t index) {
    if (ty->is_vector() || ty->is_array()) {
        return ty->element();
    }
    if (ty->is_structure()) {
        auto members = ty->members();
        LUISA_ASSERT(index < members.size(), "Index out of range. {} >= {}", index, members.size());
    }
    LUISA_ASSERT(false, "Cannot extract element from non-vector, non-array, non-structure type.");
    return nullptr;
}
size_t ir_v2_binding_type_size(const Type *ty) {
    return ty->size();
}
size_t ir_v2_binding_type_alignment(const Type *ty) {
    return ty->alignment();
}
RustyTypeTag ir_v2_binding_type_tag(const Type *ty) {
    LUISA_ASSERT(!(ty->tag() >= Type::Tag::BUFFER && ty->tag() <= Type::Tag::ACCEL), "Resource types are not valid IR types.");
    return static_cast<RustyTypeTag>(ty->tag());
}
bool ir_v2_binding_type_is_scalar(const Type *ty) {
    return ty && ty->is_scalar();
}
bool ir_v2_binding_type_is_bool(const Type *ty) {
    return ty && ty->is_bool();
}
bool ir_v2_binding_type_is_int16(const Type *ty) {
    return ty && ty->is_int16();
}
bool ir_v2_binding_type_is_int32(const Type *ty) {
    return ty && ty->is_int32();
}
bool ir_v2_binding_type_is_int64(const Type *ty) {
    return ty && ty->is_int64();
}
bool ir_v2_binding_type_is_uint16(const Type *ty) {
    return ty && ty->is_uint16();
}
bool ir_v2_binding_type_is_uint32(const Type *ty) {
    return ty && ty->is_uint32();
}
bool ir_v2_binding_type_is_uint64(const Type *ty) {
    return ty && ty->is_uint64();
}
bool ir_v2_binding_type_is_float16(const Type *ty) {
    return ty && ty->is_float16();
}
bool ir_v2_binding_type_is_float32(const Type *ty) {
    return ty && ty->is_float32();
}
bool ir_v2_binding_type_is_array(const Type *ty) {
    return ty && ty->is_array();
}
bool ir_v2_binding_type_is_vector(const Type *ty) {
    return ty && ty->is_vector();
}
bool ir_v2_binding_type_is_struct(const Type *ty) {
    return ty && ty->is_structure();
}
bool ir_v2_binding_type_is_custom(const Type *ty) {
    return ty && ty->is_custom();
}
bool ir_v2_binding_type_is_matrix(const Type *ty) {
    return ty && ty->is_matrix();
}
const Type *ir_v2_binding_type_element(const Type *ty) {
    LUISA_ASSERT(ty, "Null pointer.");
    return ty->element();
}
Slice<const char> ir_v2_binding_type_description(const Type *ty) {
    LUISA_ASSERT(ty, "Null pointer.");
    auto desc = ty->description();
    return {desc.data(), desc.size()};
}
size_t ir_v2_binding_type_dimension(const Type *ty) {
    return ty->dimension();
}
Slice<const Type *const> ir_v2_binding_type_members(const Type *ty) {
    auto members = ty->members();
    return {members.data(), members.size()};
}
const Type *ir_v2_binding_make_struct(size_t alignment, const Type **tys, uint32_t count) {
    return Type::structure(alignment, luisa::span{tys, count});
}
const Type *ir_v2_binding_make_array(const Type *ty, uint32_t count) {
    return Type::array(ty, count);
}
const Type *ir_v2_binding_make_vector(const Type *ty, uint32_t count) {
    return Type::vector(ty, count);
}
const Type *ir_v2_binding_make_matrix(uint32_t dim) {
    return Type::matrix(dim);
}
const Type *ir_v2_binding_make_custom(Slice<const char> name) {
    return Type::custom(luisa::string_view{name.data, name.len});
}
const Type *ir_v2_binding_from_desc(Slice<const char> desc) {
    return Type::from(luisa::string_view{desc.data, desc.len});
}
const Type *ir_v2_binding_type_bool() {
    return Type::of<bool>();
}
const Type *ir_v2_binding_type_int16() {
    return Type::of<int16_t>();
}
const Type *ir_v2_binding_type_int32() {
    return Type::of<int32_t>();
}
const Type *ir_v2_binding_type_int64() {
    return Type::of<int64_t>();
}
const Type *ir_v2_binding_type_uint16() {
    return Type::of<uint16_t>();
}
const Type *ir_v2_binding_type_uint32() {
    return Type::of<uint32_t>();
}
const Type *ir_v2_binding_type_uint64() {
    return Type::of<uint64_t>();
}
const Type *ir_v2_binding_type_float16() {
    return Type::of<half>();
}
const Type *ir_v2_binding_type_float32() {
    return Type::of<float>();
}

const Node *ir_v2_binding_node_prev(const Node *node) {
    return node->prev;
}
const Node *ir_v2_binding_node_next(const Node *node) {
    return node->next;
}
const CInstruction *ir_v2_binding_node_inst(const Node *node) {
    return reinterpret_cast<const CInstruction *>(&node->inst);
}
int32_t ir_v2_binding_node_get_index(const Node *node) {
    return *node->get_index();
}
const Type *ir_v2_binding_node_type(const Node *node) {
    return node->ty;
}
const Node *ir_v2_binding_basic_block_first(const BasicBlock *block) {
    return block->first();
}
const Node *ir_v2_binding_basic_block_last(const BasicBlock *block) {
    return block->last();
}
void ir_v2_binding_node_unlink(Node *node) {
    node->unlink();
}
void ir_v2_binding_node_set_next(Node *node, Node *next) {
    node->next = next;
}
void ir_v2_binding_node_set_prev(Node *node, Node *prev) {
    node->prev = prev;
}
void ir_v2_binding_node_replace(Node *node, Node *new_node) {
    node->replace_with(new_node);
}

Pool *ir_v2_binding_pool_new() {
    auto pool = luisa::make_shared<Pool>();
    auto ptr = pool.get();
    forget(std::move(pool));
    return ptr;
}
void ir_v2_binding_pool_drop(Pool *pool) {
    auto _sp = pool->shared_from_this();
}
Pool *ir_v2_binding_pool_clone(Pool *pool) {
    auto sp = pool->shared_from_this();
    auto ptr = sp.get();
    forget(std::move(sp));
    return ptr;
}

IrBuilder *ir_v2_binding_ir_builder_new(Pool *pool) {
    return luisa::new_with_allocator<IrBuilder>(pool->shared_from_this());
}
IrBuilder *ir_v2_binding_ir_builder_new_without_bb(Pool *pool) {
    return luisa::new_with_allocator<IrBuilder>(std::move(IrBuilder::create_without_bb(pool->shared_from_this())));
}
void ir_v2_binding_ir_builder_drop(IrBuilder *builder) {
    luisa::delete_with_allocator(builder);
}
void ir_v2_binding_ir_builder_set_insert_point(IrBuilder *builder, Node *node) {
    builder->set_insert_point(node);
}
Node *ir_v2_binding_ir_builder_insert_point(IrBuilder *builder) {
    return builder->insert_point();
}
Node *ir_v2_binding_ir_build_call(IrBuilder *builder, CFunc &&func, Slice<const Node *const> args, const Type *ty) {
    Func f{};
    std::memcpy(&f, &func, sizeof(CFunc));
    return builder->call(std::move(f), luisa::span{args.data, args.len}, ty);
}
Node *ir_v2_binding_ir_build_call_tag(IrBuilder *builder, RustyFuncTag tag, Slice<const Node *const> args, const Type *ty) {
    return builder->call(static_cast<FuncTag>(tag), luisa::span{args.data, args.len}, ty);
}
Node *ir_v2_binding_ir_build_if(IrBuilder *builder, const Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch) {
    return builder->if_(cond, true_branch, false_branch);
}
Node *ir_v2_binding_ir_build_generic_loop(IrBuilder *builder, const BasicBlock *prepare, const Node *cond, const BasicBlock *body, const BasicBlock *update) {
    return builder->generic_loop(prepare, cond, body, update);
}
Node *ir_v2_binding_ir_build_switch(IrBuilder *builder, const Node *value, Slice<const SwitchCase> cases, const BasicBlock *default_) {
    return builder->switch_(value, luisa::span{cases.data, cases.len}, default_);
}
Node *ir_v2_binding_ir_build_local(IrBuilder *builder, const Node *init) {
    return builder->local(init);
}
Node *ir_v2_binding_ir_build_break(IrBuilder *builder) {
    return builder->break_();
}
Node *ir_v2_binding_ir_build_continue(IrBuilder *builder) {
    return builder->continue_();
}
Node *ir_v2_binding_ir_build_return(IrBuilder *builder, const Node *value) {
    return builder->return_(value);
}
const BasicBlock *ir_v2_binding_ir_builder_finish(IrBuilder &&builder) {
    return std::move(builder).finish();
}
const CpuExternFnData *ir_v2_binding_cpu_ext_fn_data(const CpuExternFn *f) {
    return f;
}
const CpuExternFn *ir_v2_binding_cpu_ext_fn_new(CpuExternFnData data) {
    auto sp = luisa::make_shared<CpuExternFn>(data);
    auto ptr = sp.get();
    forget(std::move(sp));
    return ptr;
}
const CpuExternFn *ir_v2_binding_cpu_ext_fn_clone(const CpuExternFn *f) {
    auto sp = f->shared_from_this();
    auto ptr = sp.get();
    forget(std::move(sp));
    return ptr;
}
void ir_v2_binding_cpu_ext_fn_drop(const CpuExternFn *f) {
    auto _sp = f->shared_from_this();
}
}// namespace luisa::compute::ir_v2