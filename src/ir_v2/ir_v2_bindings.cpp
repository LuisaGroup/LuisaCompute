#include <luisa/ir_v2/ir_v2_api.h>
#include <luisa/ir_v2/ir_v2.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
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
bool ir_v2_binding_type_is_scalar(const Type *ty) {
    return ty->is_scalar();
}
bool ir_v2_binding_type_is_bool(const Type *ty) {
    return ty->is_bool();
}
bool ir_v2_binding_type_is_int16(const Type *ty) {
    return ty->is_int16();
}
bool ir_v2_binding_type_is_int32(const Type *ty) {
    return ty->is_int32();
}
bool ir_v2_binding_type_is_int64(const Type *ty) {
    return ty->is_int64();
}
bool ir_v2_binding_type_is_uint16(const Type *ty) {
    return ty->is_uint16();
}
bool ir_v2_binding_type_is_uint32(const Type *ty) {
    return ty->is_uint32();
}
bool ir_v2_binding_type_is_uint64(const Type *ty) {
    return ty->is_uint64();
}
bool ir_v2_binding_type_is_float16(const Type *ty) {
    return ty->is_float16();
}
bool ir_v2_binding_type_is_float32(const Type *ty) {
    return ty->is_float32();
}
bool ir_v2_binding_type_is_array(const Type *ty) {
    return ty->is_array();
}
bool ir_v2_binding_type_is_vector(const Type *ty) {
    return ty->is_vector();
}
bool ir_v2_binding_type_is_struct(const Type *ty) {
    return ty->is_structure();
}
bool ir_v2_binding_type_is_custom(const Type *ty) {
    return ty->is_custom();
}
bool ir_v2_binding_type_is_matrix(const Type *ty) {
    return ty->is_matrix();
}
const Type *ir_v2_binding_type_element(const Type *ty) {
    return ty->element();
}
Slice<const char> ir_v2_binding_type_description(const Type *ty) {
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
    return reinterpret_cast<const CInstruction*>(&node->inst);
}
const Node *ir_v2_binding_basic_block_first(const BasicBlock *block) {
    return block->first();
}
const Node *ir_v2_binding_basic_block_last(const BasicBlock *block) {
    return block->last();
}
}// namespace luisa::compute::ir_v2