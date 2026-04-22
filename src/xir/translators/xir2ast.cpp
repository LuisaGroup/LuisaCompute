#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <luisa/ast/constant_data.h>
#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/coro/id.h>
#include <luisa/xir/instructions/coro/register.h>
#include <luisa/xir/instructions/coro/suspend.h>
#include <luisa/xir/instructions/coro/token.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/translators/xir2ast.h>

namespace luisa::compute::xir {

namespace {

using ASTFunctionBuilder = luisa::compute::detail::FunctionBuilder;

template<typename To, typename From>
[[nodiscard]] auto xir_cast(From *value) noexcept {
    using ptr_t = std::conditional_t<std::is_const_v<From>, const To *, To *>;
    if (value != nullptr && value->template isa<To>()) {
        return static_cast<ptr_t>(value);
    }
    return static_cast<ptr_t>(nullptr);
}

[[nodiscard]] bool is_exit_target(const BasicBlock *block,
                                  luisa::span<const BasicBlock *const> exit_targets) noexcept {
    return std::find(exit_targets.begin(), exit_targets.end(), block) != exit_targets.end();
}

[[nodiscard]] luisa::string_view comment_of(const Instruction *inst) noexcept {
    if (auto comment = inst->find_metadata<CommentMD>()) {
        return comment->comment();
    }
    return {};
}

class XIR2ASTImpl {
public:
    struct Context {
        const Function *function{nullptr};
        Context *parent{nullptr};
        luisa::unordered_map<const Value *, const Expression *> value_to_exprs;
        luisa::unordered_map<const AllocaInst *, const Argument *> argument_local_copies;
    };

private:
    Context *_ctx{nullptr};
    luisa::unordered_map<const CallableFunction *,
                         luisa::shared_ptr<const ASTFunctionBuilder>>
        _converted_callables;

private:
    [[nodiscard]] static ASTFunctionBuilder *fb() noexcept {
        return ASTFunctionBuilder::current();
    }

    void _emit_comments(const Instruction *inst) noexcept;
    [[nodiscard]] static CurveBasisSet _curve_basis_set(const Instruction *inst) noexcept;
    [[nodiscard]] static compute::CallOp _decide_make_vector_op(const Type *primitive, size_t length) noexcept;
    [[nodiscard]] static compute::CallOp _decide_make_matrix_op(size_t dimension) noexcept;
    [[nodiscard]] static size_t _constant_index(const Value *value) noexcept;

    [[nodiscard]] const Expression *_convert_constant(const Constant *constant) noexcept;
    [[nodiscard]] const Expression *_convert_special_register(const SpecialRegister *sreg) noexcept;
    [[nodiscard]] const Expression *_convert_argument(const Argument *arg) noexcept;
    void _convert_alloca(const AllocaInst *alloca) noexcept;
    [[nodiscard]] const Expression *_convert_lvalue(const Value *value) noexcept;
    [[nodiscard]] luisa::vector<const Expression *> _convert_operands(const User *inst, size_t offset = 0u) noexcept;
    [[nodiscard]] const Expression *_convert_aggregate(const ArithmeticInst *inst) noexcept;
    [[nodiscard]] const Expression *_convert_extract(const ArithmeticInst *inst) noexcept;
    [[nodiscard]] const Expression *_convert_shuffle(const ArithmeticInst *inst) noexcept;
    [[nodiscard]] const Expression *_convert_arithmetic(const ArithmeticInst *inst) noexcept;
    [[nodiscard]] static compute::CallOp _atomic_call_op(AtomicOp op) noexcept;
    [[nodiscard]] static compute::CallOp _resource_read_call_op(ResourceReadOp op) noexcept;
    [[nodiscard]] static compute::CallOp _resource_write_call_op(ResourceWriteOp op) noexcept;
    [[nodiscard]] static compute::CallOp _resource_query_call_op(ResourceQueryOp op) noexcept;
    [[nodiscard]] static compute::CallOp _thread_group_call_op(ThreadGroupOp op) noexcept;
    [[nodiscard]] static const Function *_value_parent_function(const Value *value) noexcept;
    [[nodiscard]] Context *_find_context(const Function *function) const noexcept;
    [[nodiscard]] luisa::vector<const Expression *> _convert_call_arguments(const CallInst *inst,
                                                                            const CallableFunction *callee) noexcept;
    [[nodiscard]] const Expression *_convert_call(const CallInst *inst) noexcept;
    [[nodiscard]] const Expression *_convert_value(const Value *value) noexcept;
    [[nodiscard]] const Expression *_convert_loop_condition(const BasicBlock *prepare,
                                                           const Value *condition) noexcept;
    [[nodiscard]] bool _is_argument_copy_store(const StoreInst *store) const noexcept;
    void _convert_instruction(const Instruction *inst) noexcept;
    [[nodiscard]] std::pair<const Expression *, const Expression *> _extract_for_update(const LoopInst *loop) noexcept;
    void _convert_block(const BasicBlock *block, luisa::span<const BasicBlock *const> exit_targets) noexcept;
    void _analyze_argument_local_copies(const FunctionDefinition *function) noexcept;
    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> _convert_kernel(const KernelFunction *kernel) noexcept;
    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> _convert_callable(const CallableFunction *callable) noexcept;

public:
    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> build(const KernelFunction *kernel) noexcept;
    [[nodiscard]] luisa::shared_ptr<const ASTFunctionBuilder> build(const CallableFunction *callable) noexcept;
};

void XIR2ASTImpl::_emit_comments(const Instruction *inst) noexcept {
    for (auto md : inst->metadata_list()) {
        if (auto comment = xir_cast<CommentMD>(md)) {
            fb()->comment_(luisa::string{comment->comment()});
        }
    }
}

CurveBasisSet XIR2ASTImpl::_curve_basis_set(const Instruction *inst) noexcept {
    if (auto md = inst->find_metadata<CurveBasisMD>()) {
        return md->curve_basis_set();
    }
    return {};
}

compute::CallOp XIR2ASTImpl::_decide_make_vector_op(const Type *primitive, size_t length) noexcept {
    LUISA_ASSERT(primitive->is_scalar(), "Expected scalar type, got {}.", primitive->description());
    switch (primitive->tag()) {
        case Type::Tag::BOOL:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_BOOL2;
                case 3u: return compute::CallOp::MAKE_BOOL3;
                case 4u: return compute::CallOp::MAKE_BOOL4;
                default: break;
            }
            break;
        case Type::Tag::INT8:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_BYTE2;
                case 3u: return compute::CallOp::MAKE_BYTE3;
                case 4u: return compute::CallOp::MAKE_BYTE4;
                default: break;
            }
            break;
        case Type::Tag::UINT8:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_UBYTE2;
                case 3u: return compute::CallOp::MAKE_UBYTE3;
                case 4u: return compute::CallOp::MAKE_UBYTE4;
                default: break;
            }
            break;
        case Type::Tag::INT16:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_SHORT2;
                case 3u: return compute::CallOp::MAKE_SHORT3;
                case 4u: return compute::CallOp::MAKE_SHORT4;
                default: break;
            }
            break;
        case Type::Tag::UINT16:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_USHORT2;
                case 3u: return compute::CallOp::MAKE_USHORT3;
                case 4u: return compute::CallOp::MAKE_USHORT4;
                default: break;
            }
            break;
        case Type::Tag::INT32:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_INT2;
                case 3u: return compute::CallOp::MAKE_INT3;
                case 4u: return compute::CallOp::MAKE_INT4;
                default: break;
            }
            break;
        case Type::Tag::UINT32:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_UINT2;
                case 3u: return compute::CallOp::MAKE_UINT3;
                case 4u: return compute::CallOp::MAKE_UINT4;
                default: break;
            }
            break;
        case Type::Tag::INT64:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_LONG2;
                case 3u: return compute::CallOp::MAKE_LONG3;
                case 4u: return compute::CallOp::MAKE_LONG4;
                default: break;
            }
            break;
        case Type::Tag::UINT64:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_ULONG2;
                case 3u: return compute::CallOp::MAKE_ULONG3;
                case 4u: return compute::CallOp::MAKE_ULONG4;
                default: break;
            }
            break;
        case Type::Tag::FLOAT16:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_HALF2;
                case 3u: return compute::CallOp::MAKE_HALF3;
                case 4u: return compute::CallOp::MAKE_HALF4;
                default: break;
            }
            break;
        case Type::Tag::FLOAT32:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_FLOAT2;
                case 3u: return compute::CallOp::MAKE_FLOAT3;
                case 4u: return compute::CallOp::MAKE_FLOAT4;
                default: break;
            }
            break;
        case Type::Tag::FLOAT64:
            switch (length) {
                case 2u: return compute::CallOp::MAKE_DOUBLE2;
                case 3u: return compute::CallOp::MAKE_DOUBLE3;
                case 4u: return compute::CallOp::MAKE_DOUBLE4;
                default: break;
            }
            break;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported vector type '{}'.", primitive->description());
}

compute::CallOp XIR2ASTImpl::_decide_make_matrix_op(size_t dimension) noexcept {
    switch (dimension) {
        case 2u: return compute::CallOp::MAKE_FLOAT2X2;
        case 3u: return compute::CallOp::MAKE_FLOAT3X3;
        case 4u: return compute::CallOp::MAKE_FLOAT4X4;
        default: LUISA_ERROR_WITH_LOCATION("Unsupported matrix dimension {}.", dimension);
    }
}

size_t XIR2ASTImpl::_constant_index(const Value *value) noexcept {
    auto constant = xir_cast<Constant>(value);
    LUISA_ASSERT(constant != nullptr, "Expected constant index, got {}.", xir::to_string(value->derived_value_tag()));
    switch (constant->type()->tag()) {
        case Type::Tag::INT8: return static_cast<size_t>(constant->as<byte>());
        case Type::Tag::UINT8: return static_cast<size_t>(constant->as<ubyte>());
        case Type::Tag::INT16: return static_cast<size_t>(constant->as<short>());
        case Type::Tag::UINT16: return static_cast<size_t>(constant->as<ushort>());
        case Type::Tag::INT32: return static_cast<size_t>(constant->as<int>());
        case Type::Tag::UINT32: return static_cast<size_t>(constant->as<uint>());
        case Type::Tag::INT64: return static_cast<size_t>(constant->as<slong>());
        case Type::Tag::UINT64: return static_cast<size_t>(constant->as<ulong>());
        default: LUISA_ERROR_WITH_LOCATION("Invalid constant index type '{}'.", constant->type()->description());
    }
}

const Expression *XIR2ASTImpl::_convert_constant(const Constant *constant) noexcept {
    if (auto iter = _ctx->value_to_exprs.find(constant);
        iter != _ctx->value_to_exprs.end()) {
        return iter->second;
    }
    auto type = constant->type();
    auto data = constant->data();
    const Expression *expr = nullptr;
    if (type->is_scalar() || type->is_vector() || type->is_matrix()) {
#define LUISA_XIR2AST_DECODE_CONST(T) \
    if (type == Type::of<T>()) { expr = fb()->literal(type, *reinterpret_cast<const T *>(data)); }

#define LUISA_XIR2AST_DECODE_CONST_VEC(T) \
    LUISA_XIR2AST_DECODE_CONST(T)         \
    LUISA_XIR2AST_DECODE_CONST(T##2)      \
    LUISA_XIR2AST_DECODE_CONST(T##3)      \
    LUISA_XIR2AST_DECODE_CONST(T##4)

        LUISA_XIR2AST_DECODE_CONST(bool)
        LUISA_XIR2AST_DECODE_CONST_VEC(int)
        LUISA_XIR2AST_DECODE_CONST_VEC(uint)
        LUISA_XIR2AST_DECODE_CONST_VEC(short)
        LUISA_XIR2AST_DECODE_CONST_VEC(ushort)
        LUISA_XIR2AST_DECODE_CONST_VEC(byte)
        LUISA_XIR2AST_DECODE_CONST_VEC(ubyte)
        LUISA_XIR2AST_DECODE_CONST_VEC(slong)
        LUISA_XIR2AST_DECODE_CONST_VEC(ulong)
        LUISA_XIR2AST_DECODE_CONST_VEC(half)
        LUISA_XIR2AST_DECODE_CONST_VEC(float)
        LUISA_XIR2AST_DECODE_CONST_VEC(double)
        LUISA_XIR2AST_DECODE_CONST(float2x2)
        LUISA_XIR2AST_DECODE_CONST(float3x3)
        LUISA_XIR2AST_DECODE_CONST(float4x4)

#undef LUISA_XIR2AST_DECODE_CONST_VEC
#undef LUISA_XIR2AST_DECODE_CONST
    }
    if (expr == nullptr) {
        auto c = ConstantData::create(type, data, type->size());
        expr = fb()->constant(c);
    }
    _ctx->value_to_exprs.emplace(constant, expr);
    return expr;
}

const Expression *XIR2ASTImpl::_convert_special_register(const SpecialRegister *sreg) noexcept {
    if (auto iter = _ctx->value_to_exprs.find(sreg);
        iter != _ctx->value_to_exprs.end()) {
        return iter->second;
    }
    auto expr = [&]() noexcept -> const Expression * {
        switch (sreg->derived_special_register_tag()) {
            case DerivedSpecialRegisterTag::THREAD_ID: return fb()->thread_id();
            case DerivedSpecialRegisterTag::BLOCK_ID: return fb()->block_id();
            case DerivedSpecialRegisterTag::WARP_LANE_ID: return fb()->warp_lane_id();
            case DerivedSpecialRegisterTag::DISPATCH_ID: return fb()->dispatch_id();
            case DerivedSpecialRegisterTag::KERNEL_ID: return fb()->kernel_id();
            case DerivedSpecialRegisterTag::RASTER_OBJECT_ID: return fb()->raster_object_id();
            case DerivedSpecialRegisterTag::RASTER_BARYCENTRICS: return fb()->raster_barycentrics();
            case DerivedSpecialRegisterTag::WARP_SIZE: return fb()->warp_lane_count();
            case DerivedSpecialRegisterTag::DISPATCH_SIZE: return fb()->dispatch_size();
            case DerivedSpecialRegisterTag::BLOCK_SIZE:
                LUISA_ERROR_WITH_LOCATION("AST has no direct block-size expression.");
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported special register.");
    }();
    _ctx->value_to_exprs.emplace(sreg, expr);
    return expr;
}

const Expression *XIR2ASTImpl::_convert_argument(const Argument *arg) noexcept {
    if (auto iter = _ctx->value_to_exprs.find(arg);
        iter != _ctx->value_to_exprs.end()) {
        return iter->second;
    }
    auto expr = [&]() noexcept -> const Expression * {
        switch (arg->derived_argument_tag()) {
            case DerivedArgumentTag::VALUE: return fb()->argument(arg->type());
            case DerivedArgumentTag::REFERENCE: return fb()->reference(arg->type());
            case DerivedArgumentTag::RESOURCE: {
                if (arg->type()->is_buffer()) { return fb()->buffer(arg->type()); }
                if (arg->type()->is_texture()) { return fb()->texture(arg->type()); }
                if (arg->type()->is_bindless_array()) { return fb()->bindless_array(); }
                if (arg->type()->is_accel()) { return fb()->accel(); }
                LUISA_ERROR_WITH_LOCATION("Unsupported resource argument type '{}'.",
                                          arg->type()->description());
            }
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported argument.");
    }();
    _ctx->value_to_exprs.emplace(arg, expr);
    return expr;
}

void XIR2ASTImpl::_convert_alloca(const AllocaInst *alloca) noexcept {
    if (_ctx->value_to_exprs.contains(alloca)) { return; }
    if (auto iter = _ctx->argument_local_copies.find(alloca);
        iter != _ctx->argument_local_copies.end()) {
        _ctx->value_to_exprs.emplace(alloca, _convert_argument(iter->second));
        return;
    }
    auto expr = alloca->is_shared() ? fb()->shared(alloca->type()) :
                                      fb()->local(alloca->type());
    _ctx->value_to_exprs.emplace(alloca, expr);
}

const Expression *XIR2ASTImpl::_convert_lvalue(const Value *value) noexcept {
    if (auto owner = _value_parent_function(value);
        owner != nullptr && _ctx != nullptr && _ctx->function != owner) {
        auto owner_ctx = _find_context(owner);
        LUISA_ASSERT(owner_ctx != nullptr,
                     "Missing outer context for XIR lvalue from another function.");
        auto old_ctx = std::exchange(_ctx, owner_ctx);
        auto expr = _convert_lvalue(value);
        _ctx = old_ctx;
        return expr;
    }
    if (auto iter = _ctx->value_to_exprs.find(value);
        iter != _ctx->value_to_exprs.end()) {
        return iter->second;
    }
    if (auto arg = xir_cast<Argument>(value)) { return _convert_argument(arg); }
    if (auto alloca = xir_cast<AllocaInst>(value)) {
        _convert_alloca(alloca);
        return _ctx->value_to_exprs.at(alloca);
    }
    if (auto gep = xir_cast<GEPInst>(value)) {
        auto self = _convert_lvalue(gep->base());
        for (auto i = 0u; i < gep->index_count(); i++) {
            auto self_type = self->type();
            if (self_type->is_structure()) {
                auto member_index = _constant_index(gep->index(i));
                self = fb()->member(self_type->members()[member_index], self, member_index);
            } else if (self_type->is_array() || self_type->is_vector()) {
                self = fb()->access(self_type->element(), self, _convert_value(gep->index(i)));
            } else {
                LUISA_ASSERT(self_type->is_matrix(), "Invalid GEP base type '{}'.",
                             self_type->description());
                auto inner_type = Type::vector(self_type->element(), self_type->dimension());
                self = fb()->access(inner_type, self, _convert_value(gep->index(i)));
            }
        }
        LUISA_ASSERT(self->type() == gep->type(),
                     "GEP type mismatch: expected '{}', got '{}'.",
                     gep->type()->description(), self->type()->description());
        _ctx->value_to_exprs.emplace(gep, self);
        return self;
    }
    LUISA_ERROR_WITH_LOCATION("Value '{}' is not an lvalue.", xir::to_string(value->derived_value_tag()));
}

luisa::vector<const Expression *> XIR2ASTImpl::_convert_operands(const User *inst, size_t offset) noexcept {
    auto args = luisa::vector<const Expression *>{};
    args.reserve(inst->operand_count() - offset);
    for (auto i = offset; i < inst->operand_count(); i++) {
        args.emplace_back(_convert_value(inst->operand(i)));
    }
    return args;
}

const Expression *XIR2ASTImpl::_convert_aggregate(const ArithmeticInst *inst) noexcept {
    auto type = inst->type();
    if (type->is_vector()) {
        if (inst->operand_count() == type->dimension() && inst->operand_count() > 0u) {
            auto first = inst->operand(0u);
            auto is_broadcast = first->type()->is_scalar();
            for (auto i = 1u; is_broadcast && i < inst->operand_count(); i++) {
                is_broadcast = inst->operand(i) == first;
            }
            if (is_broadcast) {
                auto arg = std::array{_convert_value(first)};
                return fb()->call(type, _decide_make_vector_op(type->element(), type->dimension()),
                                  luisa::span{arg});
            }
        }
        auto args = _convert_operands(inst);
        return fb()->call(type, _decide_make_vector_op(type->element(), type->dimension()),
                          luisa::span{args});
    }
    if (type->is_matrix()) {
        auto args = _convert_operands(inst);
        return fb()->call(type, _decide_make_matrix_op(type->dimension()),
                          luisa::span{args});
    }
    if (type->is_structure()) {
        auto local = fb()->local(type);
        auto members = type->members();
        LUISA_ASSERT(members.size() == inst->operand_count(),
                     "Structure aggregate field count mismatch.");
        for (auto i = 0u; i < members.size(); i++) {
            fb()->assign(fb()->member(members[i], local, i), _convert_value(inst->operand(i)));
        }
        return local;
    }
    if (type->is_array()) {
        auto local = fb()->local(type);
        LUISA_ASSERT(type->dimension() == inst->operand_count(),
                     "Array aggregate element count mismatch.");
        for (auto i = 0u; i < inst->operand_count(); i++) {
            auto index = fb()->literal(Type::of<uint>(), static_cast<uint>(i));
            fb()->assign(fb()->access(type->element(), local, index), _convert_value(inst->operand(i)));
        }
        return local;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported aggregate type '{}'.", type->description());
}

const Expression *XIR2ASTImpl::_convert_extract(const ArithmeticInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() >= 2u, "extract requires at least 2 operands.");
    auto self = _convert_value(inst->operand(0u));
    for (auto i = 1u; i < inst->operand_count(); i++) {
        auto self_type = self->type();
        if (self_type->is_structure()) {
            auto member_index = _constant_index(inst->operand(i));
            self = fb()->member(self_type->members()[member_index], self, member_index);
        } else if (self_type->is_vector()) {
            auto swizzle_index = _constant_index(inst->operand(i));
            self = fb()->swizzle(self_type->element(), self, 1u, static_cast<uint64_t>(swizzle_index));
        } else if (self_type->is_array()) {
            self = fb()->access(self_type->element(), self, _convert_value(inst->operand(i)));
        } else {
            LUISA_ASSERT(self_type->is_matrix(), "Invalid extract base type '{}'.",
                         self_type->description());
            auto inner_type = Type::vector(self_type->element(), self_type->dimension());
            self = fb()->access(inner_type, self, _convert_value(inst->operand(i)));
        }
    }
    LUISA_ASSERT(self->type() == inst->type(),
                 "Extract type mismatch: expected '{}', got '{}'.",
                 inst->type()->description(), self->type()->description());
    return self;
}

const Expression *XIR2ASTImpl::_convert_shuffle(const ArithmeticInst *inst) noexcept {
    LUISA_ASSERT(inst->operand_count() >= 2u, "shuffle requires at least 2 operands.");
    auto self = _convert_value(inst->operand(0u));
    LUISA_ASSERT(self->type()->is_vector(), "shuffle currently only supports vectors.");
    auto swizzle_size = inst->operand_count() - 1u;
    uint64_t swizzle_code = 0u;
    for (auto i = 0u; i < swizzle_size; i++) {
        auto index = _constant_index(inst->operand(i + 1u));
        swizzle_code |= (static_cast<uint64_t>(index) & 0xfu) << (i * 4u);
    }
    return fb()->swizzle(inst->type(), self, swizzle_size, swizzle_code);
}

const Expression *XIR2ASTImpl::_convert_arithmetic(const ArithmeticInst *inst) noexcept {
    auto type = inst->type();
    auto unary = [&](compute::UnaryOp op) noexcept -> const Expression * {
        LUISA_ASSERT(inst->operand_count() == 1u, "Unary op '{}' expects 1 operand.", xir::to_string(inst->op()));
        return fb()->unary(type, op, _convert_value(inst->operand(0u)));
    };
    auto binary = [&](compute::BinaryOp op) noexcept -> const Expression * {
        LUISA_ASSERT(inst->operand_count() == 2u, "Binary op '{}' expects 2 operands.", xir::to_string(inst->op()));
        return fb()->binary(type, op, _convert_value(inst->operand(0u)), _convert_value(inst->operand(1u)));
    };
    auto call = [&](compute::CallOp op) noexcept -> const Expression * {
        auto args = _convert_operands(inst);
        return fb()->call(type, op, luisa::span{args}, _curve_basis_set(inst));
    };
    switch (inst->op()) {
        case ArithmeticOp::UNARY_MINUS: return unary(compute::UnaryOp::MINUS);
        case ArithmeticOp::UNARY_BIT_NOT:
            if (type->is_bool() || type->is_bool_vector()) { return unary(compute::UnaryOp::NOT); }
            return unary(compute::UnaryOp::BIT_NOT);
        case ArithmeticOp::BINARY_ADD: [[fallthrough]];
        case ArithmeticOp::MATRIX_COMP_ADD: return binary(compute::BinaryOp::ADD);
        case ArithmeticOp::BINARY_SUB: [[fallthrough]];
        case ArithmeticOp::MATRIX_COMP_SUB: return binary(compute::BinaryOp::SUB);
        case ArithmeticOp::BINARY_MUL: [[fallthrough]];
        case ArithmeticOp::MATRIX_COMP_MUL: [[fallthrough]];
        case ArithmeticOp::MATRIX_LINALG_MUL: return binary(compute::BinaryOp::MUL);
        case ArithmeticOp::BINARY_DIV: [[fallthrough]];
        case ArithmeticOp::MATRIX_COMP_DIV: return binary(compute::BinaryOp::DIV);
        case ArithmeticOp::BINARY_MOD: return binary(compute::BinaryOp::MOD);
        case ArithmeticOp::BINARY_BIT_AND:
            if (type->is_bool() || type->is_bool_vector()) { return binary(compute::BinaryOp::AND); }
            return binary(compute::BinaryOp::BIT_AND);
        case ArithmeticOp::BINARY_BIT_OR:
            if (type->is_bool() || type->is_bool_vector()) { return binary(compute::BinaryOp::OR); }
            return binary(compute::BinaryOp::BIT_OR);
        case ArithmeticOp::BINARY_BIT_XOR: return binary(compute::BinaryOp::BIT_XOR);
        case ArithmeticOp::BINARY_SHIFT_LEFT: return binary(compute::BinaryOp::SHL);
        case ArithmeticOp::BINARY_SHIFT_RIGHT: return binary(compute::BinaryOp::SHR);
        case ArithmeticOp::BINARY_LESS: return binary(compute::BinaryOp::LESS);
        case ArithmeticOp::BINARY_GREATER: return binary(compute::BinaryOp::GREATER);
        case ArithmeticOp::BINARY_LESS_EQUAL: return binary(compute::BinaryOp::LESS_EQUAL);
        case ArithmeticOp::BINARY_GREATER_EQUAL: return binary(compute::BinaryOp::GREATER_EQUAL);
        case ArithmeticOp::BINARY_EQUAL: return binary(compute::BinaryOp::EQUAL);
        case ArithmeticOp::BINARY_NOT_EQUAL: return binary(compute::BinaryOp::NOT_EQUAL);
        case ArithmeticOp::ALL: return call(compute::CallOp::ALL);
        case ArithmeticOp::ANY: return call(compute::CallOp::ANY);
        case ArithmeticOp::SELECT: return call(compute::CallOp::SELECT);
        case ArithmeticOp::CLAMP: return call(compute::CallOp::CLAMP);
        case ArithmeticOp::SATURATE: return call(compute::CallOp::SATURATE);
        case ArithmeticOp::LERP: return call(compute::CallOp::LERP);
        case ArithmeticOp::SMOOTHSTEP: return call(compute::CallOp::SMOOTHSTEP);
        case ArithmeticOp::STEP: return call(compute::CallOp::STEP);
        case ArithmeticOp::ABS: return call(compute::CallOp::ABS);
        case ArithmeticOp::MIN: return call(compute::CallOp::MIN);
        case ArithmeticOp::MAX: return call(compute::CallOp::MAX);
        case ArithmeticOp::CLZ: return call(compute::CallOp::CLZ);
        case ArithmeticOp::CTZ: return call(compute::CallOp::CTZ);
        case ArithmeticOp::POPCOUNT: return call(compute::CallOp::POPCOUNT);
        case ArithmeticOp::REVERSE: return call(compute::CallOp::REVERSE);
        case ArithmeticOp::ISINF: return call(compute::CallOp::ISINF);
        case ArithmeticOp::ISNAN: return call(compute::CallOp::ISNAN);
        case ArithmeticOp::ACOS: return call(compute::CallOp::ACOS);
        case ArithmeticOp::ACOSH: return call(compute::CallOp::ACOSH);
        case ArithmeticOp::ASIN: return call(compute::CallOp::ASIN);
        case ArithmeticOp::ASINH: return call(compute::CallOp::ASINH);
        case ArithmeticOp::ATAN: return call(compute::CallOp::ATAN);
        case ArithmeticOp::ATAN2: return call(compute::CallOp::ATAN2);
        case ArithmeticOp::ATANH: return call(compute::CallOp::ATANH);
        case ArithmeticOp::COS: return call(compute::CallOp::COS);
        case ArithmeticOp::COSH: return call(compute::CallOp::COSH);
        case ArithmeticOp::SIN: return call(compute::CallOp::SIN);
        case ArithmeticOp::SINH: return call(compute::CallOp::SINH);
        case ArithmeticOp::TAN: return call(compute::CallOp::TAN);
        case ArithmeticOp::TANH: return call(compute::CallOp::TANH);
        case ArithmeticOp::EXP: return call(compute::CallOp::EXP);
        case ArithmeticOp::EXP2: return call(compute::CallOp::EXP2);
        case ArithmeticOp::EXP10: return call(compute::CallOp::EXP10);
        case ArithmeticOp::LOG: return call(compute::CallOp::LOG);
        case ArithmeticOp::LOG2: return call(compute::CallOp::LOG2);
        case ArithmeticOp::LOG10: return call(compute::CallOp::LOG10);
        case ArithmeticOp::POW: return call(compute::CallOp::POW);
        case ArithmeticOp::SQRT: return call(compute::CallOp::SQRT);
        case ArithmeticOp::RSQRT: return call(compute::CallOp::RSQRT);
        case ArithmeticOp::CEIL: return call(compute::CallOp::CEIL);
        case ArithmeticOp::FLOOR: return call(compute::CallOp::FLOOR);
        case ArithmeticOp::FRACT: return call(compute::CallOp::FRACT);
        case ArithmeticOp::TRUNC: return call(compute::CallOp::TRUNC);
        case ArithmeticOp::ROUND: return call(compute::CallOp::ROUND);
        case ArithmeticOp::FMA: return call(compute::CallOp::FMA);
        case ArithmeticOp::COPYSIGN: return call(compute::CallOp::COPYSIGN);
        case ArithmeticOp::CROSS: return call(compute::CallOp::CROSS);
        case ArithmeticOp::DOT: return call(compute::CallOp::DOT);
        case ArithmeticOp::LENGTH: return call(compute::CallOp::LENGTH);
        case ArithmeticOp::LENGTH_SQUARED: return call(compute::CallOp::LENGTH_SQUARED);
        case ArithmeticOp::NORMALIZE: return call(compute::CallOp::NORMALIZE);
        case ArithmeticOp::FACEFORWARD: return call(compute::CallOp::FACEFORWARD);
        case ArithmeticOp::REFLECT: return call(compute::CallOp::REFLECT);
        case ArithmeticOp::REDUCE_SUM: return call(compute::CallOp::REDUCE_SUM);
        case ArithmeticOp::REDUCE_PRODUCT: return call(compute::CallOp::REDUCE_PRODUCT);
        case ArithmeticOp::REDUCE_MIN: return call(compute::CallOp::REDUCE_MIN);
        case ArithmeticOp::REDUCE_MAX: return call(compute::CallOp::REDUCE_MAX);
        case ArithmeticOp::OUTER_PRODUCT: return call(compute::CallOp::OUTER_PRODUCT);
        case ArithmeticOp::MATRIX_COMP_NEG: return unary(compute::UnaryOp::MINUS);
        case ArithmeticOp::MATRIX_DETERMINANT: return call(compute::CallOp::DETERMINANT);
        case ArithmeticOp::MATRIX_TRANSPOSE: return call(compute::CallOp::TRANSPOSE);
        case ArithmeticOp::MATRIX_INVERSE: return call(compute::CallOp::INVERSE);
        case ArithmeticOp::AGGREGATE: return _convert_aggregate(inst);
        case ArithmeticOp::SHUFFLE: return _convert_shuffle(inst);
        case ArithmeticOp::EXTRACT: return _convert_extract(inst);
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::RINT:
        case ArithmeticOp::INSERT:
            LUISA_NOT_IMPLEMENTED("Unsupported arithmetic op '{}'.", xir::to_string(inst->op()));
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported arithmetic op '{}'.", xir::to_string(inst->op()));
}

compute::CallOp XIR2ASTImpl::_resource_read_call_op(ResourceReadOp op) noexcept {
    switch (op) {
        case ResourceReadOp::BUFFER_READ: return compute::CallOp::BUFFER_READ;
        case ResourceReadOp::BUFFER_VOLATILE_READ: return compute::CallOp::BUFFER_VOLATILE_READ;
        case ResourceReadOp::BYTE_BUFFER_READ: return compute::CallOp::BYTE_BUFFER_READ;
        case ResourceReadOp::BYTE_BUFFER_VOLATILE_READ: return compute::CallOp::BYTE_BUFFER_VOLATILE_READ;
        case ResourceReadOp::TEXTURE2D_READ: [[fallthrough]];
        case ResourceReadOp::TEXTURE3D_READ: return compute::CallOp::TEXTURE_READ;
        case ResourceReadOp::BINDLESS_BUFFER_READ: return compute::CallOp::BINDLESS_BUFFER_READ;
        case ResourceReadOp::BINDLESS_BYTE_BUFFER_READ: return compute::CallOp::BINDLESS_BYTE_BUFFER_READ;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ: return compute::CallOp::BINDLESS_TEXTURE2D_READ;
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ: return compute::CallOp::BINDLESS_TEXTURE3D_READ;
        case ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL: return compute::CallOp::BINDLESS_TEXTURE2D_READ_LEVEL;
        case ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL: return compute::CallOp::BINDLESS_TEXTURE3D_READ_LEVEL;
        case ResourceReadOp::DEVICE_ADDRESS_READ:
            LUISA_NOT_IMPLEMENTED("DEVICE_ADDRESS_READ is not representable in AST.");
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported resource read op '{}'.", xir::to_string(op));
}

compute::CallOp XIR2ASTImpl::_atomic_call_op(AtomicOp op) noexcept {
    switch (op) {
        case AtomicOp::EXCHANGE: return compute::CallOp::ATOMIC_EXCHANGE;
        case AtomicOp::COMPARE_EXCHANGE: return compute::CallOp::ATOMIC_COMPARE_EXCHANGE;
        case AtomicOp::FETCH_ADD: return compute::CallOp::ATOMIC_FETCH_ADD;
        case AtomicOp::FETCH_SUB: return compute::CallOp::ATOMIC_FETCH_SUB;
        case AtomicOp::FETCH_AND: return compute::CallOp::ATOMIC_FETCH_AND;
        case AtomicOp::FETCH_OR: return compute::CallOp::ATOMIC_FETCH_OR;
        case AtomicOp::FETCH_XOR: return compute::CallOp::ATOMIC_FETCH_XOR;
        case AtomicOp::FETCH_MIN: return compute::CallOp::ATOMIC_FETCH_MIN;
        case AtomicOp::FETCH_MAX: return compute::CallOp::ATOMIC_FETCH_MAX;
        default: LUISA_NOT_IMPLEMENTED("Unsupported atomic op '{}'.", xir::to_string(op));
    }
}

compute::CallOp XIR2ASTImpl::_resource_write_call_op(ResourceWriteOp op) noexcept {
    switch (op) {
        case ResourceWriteOp::BUFFER_WRITE: return compute::CallOp::BUFFER_WRITE;
        case ResourceWriteOp::BUFFER_VOLATILE_WRITE: return compute::CallOp::BUFFER_VOLATILE_WRITE;
        case ResourceWriteOp::BYTE_BUFFER_WRITE: return compute::CallOp::BYTE_BUFFER_WRITE;
        case ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE: return compute::CallOp::BYTE_BUFFER_VOLATILE_WRITE;
        case ResourceWriteOp::TEXTURE2D_WRITE: [[fallthrough]];
        case ResourceWriteOp::TEXTURE3D_WRITE: return compute::CallOp::TEXTURE_WRITE;
        case ResourceWriteOp::BINDLESS_BUFFER_WRITE: return compute::CallOp::BINDLESS_BUFFER_WRITE;
        case ResourceWriteOp::BINDLESS_BYTE_BUFFER_WRITE:
        case ResourceWriteOp::DEVICE_ADDRESS_WRITE:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
        case ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL:
        case ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT:
            LUISA_NOT_IMPLEMENTED("Unsupported resource write op '{}'.", xir::to_string(op));
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported resource write op '{}'.", xir::to_string(op));
}

compute::CallOp XIR2ASTImpl::_resource_query_call_op(ResourceQueryOp op) noexcept {
    switch (op) {
        case ResourceQueryOp::BUFFER_SIZE: return compute::CallOp::BUFFER_SIZE;
        case ResourceQueryOp::BYTE_BUFFER_SIZE: return compute::CallOp::BYTE_BUFFER_SIZE;
        case ResourceQueryOp::TEXTURE2D_SIZE: [[fallthrough]];
        case ResourceQueryOp::TEXTURE3D_SIZE: return compute::CallOp::TEXTURE_SIZE;
        case ResourceQueryOp::BINDLESS_BUFFER_SIZE: return compute::CallOp::BINDLESS_BUFFER_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: return compute::CallOp::BINDLESS_TEXTURE2D_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: return compute::CallOp::BINDLESS_TEXTURE3D_SIZE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return compute::CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return compute::CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL;
        case ResourceQueryOp::BUFFER_DEVICE_ADDRESS: return compute::CallOp::BUFFER_ADDRESS;
        case ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: return compute::CallOp::BINDLESS_BUFFER_ADDRESS;
        case ResourceQueryOp::TEXTURE2D_SAMPLE: return compute::CallOp::TEXTURE2D_SAMPLE;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: return compute::CallOp::TEXTURE2D_SAMPLE_LEVEL;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: return compute::CallOp::TEXTURE2D_SAMPLE_GRAD;
        case ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: return compute::CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::TEXTURE3D_SAMPLE: return compute::CallOp::TEXTURE3D_SAMPLE;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: return compute::CallOp::TEXTURE3D_SAMPLE_LEVEL;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: return compute::CallOp::TEXTURE3D_SAMPLE_GRAD;
        case ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: return compute::CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER;
        case ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return compute::CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: return compute::CallOp::RAY_TRACING_INSTANCE_TRANSFORM;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: return compute::CallOp::RAY_TRACING_INSTANCE_USER_ID;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return compute::CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK;
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: return compute::CallOp::RAY_TRACING_TRACE_CLOSEST;
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY: return compute::CallOp::RAY_TRACING_TRACE_ANY;
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL: return compute::CallOp::RAY_TRACING_QUERY_ALL;
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY: return compute::CallOp::RAY_TRACING_QUERY_ANY;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return compute::CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX;
        case ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: return compute::CallOp::RAY_TRACING_INSTANCE_MOTION_SRT;
        case ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return compute::CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return compute::CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return compute::CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR;
        case ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return compute::CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
        default: LUISA_NOT_IMPLEMENTED("Unsupported resource query op '{}'.", xir::to_string(op));
    }
}

compute::CallOp XIR2ASTImpl::_thread_group_call_op(ThreadGroupOp op) noexcept {
    switch (op) {
        case ThreadGroupOp::SHADER_EXECUTION_REORDER: return compute::CallOp::SHADER_EXECUTION_REORDER;
        case ThreadGroupOp::RASTER_QUAD_DDX: return compute::CallOp::DDX;
        case ThreadGroupOp::RASTER_QUAD_DDY: return compute::CallOp::DDY;
        case ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: return compute::CallOp::WARP_IS_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: return compute::CallOp::WARP_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: return compute::CallOp::WARP_ACTIVE_ALL_EQUAL;
        case ThreadGroupOp::WARP_ACTIVE_BIT_AND: return compute::CallOp::WARP_ACTIVE_BIT_AND;
        case ThreadGroupOp::WARP_ACTIVE_BIT_OR: return compute::CallOp::WARP_ACTIVE_BIT_OR;
        case ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return compute::CallOp::WARP_ACTIVE_BIT_XOR;
        case ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: return compute::CallOp::WARP_ACTIVE_COUNT_BITS;
        case ThreadGroupOp::WARP_ACTIVE_MAX: return compute::CallOp::WARP_ACTIVE_MAX;
        case ThreadGroupOp::WARP_ACTIVE_MIN: return compute::CallOp::WARP_ACTIVE_MIN;
        case ThreadGroupOp::WARP_ACTIVE_PRODUCT: return compute::CallOp::WARP_ACTIVE_PRODUCT;
        case ThreadGroupOp::WARP_ACTIVE_SUM: return compute::CallOp::WARP_ACTIVE_SUM;
        case ThreadGroupOp::WARP_ACTIVE_ALL: return compute::CallOp::WARP_ACTIVE_ALL;
        case ThreadGroupOp::WARP_ACTIVE_ANY: return compute::CallOp::WARP_ACTIVE_ANY;
        case ThreadGroupOp::WARP_ACTIVE_BIT_MASK: return compute::CallOp::WARP_ACTIVE_BIT_MASK;
        case ThreadGroupOp::WARP_PREFIX_COUNT_BITS: return compute::CallOp::WARP_PREFIX_COUNT_BITS;
        case ThreadGroupOp::WARP_PREFIX_SUM: return compute::CallOp::WARP_PREFIX_SUM;
        case ThreadGroupOp::WARP_PREFIX_PRODUCT: return compute::CallOp::WARP_PREFIX_PRODUCT;
        case ThreadGroupOp::WARP_READ_LANE: return compute::CallOp::WARP_READ_LANE;
        case ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: return compute::CallOp::WARP_READ_FIRST_ACTIVE_LANE;
        case ThreadGroupOp::SYNCHRONIZE_BLOCK: return compute::CallOp::SYNCHRONIZE_BLOCK;
        default: LUISA_NOT_IMPLEMENTED("Unsupported thread-group op '{}'.", xir::to_string(op));
    }
}

const Function *XIR2ASTImpl::_value_parent_function(const Value *value) noexcept {
    if (auto arg = xir_cast<Argument>(value)) { return arg->parent_function(); }
    if (auto inst = xir_cast<Instruction>(value)) { return inst->parent_function(); }
    return nullptr;
}

XIR2ASTImpl::Context *XIR2ASTImpl::_find_context(const Function *function) const noexcept {
    for (auto ctx = _ctx; ctx != nullptr; ctx = ctx->parent) {
        if (ctx->function == function) { return ctx; }
    }
    return nullptr;
}

const Expression *XIR2ASTImpl::_convert_call(const CallInst *inst) noexcept {
    auto callee = inst->callee();
    LUISA_ASSERT(callee != nullptr, "Null callee in call instruction.");
    LUISA_ASSERT(callee->isa<CallableFunction>(), "Only callable XIR call instructions are supported.");
    auto callable_callee = static_cast<const CallableFunction *>(callee);
    auto callable = _convert_callable(callable_callee);
    auto args = _convert_call_arguments(inst, callable_callee);
    auto function = callable->function();
    LUISA_ASSERT(function.tag() != compute::Function::Tag::COROUTINE,
                 "Unexpected coroutine callable '{}' reached XIR2AST call conversion.",
                 function.name());
    LUISA_ASSERT(inst->type() != nullptr, "Void call instruction must not be requested as a value.");
    return fb()->call(inst->type(), function, luisa::span{args});
}

luisa::vector<const Expression *> XIR2ASTImpl::_convert_call_arguments(const CallInst *inst,
                                                                       const CallableFunction *callee) noexcept {
    auto args = luisa::vector<const Expression *>{};
    args.reserve(inst->argument_count());
    auto formal_count = size_t{0u};
    for (auto _ : callee->arguments()) {
        static_cast<void>(_);
        formal_count++;
    }
    LUISA_ASSERT(formal_count == inst->argument_count(),
                 "Call argument count mismatch: expected {}, got {}.",
                 formal_count, inst->argument_count());
    auto formal_iter = callee->arguments().begin();
    for (auto i = 0u; i < inst->argument_count(); i++, ++formal_iter) {
        auto formal = *formal_iter;
        auto arg = inst->argument(i);
        if (formal->is_reference() || formal->is_resource()) {
            args.emplace_back(_convert_lvalue(arg));
        } else {
            args.emplace_back(_convert_value(arg));
        }
    }
    return args;
}

const Expression *XIR2ASTImpl::_convert_value(const Value *value) noexcept {
    if (auto owner = _value_parent_function(value);
        owner != nullptr && _ctx != nullptr && _ctx->function != owner) {
        auto owner_ctx = _find_context(owner);
        LUISA_ASSERT(owner_ctx != nullptr,
                     "Missing outer context for XIR value from another function.");
        auto old_ctx = std::exchange(_ctx, owner_ctx);
        auto expr = _convert_value(value);
        _ctx = old_ctx;
        return expr;
    }
    if (auto iter = _ctx->value_to_exprs.find(value);
        iter != _ctx->value_to_exprs.end()) {
        return iter->second;
    }
    auto expr = [&]() noexcept -> const Expression * {
        if (auto constant = xir_cast<Constant>(value)) { return _convert_constant(constant); }
        if (auto arg = xir_cast<Argument>(value)) { return _convert_argument(arg); }
        if (auto sreg = xir_cast<SpecialRegister>(value)) { return _convert_special_register(sreg); }
        if (auto alloca = xir_cast<AllocaInst>(value)) {
            _convert_alloca(alloca);
            return _ctx->value_to_exprs.at(alloca);
        }
        if (auto gep = xir_cast<GEPInst>(value)) { return _convert_lvalue(gep); }
        if (auto load = xir_cast<LoadInst>(value)) { return _convert_lvalue(load->variable()); }
        if (auto coro_id = xir_cast<CoroIdInst>(value)) {
            auto expr = fb()->coro_id();
            _ctx->value_to_exprs.emplace(coro_id, expr);
            return expr;
        }
        if (auto coro_token = xir_cast<CoroTokenInst>(value)) {
            auto expr = fb()->coro_token();
            _ctx->value_to_exprs.emplace(coro_token, expr);
            return expr;
        }
        if (auto arithmetic = xir_cast<ArithmeticInst>(value)) { return _convert_arithmetic(arithmetic); }
        if (auto cast = xir_cast<CastInst>(value)) {
            auto op = cast->op() == xir::CastOp::STATIC_CAST ? compute::CastOp::STATIC :
                                                              compute::CastOp::BITWISE;
            return fb()->cast(cast->type(), op, _convert_value(cast->value()));
        }
        if (auto call = xir_cast<CallInst>(value)) { return _convert_call(call); }
        if (auto resource_read = xir_cast<ResourceReadInst>(value)) {
            auto args = _convert_operands(resource_read);
            return fb()->call(resource_read->type(),
                              _resource_read_call_op(resource_read->op()),
                              luisa::span{args}, _curve_basis_set(resource_read));
        }
        if (auto resource_query = xir_cast<ResourceQueryInst>(value)) {
            auto args = _convert_operands(resource_query);
            return fb()->call(resource_query->type(),
                              _resource_query_call_op(resource_query->op()),
                              luisa::span{args}, _curve_basis_set(resource_query));
        }
        if (auto atomic = xir_cast<AtomicInst>(value)) {
            _convert_instruction(atomic);
            if (auto iter = _ctx->value_to_exprs.find(atomic);
                iter != _ctx->value_to_exprs.end()) {
                return iter->second;
            }
            LUISA_ERROR_WITH_LOCATION("Atomic instruction '{}' was requested as a value but was not materialized.",
                                      xir::to_string(atomic->op()));
        }
        if (auto thread_group = xir_cast<ThreadGroupInst>(value)) {
            if (thread_group->type() == nullptr) {
                LUISA_ERROR_WITH_LOCATION("thread-group op '{}' is a statement, not an expression.",
                                          xir::to_string(thread_group->op()));
            }
            auto args = _convert_operands(thread_group);
            return fb()->call(thread_group->type(),
                              _thread_group_call_op(thread_group->op()),
                              luisa::span{args});
        }
        LUISA_NOT_IMPLEMENTED("Unsupported XIR value '{}'.", xir::to_string(value->derived_value_tag()));
    }();
    _ctx->value_to_exprs.emplace(value, expr);
    return expr;
}

const Expression *XIR2ASTImpl::_convert_loop_condition(const BasicBlock *prepare,
                                                       const Value *condition) noexcept {
    LUISA_ASSERT(prepare != nullptr, "Loop prepare block must not be null.");
    auto current = condition;
    for (auto depth = 0u; depth < 16u; depth++) {
        auto load = xir_cast<LoadInst>(current);
        if (load == nullptr) { break; }
        const StoreInst *store = nullptr;
        for (auto inst : prepare->instructions()) {
            if (inst == prepare->terminator()) { break; }
            auto maybe_store = xir_cast<StoreInst>(inst);
            if (maybe_store != nullptr && maybe_store->variable() == load->variable()) {
                store = maybe_store;
            }
        }
        if (store == nullptr || store->value() == current) { break; }
        current = store->value();
    }
    return _convert_value(current);
}

bool XIR2ASTImpl::_is_argument_copy_store(const StoreInst *store) const noexcept {
    auto alloca = xir_cast<AllocaInst>(store->variable());
    auto arg = xir_cast<Argument>(store->value());
    if (alloca == nullptr || arg == nullptr) { return false; }
    if (auto iter = _ctx->argument_local_copies.find(alloca);
        iter != _ctx->argument_local_copies.end()) {
        return iter->second == arg;
    }
    return false;
}

void XIR2ASTImpl::_convert_instruction(const Instruction *inst) noexcept {
    if (auto alloca = xir_cast<AllocaInst>(inst)) {
        _convert_alloca(alloca);
        return;
    }
    if (auto store = xir_cast<StoreInst>(inst)) {
        if (_is_argument_copy_store(store)) { return; }
        _emit_comments(inst);
        fb()->assign(_convert_lvalue(store->variable()), _convert_value(store->value()));
        return;
    }
    if (auto print = xir_cast<PrintInst>(inst)) {
        _emit_comments(inst);
        auto args = _convert_operands(print);
        fb()->print_(luisa::string{print->format()}, luisa::span{args});
        return;
    }
    if (auto assert_ = xir_cast<AssertInst>(inst)) {
        _emit_comments(inst);
        auto args = luisa::vector<const Expression *>{_convert_value(assert_->condition())};
        if (!assert_->message().empty()) {
            args.emplace_back(fb()->string_id(luisa::string{assert_->message()}));
        }
        fb()->call(compute::CallOp::ASSERT, luisa::span{args});
        return;
    }
    if (auto assume = xir_cast<AssumeInst>(inst)) {
        _emit_comments(inst);
        auto args = luisa::vector<const Expression *>{_convert_value(assume->condition())};
        if (!assume->message().empty()) {
            args.emplace_back(fb()->string_id(luisa::string{assume->message()}));
        }
        fb()->call(compute::CallOp::ASSUME, luisa::span{args});
        return;
    }
    if (auto resource_write = xir_cast<ResourceWriteInst>(inst)) {
        _emit_comments(inst);
        auto args = _convert_operands(resource_write);
        fb()->call(_resource_write_call_op(resource_write->op()), luisa::span{args});
        return;
    }
    if (auto atomic = xir_cast<AtomicInst>(inst)) {
        _emit_comments(inst);
        luisa::vector<const Expression *> args;
        args.reserve(1u + atomic->index_count() + atomic->value_count());
        args.emplace_back(_convert_lvalue(atomic->base()));
        for (auto index_use : atomic->index_uses()) {
            args.emplace_back(_convert_value(index_use->value()));
        }
        for (auto value_use : atomic->value_uses()) {
            args.emplace_back(_convert_value(value_use->value()));
        }
        auto op = _atomic_call_op(atomic->op());
        if (atomic->use_list().empty()) {
            fb()->call(op, luisa::span{args});
        } else {
            auto local = fb()->local(atomic->type());
            auto expr = fb()->call(atomic->type(), op, luisa::span{args});
            fb()->assign(local, expr);
            _ctx->value_to_exprs.emplace(atomic, local);
        }
        return;
    }
    if (auto thread_group = xir_cast<ThreadGroupInst>(inst)) {
        _emit_comments(inst);
        auto args = _convert_operands(thread_group);
        if (thread_group->type() == nullptr) {
            fb()->call(_thread_group_call_op(thread_group->op()), luisa::span{args});
            return;
        }
        if (thread_group->use_list().empty()) {
            fb()->call(_thread_group_call_op(thread_group->op()), luisa::span{args});
        }
        return;
    }
    if (auto suspend = xir_cast<SuspendInst>(inst)) {
        _emit_comments(inst);
        fb()->suspend_token_(suspend->coro_token);
        return;
    }
    if (auto coro_register = xir_cast<CoroRegisterInst>(inst)) {
        _emit_comments(inst);
        fb()->bind_promise_(_convert_value(coro_register->value()), luisa::string{coro_register->name()});
        return;
    }
    if (auto call = xir_cast<CallInst>(inst)) {
        if (!call->use_list().empty()) { return; }
        auto callee = call->callee();
        LUISA_ASSERT(callee != nullptr, "Null callee in call instruction.");
        LUISA_ASSERT(callee->isa<CallableFunction>(), "Only callable XIR call instructions are supported.");
        auto callable_callee = static_cast<const CallableFunction *>(callee);
        auto callable = _convert_callable(callable_callee);
        auto args = _convert_call_arguments(call, callable_callee);
        auto function = callable->function();
        LUISA_ASSERT(function.tag() != compute::Function::Tag::COROUTINE,
                     "Unexpected coroutine callable '{}' reached XIR2AST statement call conversion.",
                     function.name());
        _emit_comments(inst);
        fb()->call(function, luisa::span{args});
        return;
    }
    if (inst->is_terminator()) { return; }
}

std::pair<const Expression *, const Expression *>
XIR2ASTImpl::_extract_for_update(const LoopInst *loop) noexcept {
    auto update = loop->update_block();
    LUISA_ASSERT(update != nullptr && update->is_terminated(), "Invalid for-loop update block.");
    auto term = xir_cast<BranchInst>(update->terminator());
    LUISA_ASSERT(term != nullptr && term->target_block() == loop->prepare_block(),
                 "Unexpected for-loop update terminator.");
    const StoreInst *store = nullptr;
    for (auto inst : update->instructions()) {
        if (inst == term) { break; }
        if (auto maybe_store = xir_cast<StoreInst>(inst)) {
            store = maybe_store;
        }
    }
    LUISA_ASSERT(store != nullptr, "For-loop update block does not contain a store.");
    auto add = xir_cast<ArithmeticInst>(store->value());
    LUISA_ASSERT(add != nullptr && add->op() == ArithmeticOp::BINARY_ADD,
                 "For-loop update is expected to be an add.");
    auto prev = xir_cast<LoadInst>(add->operand(0u));
    LUISA_ASSERT(prev != nullptr, "For-loop update add lhs is expected to be a load.");
    LUISA_ASSERT(prev->variable() == store->variable(),
                 "For-loop update add lhs does not match store target.");
    return {_convert_lvalue(store->variable()), _convert_value(add->operand(1u))};
}

void XIR2ASTImpl::_convert_block(const BasicBlock *block,
                                 luisa::span<const BasicBlock *const> exit_targets) noexcept {
    LUISA_ASSERT(block != nullptr, "Null block.");
    for (auto inst : block->instructions()) {
        if (inst->isa<IfInst>()) {
            auto if_inst = static_cast<const IfInst *>(inst);
            _emit_comments(inst);
            auto stmt = fb()->if_(_convert_value(if_inst->condition()));
            auto exits = std::array{if_inst->merge_block()};
            fb()->with(stmt->true_branch(), [&] {
                _convert_block(if_inst->true_block(), luisa::span{exits});
            });
            fb()->with(stmt->false_branch(), [&] {
                _convert_block(if_inst->false_block(), luisa::span{exits});
            });
            if (auto merge = if_inst->merge_block();
                merge != nullptr && !is_exit_target(merge, exit_targets)) {
                _convert_block(merge, exit_targets);
            }
            return;
        }
        if (inst->isa<SwitchInst>()) {
            auto switch_inst = static_cast<const SwitchInst *>(inst);
            _emit_comments(inst);
            auto stmt = fb()->switch_(_convert_value(switch_inst->value()));
            fb()->with(stmt->body(), [&] {
                auto exits = std::array{switch_inst->merge_block()};
                for (auto i = 0u; i < switch_inst->case_count(); i++) {
                    auto case_type = switch_inst->value()->type();
                    const Expression *case_value = nullptr;
                    switch (case_type->tag()) {
                        case Type::Tag::INT8: case_value = fb()->literal(case_type, static_cast<byte>(switch_inst->case_value(i))); break;
                        case Type::Tag::UINT8: case_value = fb()->literal(case_type, static_cast<ubyte>(switch_inst->case_value(i))); break;
                        case Type::Tag::INT16: case_value = fb()->literal(case_type, static_cast<short>(switch_inst->case_value(i))); break;
                        case Type::Tag::UINT16: case_value = fb()->literal(case_type, static_cast<ushort>(switch_inst->case_value(i))); break;
                        case Type::Tag::INT32: case_value = fb()->literal(case_type, static_cast<int>(switch_inst->case_value(i))); break;
                        case Type::Tag::UINT32: case_value = fb()->literal(case_type, static_cast<uint>(switch_inst->case_value(i))); break;
                        default: LUISA_ERROR_WITH_LOCATION("Unsupported switch case type '{}'.", case_type->description());
                    }
                    auto case_stmt = fb()->case_(case_value);
                    fb()->with(case_stmt->body(), [&] {
                        _convert_block(switch_inst->case_block(i), luisa::span{exits});
                    });
                }
                if (auto default_block = switch_inst->default_block(); default_block != nullptr) {
                    auto default_stmt = fb()->default_();
                    fb()->with(default_stmt->body(), [&] {
                        _convert_block(default_block, luisa::span{exits});
                    });
                }
            });
            if (auto merge = switch_inst->merge_block();
                merge != nullptr && !is_exit_target(merge, exit_targets)) {
                _convert_block(merge, exit_targets);
            }
            return;
        }
        if (inst->isa<SimpleLoopInst>()) {
            auto loop_inst = static_cast<const SimpleLoopInst *>(inst);
            _emit_comments(inst);
            auto stmt = fb()->loop_();
            auto exits = std::array{loop_inst->body_block()};
            fb()->with(stmt->body(), [&] {
                _convert_block(loop_inst->body_block(), luisa::span{exits});
            });
            if (auto merge = loop_inst->merge_block();
                merge != nullptr && !is_exit_target(merge, exit_targets)) {
                _convert_block(merge, exit_targets);
            }
            return;
        }
        if (inst->isa<LoopInst>()) {
            auto loop_inst = static_cast<const LoopInst *>(inst);
            auto prepare = loop_inst->prepare_block();
            LUISA_ASSERT(prepare != nullptr && prepare->is_terminated(),
                         "Invalid for-loop prepare block.");
            auto cond_br = xir_cast<ConditionalBranchInst>(prepare->terminator());
            LUISA_ASSERT(cond_br != nullptr &&
                             cond_br->true_block() == loop_inst->body_block() &&
                             cond_br->false_block() == loop_inst->merge_block(),
                         "LoopInst is not in AST2XIR for-loop form.");
            auto [var, step] = _extract_for_update(loop_inst);
            _emit_comments(inst);
            auto stmt = fb()->for_(var, _convert_loop_condition(prepare, cond_br->condition()), step);
            auto exits = std::array{loop_inst->update_block()};
            fb()->with(stmt->body(), [&] {
                _convert_block(loop_inst->body_block(), luisa::span{exits});
            });
            if (auto merge = loop_inst->merge_block();
                merge != nullptr && !is_exit_target(merge, exit_targets)) {
                _convert_block(merge, exit_targets);
            }
            return;
        }
        if (auto branch = xir_cast<BranchInst>(inst)) {
            LUISA_ASSERT(is_exit_target(branch->target_block(), exit_targets),
                         "Unexpected branch target in XIR2AST.");
            return;
        }
        if (inst->isa<BreakInst>()) {
            _emit_comments(inst);
            fb()->break_();
            return;
        }
        if (inst->isa<ContinueInst>()) {
            _emit_comments(inst);
            fb()->continue_();
            return;
        }
        if (auto return_ = xir_cast<ReturnInst>(inst)) {
            _emit_comments(inst);
            fb()->return_(return_->return_value() == nullptr ? nullptr :
                          _convert_value(return_->return_value()));
            return;
        }
        if (auto unreachable = xir_cast<UnreachableInst>(inst)) {
            _emit_comments(inst);
            luisa::vector<const Expression *> args;
            if (!unreachable->message().empty()) {
                args.emplace_back(fb()->string_id(luisa::string{unreachable->message()}));
            }
            fb()->call(compute::CallOp::UNREACHABLE, luisa::span{args});
            return;
        }
        if (inst->isa<ConditionalBranchInst>()) {
            LUISA_ERROR_WITH_LOCATION("Unexpected low-level conditional branch outside loop prepare block.");
        }
        _convert_instruction(inst);
    }
}

void XIR2ASTImpl::_analyze_argument_local_copies(const FunctionDefinition *function) noexcept {
    auto body = function->body_block();
    LUISA_ASSERT(body != nullptr, "Function has no body block.");
    for (auto inst : body->instructions()) {
        auto store = xir_cast<StoreInst>(inst);
        if (store == nullptr) { continue; }
        auto alloca = xir_cast<AllocaInst>(store->variable());
        auto arg = xir_cast<Argument>(store->value());
        if (alloca == nullptr || arg == nullptr) { continue; }
        if (!alloca->is_local()) { continue; }
        if (comment_of(alloca) == luisa::string_view{"Local copy of argument"} &&
            arg->derived_argument_tag() == DerivedArgumentTag::VALUE) {
            _ctx->argument_local_copies.emplace(alloca, arg);
        }
    }
}

luisa::shared_ptr<const ASTFunctionBuilder>
XIR2ASTImpl::_convert_kernel(const KernelFunction *kernel) noexcept {
    return ASTFunctionBuilder::define_kernel([&] {
        Context ctx;
        auto old_ctx = std::exchange(_ctx, &ctx);
        _ctx->function = kernel;
        _ctx->parent = old_ctx;
        if (auto name = kernel->name(); name.has_value() && !name->empty()) {
            fb()->set_name(*name);
        }
        fb()->set_block_size(kernel->block_size());
        for (auto arg : kernel->arguments()) {
            _ctx->value_to_exprs.emplace(arg, _convert_argument(arg));
        }
        _analyze_argument_local_copies(kernel);
        _convert_block(kernel->body_block(), {});
        _ctx = old_ctx;
    });
}

luisa::shared_ptr<const ASTFunctionBuilder>
XIR2ASTImpl::_convert_callable(const CallableFunction *callable) noexcept {
    if (auto iter = _converted_callables.find(callable);
        iter != _converted_callables.end()) {
        return iter->second;
    }
    auto has_coro_markers = false;
    callable->traverse_instructions([&](const Instruction *inst) noexcept {
        has_coro_markers = has_coro_markers ||
                           inst->isa<SuspendInst>() ||
                           inst->isa<CoroIdInst>() ||
                           inst->isa<CoroTokenInst>() ||
                           inst->isa<CoroRegisterInst>();
    });
    auto convert_body = [&] {
        Context ctx;
        auto old_ctx = std::exchange(_ctx, &ctx);
        _ctx->function = callable;
        _ctx->parent = old_ctx;
        if (auto name = callable->name(); name.has_value() && !name->empty()) {
            fb()->set_name(*name);
        }
        for (auto arg : callable->arguments()) {
            _ctx->value_to_exprs.emplace(arg, _convert_argument(arg));
        }
        _analyze_argument_local_copies(callable);
        _convert_block(callable->body_block(), {});
        _ctx = old_ctx;
    };
    auto converted = has_coro_markers ?
                         ASTFunctionBuilder::define_coroutine(convert_body) :
                         ASTFunctionBuilder::define_callable(convert_body);
    _converted_callables.emplace(callable, converted);
    return converted;
}

luisa::shared_ptr<const ASTFunctionBuilder>
XIR2ASTImpl::build(const KernelFunction *kernel) noexcept {
    LUISA_ASSERT(kernel != nullptr, "Kernel must not be null.");
    return _convert_kernel(kernel);
}

luisa::shared_ptr<const ASTFunctionBuilder>
XIR2ASTImpl::build(const CallableFunction *callable) noexcept {
    LUISA_ASSERT(callable != nullptr, "Callable must not be null.");
    return _convert_callable(callable);
}

}// namespace

luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>
XIR2AST::build(const KernelFunction *kernel) noexcept {
    return XIR2ASTImpl{}.build(kernel);
}

luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>
XIR2AST::build(const CallableFunction *callable) noexcept {
    return XIR2ASTImpl{}.build(callable);
}

}// namespace luisa::compute::xir
