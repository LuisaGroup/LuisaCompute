#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::xir {

XIRBuilder::XIRBuilder() noexcept = default;

template<typename T, typename... Args>
auto XIRBuilder::_create_and_append_instruction(Args &&...args) noexcept -> T * {
    LUISA_DEBUG_ASSERT(_insertion_point != nullptr, "Invalid insertion point.");
    auto inst = luisa::make_managed<T>(std::forward<Args>(args)...);
    return static_cast<T *>(append(std::move(inst)));
}

Instruction *XIRBuilder::append(ManagedPtr<Instruction> inst) noexcept {
    auto p = _insertion_point->insert_after_self(std::move(inst));
    set_insertion_point(p);
    return p;
}

IfInst *XIRBuilder::if_(Value *cond) noexcept {
    LUISA_ASSERT(cond != nullptr && cond->type() == Type::of<bool>(), "Invalid condition.");
    return _create_and_append_instruction<IfInst>(_insertion_point->parent_block(), cond);
}

SwitchInst *XIRBuilder::switch_(Value *value) noexcept {
    LUISA_ASSERT(value != nullptr, "Switch value cannot be null.");
    return _create_and_append_instruction<SwitchInst>(_insertion_point->parent_block(), value);
}

LoopInst *XIRBuilder::loop() noexcept {
    return _create_and_append_instruction<LoopInst>(_insertion_point->parent_block());
}

SimpleLoopInst *XIRBuilder::simple_loop() noexcept {
    return _create_and_append_instruction<SimpleLoopInst>(_insertion_point->parent_block());
}

BranchInst *XIRBuilder::br(BasicBlock *target) noexcept {
    auto inst = _create_and_append_instruction<BranchInst>(_insertion_point->parent_block());
    inst->set_target_block(target);
    return inst;
}

ConditionalBranchInst *XIRBuilder::cond_br(Value *cond, BasicBlock *true_target, BasicBlock *false_target) noexcept {
    LUISA_ASSERT(cond != nullptr && cond->type() == Type::of<bool>(), "Invalid condition.");
    auto inst = _create_and_append_instruction<ConditionalBranchInst>(_insertion_point->parent_block(), cond);
    inst->set_true_target(true_target);
    inst->set_false_target(false_target);
    return inst;
}

BreakInst *XIRBuilder::break_(BasicBlock *target_block) noexcept {
    auto inst = _create_and_append_instruction<BreakInst>(_insertion_point->parent_block());
    inst->set_target_block(target_block);
    return inst;
}

ContinueInst *XIRBuilder::continue_(BasicBlock *target_block) noexcept {
    auto inst = _create_and_append_instruction<ContinueInst>(_insertion_point->parent_block());
    inst->set_target_block(target_block);
    return inst;
}

UnreachableInst *XIRBuilder::unreachable_(luisa::string_view message) noexcept {
    return _create_and_append_instruction<UnreachableInst>(_insertion_point->parent_block(), luisa::string{message});
}

AssertInst *XIRBuilder::assert_(Value *condition, luisa::string_view message) noexcept {
    return _create_and_append_instruction<AssertInst>(_insertion_point->parent_block(), condition, luisa::string{message});
}

AssumeInst *XIRBuilder::assume_(Value *condition, luisa::string_view message) noexcept {
    return _create_and_append_instruction<AssumeInst>(_insertion_point->parent_block(), condition, luisa::string{message});
}

ReturnInst *XIRBuilder::return_(Value *value) noexcept {
    return _create_and_append_instruction<ReturnInst>(_insertion_point->parent_block(), value);
}

ReturnInst *XIRBuilder::return_void() noexcept {
    return _create_and_append_instruction<ReturnInst>(_insertion_point->parent_block());
}

RasterDiscardInst *XIRBuilder::raster_discard() noexcept {
    return _create_and_append_instruction<RasterDiscardInst>(_insertion_point->parent_block());
}

CallInst *XIRBuilder::call(const Type *type, Function *callee, luisa::span<Value *const> arguments) noexcept {
    return _create_and_append_instruction<CallInst>(_insertion_point->parent_block(), type, callee, arguments);
}

CallInst *XIRBuilder::call(const Type *type, Function *callee, std::initializer_list<Value *> arguments) noexcept {
    return _create_and_append_instruction<CallInst>(_insertion_point->parent_block(), type, callee, luisa::span{arguments.begin(), arguments.end()});
}

AutodiffIntrinsicInst *XIRBuilder::call(const Type *type, AutodiffIntrinsicOp op, std::initializer_list<Value *> arguments) noexcept {
    return _create_and_append_instruction<AutodiffIntrinsicInst>(_insertion_point->parent_block(), type, op, luisa::span{arguments.begin(), arguments.end()});
}

AutodiffIntrinsicInst *XIRBuilder::call(const Type *type, AutodiffIntrinsicOp op, luisa::span<Value *const> arguments) noexcept {
    return _create_and_append_instruction<AutodiffIntrinsicInst>(_insertion_point->parent_block(), type, op, arguments);
}

PhiInst *XIRBuilder::phi(const Type *type, std::initializer_list<PhiIncoming> incomings) noexcept {
    auto inst = _create_and_append_instruction<PhiInst>(_insertion_point->parent_block(), type);
    for (auto incoming : incomings) { inst->add_incoming(incoming.value, incoming.block); }
    return inst;
}

PrintInst *XIRBuilder::print(luisa::string format, std::initializer_list<Value *> values) noexcept {
    return _create_and_append_instruction<PrintInst>(_insertion_point->parent_block(), std::move(format), luisa::span{values.begin(), values.end()});
}

DebugBreakInst *XIRBuilder::debug_break(DebugBreakInst::Callback callback) noexcept {
    return _create_and_append_instruction<DebugBreakInst>(_insertion_point->parent_block(), callback);
}

AllocaInst *XIRBuilder::alloca_(const Type *type, AllocaOp space) noexcept {
    return _create_and_append_instruction<AllocaInst>(_insertion_point->parent_block(), type, space);
}

AllocaInst *XIRBuilder::alloca_local(const Type *type) noexcept {
    return alloca_(type, AllocaOp::LOCAL);
}

AllocaInst *XIRBuilder::alloca_shared(const Type *type) noexcept {
    return alloca_(type, AllocaOp::SHARED);
}

GEPInst *XIRBuilder::gep(const Type *type, Value *base, std::initializer_list<Value *> indices) noexcept {
    return _create_and_append_instruction<GEPInst>(_insertion_point->parent_block(), type, base, luisa::span{indices.begin(), indices.end()});
}

CastInst *XIRBuilder::cast_(const Type *type, CastOp op, Value *value) noexcept {
    return _create_and_append_instruction<CastInst>(_insertion_point->parent_block(), type, op, value);
}

Instruction *XIRBuilder::static_cast_(const Type *type, Value *value) noexcept {
    LUISA_ASSERT(type->is_scalar() && value->type()->is_scalar(), "Invalid cast operation.");
    return _create_and_append_instruction<CastInst>(_insertion_point->parent_block(), type, CastOp::STATIC_CAST, value);
}

CastInst *XIRBuilder::bit_cast_(const Type *type, Value *value) noexcept {
    return _create_and_append_instruction<CastInst>(_insertion_point->parent_block(), type, CastOp::BITWISE_CAST, value);
}

Value *XIRBuilder::static_cast_if_necessary(const Type *type, Value *value) noexcept {
    return value->type() == type ? value : static_cast_(type, value);
}

Value *XIRBuilder::bit_cast_if_necessary(const Type *type, Value *value) noexcept {
    return value->type() == type ? value : bit_cast_(type, value);
}

PhiInst *XIRBuilder::phi(const Type *type, luisa::span<const PhiIncoming> incomings) noexcept {
    auto inst = _create_and_append_instruction<PhiInst>(_insertion_point->parent_block(), type);
    for (auto incoming : incomings) { inst->add_incoming(incoming.value, incoming.block); }
    return inst;
}

PrintInst *XIRBuilder::print(luisa::string format, luisa::span<Value *const> values) noexcept {
    return _create_and_append_instruction<PrintInst>(_insertion_point->parent_block(), std::move(format), values);
}

GEPInst *XIRBuilder::gep(const Type *type, Value *base, luisa::span<Value *const> indices) noexcept {
    return _create_and_append_instruction<GEPInst>(_insertion_point->parent_block(), type, base, indices);
}

LoadInst *XIRBuilder::load(const Type *type, Value *variable) noexcept {
    LUISA_ASSERT(variable->is_lvalue(), "Load source must be an lvalue.");
    LUISA_ASSERT(type == variable->type(), "Type mismatch in Load");
    return _create_and_append_instruction<LoadInst>(_insertion_point->parent_block(), type, variable);
}

StoreInst *XIRBuilder::store(Value *variable, Value *value) noexcept {
    LUISA_ASSERT(variable->is_lvalue(), "Store destination must be an lvalue.");
    LUISA_ASSERT(!value->is_lvalue(), "Store source cannot be an lvalue.");
    LUISA_ASSERT(variable->type() == value->type(), "Type mismatch in Store");
    return _create_and_append_instruction<StoreInst>(_insertion_point->parent_block(), variable, value);
}

ClockInst *XIRBuilder::clock() noexcept {
    return _create_and_append_instruction<ClockInst>(_insertion_point->parent_block());
}

OutlineInst *XIRBuilder::outline() noexcept {
    return _create_and_append_instruction<OutlineInst>(_insertion_point->parent_block());
}

AutodiffScopeInst *XIRBuilder::autodiff_scope() noexcept {
    return _create_and_append_instruction<AutodiffScopeInst>(_insertion_point->parent_block());
}

CoroSuspendInst *XIRBuilder::coro_suspend(uint32_t token, luisa::string name, Value *frame) noexcept {
    return _create_and_append_instruction<CoroSuspendInst>(_insertion_point->parent_block(), token, std::move(name), frame);
}

CoroResumeInst *XIRBuilder::coro_resume(uint32_t token, Value *frame) noexcept {
    return _create_and_append_instruction<CoroResumeInst>(_insertion_point->parent_block(), token, frame);
}

CoroTerminateInst *XIRBuilder::coro_terminate() noexcept {
    return _create_and_append_instruction<CoroTerminateInst>(_insertion_point->parent_block());
}

CoroRegisterInst *XIRBuilder::coro_register(luisa::string name, Value *value, Value *frame) noexcept {
    return _create_and_append_instruction<CoroRegisterInst>(_insertion_point->parent_block(), std::move(name), value, frame);
}

RayQueryLoopInst *XIRBuilder::ray_query_loop() noexcept {
    return _create_and_append_instruction<RayQueryLoopInst>(_insertion_point->parent_block());
}

RayQueryDispatchInst *XIRBuilder::ray_query_dispatch(Value *query_object) noexcept {
    return _create_and_append_instruction<RayQueryDispatchInst>(_insertion_point->parent_block(), query_object);
}

RayQueryObjectReadInst *XIRBuilder::call(const Type *type, RayQueryObjectReadOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<RayQueryObjectReadInst>(_insertion_point->parent_block(), type, op, operands);
}

RayQueryObjectReadInst *XIRBuilder::call(const Type *type, RayQueryObjectReadOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(type, op, luisa::span{operands.begin(), operands.end()});
}

RayQueryObjectWriteInst *XIRBuilder::call(RayQueryObjectWriteOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<RayQueryObjectWriteInst>(_insertion_point->parent_block(), op, operands);
}

RayQueryObjectWriteInst *XIRBuilder::call(RayQueryObjectWriteOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(op, luisa::span{operands.begin(), operands.end()});
}

RayQueryPipelineInst *XIRBuilder::ray_query_pipeline(Value *query_object, Function *on_surface, Function *on_procedural,
                                                     luisa::span<Value *const> captured_args) noexcept {
    return _create_and_append_instruction<RayQueryPipelineInst>(_insertion_point->parent_block(), query_object, on_surface, on_procedural, captured_args);
}

ThreadGroupInst *XIRBuilder::call(const Type *type, ThreadGroupOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<ThreadGroupInst>(_insertion_point->parent_block(), type, op, operands);
}

ThreadGroupInst *XIRBuilder::call(const Type *type, ThreadGroupOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(type, op, luisa::span{operands.begin(), operands.end()});
}

ThreadGroupInst *XIRBuilder::shader_execution_reorder() noexcept {
    return this->call(nullptr, ThreadGroupOp::SHADER_EXECUTION_REORDER, {});
}

ThreadGroupInst *XIRBuilder::shader_execution_reorder(Value *hint, Value *hint_bits) noexcept {
    return this->call(nullptr, ThreadGroupOp::SHADER_EXECUTION_REORDER, std::array{hint, hint_bits});
}

ThreadGroupInst *XIRBuilder::synchronize_block() noexcept {
    return this->call(nullptr, ThreadGroupOp::SYNCHRONIZE_BLOCK, {});
}

ThreadGroupInst *XIRBuilder::raster_quad_ddx(const Type *type, Value *value) noexcept {
    return this->call(type, ThreadGroupOp::RASTER_QUAD_DDX, std::array{value});
}

ThreadGroupInst *XIRBuilder::raster_quad_ddy(const Type *type, Value *value) noexcept {
    return this->call(type, ThreadGroupOp::RASTER_QUAD_DDY, std::array{value});
}

AtomicInst *XIRBuilder::call(const Type *type, AtomicOp op, Value *base, luisa::span<Value *const> indices, luisa::span<Value *const> values) noexcept {
    return _create_and_append_instruction<AtomicInst>(_insertion_point->parent_block(), type, op, base, indices, values);
}

AtomicInst *XIRBuilder::call(const Type *type, AtomicOp op, Value *base, luisa::span<Value *const> indices, std::initializer_list<Value *> values) noexcept {
    return this->call(type, op, base, indices, luisa::span{values.begin(), values.end()});
}

ArithmeticInst *XIRBuilder::call(const Type *type, ArithmeticOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<ArithmeticInst>(_insertion_point->parent_block(), type, op, operands);
}

ArithmeticInst *XIRBuilder::call(const Type *type, ArithmeticOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(type, op, luisa::span{operands.begin(), operands.end()});
}

ResourceQueryInst *XIRBuilder::call(const Type *type, ResourceQueryOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<ResourceQueryInst>(_insertion_point->parent_block(), type, op, operands);
}

ResourceQueryInst *XIRBuilder::call(const Type *type, ResourceQueryOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(type, op, luisa::span{operands.begin(), operands.end()});
}

ResourceReadInst *XIRBuilder::call(const Type *type, ResourceReadOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<ResourceReadInst>(_insertion_point->parent_block(), type, op, operands);
}

ResourceReadInst *XIRBuilder::call(const Type *type, ResourceReadOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(type, op, luisa::span{operands.begin(), operands.end()});
}

ResourceWriteInst *XIRBuilder::call(ResourceWriteOp op, luisa::span<Value *const> operands) noexcept {
    return _create_and_append_instruction<ResourceWriteInst>(_insertion_point->parent_block(), op, operands);
}

ResourceWriteInst *XIRBuilder::call(ResourceWriteOp op, std::initializer_list<Value *> operands) noexcept {
    return this->call(op, luisa::span{operands.begin(), operands.end()});
}

AtomicInst *XIRBuilder::atomic_fetch_add(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_ADD, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_sub(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_SUB, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_and(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_AND, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_or(const Type *type, Value *base,
                                        luisa::span<Value *const> indices,
                                        Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_OR, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_xor(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_XOR, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_min(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_MIN, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_fetch_max(const Type *type, Value *base,
                                         luisa::span<Value *const> indices,
                                         Value *value) noexcept {
    return this->call(type, AtomicOp::FETCH_MAX, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_exchange(const Type *type, Value *base,
                                        luisa::span<Value *const> indices,
                                        Value *value) noexcept {
    return this->call(type, AtomicOp::EXCHANGE, base, indices, std::array{value});
}

AtomicInst *XIRBuilder::atomic_compare_exchange(const Type *type, Value *base,
                                                luisa::span<Value *const> indices,
                                                Value *expected, Value *desired) noexcept {
    return this->call(type, AtomicOp::COMPARE_EXCHANGE, base, indices, std::array{expected, desired});
}

void XIRBuilder::set_insertion_point(Instruction *insertion_point) noexcept {
    _insertion_point = insertion_point;
}

void XIRBuilder::set_insertion_point(BasicBlock *block) noexcept {
    LUISA_ASSERT(block != nullptr, "Insertion point block cannot be null.");
    set_insertion_point(block->instructions().tail_sentinel()->prev());
}

}// namespace luisa::compute::xir
