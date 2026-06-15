#pragma once

#include <luisa/ast/type_registry.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/assert.h>
#include <luisa/xir/instructions/assume.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/outline.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/print.h>
#include <luisa/xir/instructions/debug_break.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/raster_discard.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/thread_group.h>
#include <luisa/xir/instructions/unreachable.h>

namespace luisa::compute::xir {

class LUISA_XIR_API XIRBuilder : luisa::concepts::Noncopyable {

private:
    Instruction *_insertion_point = nullptr;

private:
    template<typename T, typename... Args>
    [[nodiscard]] auto _create_and_append_instruction(Args &&...args) noexcept -> T *;

public:
    XIRBuilder() noexcept;
    ~XIRBuilder() noexcept = default;
    void set_insertion_point(Instruction *insertion_point) noexcept;
    void set_insertion_point(BasicBlock *block) noexcept;
    [[nodiscard]] auto insertion_point() noexcept -> Instruction * { return _insertion_point; }
    [[nodiscard]] auto insertion_point() const noexcept -> const Instruction * { return _insertion_point; }

    [[nodiscard]] auto is_insertion_point_terminator() const noexcept {
        return _insertion_point != nullptr &&
               _insertion_point->is_terminator();
    }

public:
    Instruction *append(ManagedPtr<Instruction> inst) noexcept;

    IfInst *if_(Value *cond) noexcept;
    SwitchInst *switch_(Value *value) noexcept;
    LoopInst *loop() noexcept;
    SimpleLoopInst *simple_loop() noexcept;

    BranchInst *br(BasicBlock *target = nullptr) noexcept;
    ConditionalBranchInst *cond_br(Value *cond, BasicBlock *true_target = nullptr, BasicBlock *false_target = nullptr) noexcept;

    BreakInst *break_(BasicBlock *target_block = nullptr) noexcept;
    ContinueInst *continue_(BasicBlock *target_block = nullptr) noexcept;
    UnreachableInst *unreachable_(luisa::string_view message = {}) noexcept;
    ReturnInst *return_(Value *value) noexcept;
    ReturnInst *return_void() noexcept;
    RasterDiscardInst *raster_discard() noexcept;

    AssertInst *assert_(Value *condition, luisa::string_view message = {}) noexcept;
    AssumeInst *assume_(Value *condition, luisa::string_view message = {}) noexcept;

    CallInst *call(const Type *type, Function *callee, luisa::span<Value *const> arguments) noexcept;
    CallInst *call(const Type *type, Function *callee, std::initializer_list<Value *> arguments) noexcept;

    AutodiffIntrinsicInst *call(const Type *type, AutodiffIntrinsicOp op, luisa::span<Value *const> arguments) noexcept;
    AutodiffIntrinsicInst *call(const Type *type, AutodiffIntrinsicOp op, std::initializer_list<Value *> arguments) noexcept;

    AtomicInst *call(const Type *type, AtomicOp op, Value *base, luisa::span<Value *const> indices, luisa::span<Value *const> values) noexcept;
    AtomicInst *call(const Type *type, AtomicOp op, Value *base, luisa::span<Value *const> indices, std::initializer_list<Value *> values) noexcept;

    ThreadGroupInst *call(const Type *type, ThreadGroupOp op, luisa::span<Value *const> operands) noexcept;
    ThreadGroupInst *call(const Type *type, ThreadGroupOp op, std::initializer_list<Value *> operands) noexcept;

    ArithmeticInst *call(const Type *type, ArithmeticOp op, luisa::span<Value *const> operands) noexcept;
    ArithmeticInst *call(const Type *type, ArithmeticOp op, std::initializer_list<Value *> operands) noexcept;

    ResourceQueryInst *call(const Type *type, ResourceQueryOp op, luisa::span<Value *const> operands) noexcept;
    ResourceQueryInst *call(const Type *type, ResourceQueryOp op, std::initializer_list<Value *> operands) noexcept;

    ResourceReadInst *call(const Type *type, ResourceReadOp op, luisa::span<Value *const> operands) noexcept;
    ResourceReadInst *call(const Type *type, ResourceReadOp op, std::initializer_list<Value *> operands) noexcept;

    ResourceWriteInst *call(ResourceWriteOp op, luisa::span<Value *const> operands) noexcept;
    ResourceWriteInst *call(ResourceWriteOp op, std::initializer_list<Value *> operands) noexcept;

    CastInst *cast_(const Type *type, CastOp op, Value *value) noexcept;

    Instruction *static_cast_(const Type *type, Value *value) noexcept;
    CastInst *bit_cast_(const Type *type, Value *value) noexcept;

    Value *static_cast_if_necessary(const Type *type, Value *value) noexcept;
    Value *bit_cast_if_necessary(const Type *type, Value *value) noexcept;

    PhiInst *phi(const Type *type, luisa::span<const PhiIncoming> incomings = {}) noexcept;
    PhiInst *phi(const Type *type, std::initializer_list<PhiIncoming> incomings) noexcept;

    PrintInst *print(luisa::string format, luisa::span<Value *const> values) noexcept;
    PrintInst *print(luisa::string format, std::initializer_list<Value *> values) noexcept;

    DebugBreakInst *debug_break(DebugBreakInst::Callback callback = nullptr) noexcept;

    AllocaInst *alloca_(const Type *type, AllocaOp space) noexcept;
    AllocaInst *alloca_local(const Type *type) noexcept;
    AllocaInst *alloca_shared(const Type *type) noexcept;

    GEPInst *gep(const Type *type, Value *base, luisa::span<Value *const> indices) noexcept;
    GEPInst *gep(const Type *type, Value *base, std::initializer_list<Value *> indices) noexcept;

    LoadInst *load(const Type *type, Value *variable) noexcept;
    StoreInst *store(Value *variable, Value *value) noexcept;

    ClockInst *clock() noexcept;

    OutlineInst *outline() noexcept;

    AutodiffScopeInst *autodiff_scope() noexcept;

    CoroSuspendInst *coro_suspend(uint32_t token, luisa::string name, Value *frame) noexcept;
    CoroResumeInst *coro_resume(uint32_t token, Value *frame) noexcept;
    CoroTerminateInst *coro_terminate() noexcept;


    RayQueryLoopInst *ray_query_loop() noexcept;
    RayQueryDispatchInst *ray_query_dispatch(Value *query_object) noexcept;

    RayQueryObjectReadInst *call(const Type *type, RayQueryObjectReadOp op, luisa::span<Value *const> operands) noexcept;
    RayQueryObjectReadInst *call(const Type *type, RayQueryObjectReadOp op, std::initializer_list<Value *> operands) noexcept;
    RayQueryObjectWriteInst *call(RayQueryObjectWriteOp op, luisa::span<Value *const> operands) noexcept;
    RayQueryObjectWriteInst *call(RayQueryObjectWriteOp op, std::initializer_list<Value *> operands) noexcept;

    RayQueryPipelineInst *ray_query_pipeline(Value *query_object = nullptr,
                                             Function *on_surface = nullptr,
                                             Function *on_procedural = nullptr,
                                             luisa::span<Value *const> captured_args = {}) noexcept;

    AtomicInst *atomic_fetch_add(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_sub(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_and(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_or(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_xor(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_min(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_fetch_max(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_exchange(const Type *type, Value *base, luisa::span<Value *const> indices, Value *value) noexcept;
    AtomicInst *atomic_compare_exchange(const Type *type, Value *base, luisa::span<Value *const> indices, Value *expected, Value *desired) noexcept;

    ThreadGroupInst *shader_execution_reorder() noexcept;
    ThreadGroupInst *shader_execution_reorder(Value *hint, Value *hint_bits) noexcept;
    ThreadGroupInst *synchronize_block() noexcept;
    ThreadGroupInst *raster_quad_ddx(const Type *type, Value *value) noexcept;
    ThreadGroupInst *raster_quad_ddy(const Type *type, Value *value) noexcept;
};

}// namespace luisa::compute::xir
