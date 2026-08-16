//
// Created by mike on 4/8/26.
//

#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/debug_printer.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/store.h>

#include <llvm/ADT/SmallPtrSet.h>

#include <algorithm>
#include <array>
#include <limits>
#include <cstdlib>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

namespace {

// Byte layout shared with LuisaPipelineRayQueryState in the native HIPRT
// wrapper. A compact callback transaction initializes only these observable
// fields; named offsets keep the cross-bitcode ABI reviewable and are exercised
// by the native callback regression rather than hidden behind numeric GEPs.
namespace compact_query_layout {
constexpr auto ray_origin_x = 16u;
constexpr auto ray_origin_y = 20u;
constexpr auto ray_origin_z = 24u;
constexpr auto ray_t_min = 28u;
constexpr auto ray_direction_x = 32u;
constexpr auto ray_direction_y = 36u;
constexpr auto ray_direction_z = 40u;
constexpr auto ray_t_max = 44u;
constexpr auto candidate_instance = 48u;
constexpr auto candidate_primitive = 52u;
constexpr auto candidate_u = 56u;
constexpr auto candidate_v = 60u;
constexpr auto candidate_t = 64u;
constexpr auto query_address = 88u;
constexpr auto flags = 96u;
constexpr auto state = 104u;
constexpr auto candidate_committed = 106u;
constexpr auto terminated = 107u;
}// namespace compact_query_layout

static_assert(compact_query_layout::terminated + sizeof(uint8_t) <=
              hip_synchronous_ray_query_state_size);

void accumulate_ray_query_handler_observations(
    const xir::Function *function,
    llvm::DenseSet<const xir::Function *> &visited,
    uint32_t &mask) noexcept {
    if (function == nullptr || !visited.insert(function).second) { return; }
    auto definition = function->definition();
    if (definition == nullptr) {
        // A reachable external/native Callable has no inspectable body. It may
        // observe any query component passed through its arguments, so compact
        // eligibility must fail closed instead of treating absence of XIR as
        // absence of observation.
        mask |= HIPCodegenLLVMImpl::llvm_ray_query_observes_committed_hit |
                HIPCodegenLLVMImpl::llvm_ray_query_observes_world_ray |
                HIPCodegenLLVMImpl::llvm_ray_query_observes_object_ray;
        return;
    }
    definition->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            if (instruction->isa<xir::RayQueryObjectReadInst>()) {
                auto read = static_cast<
                    const xir::RayQueryObjectReadInst *>(instruction);
                switch (read->op()) {
                    case xir::RayQueryObjectReadOp::
                        RAY_QUERY_OBJECT_COMMITTED_HIT:
                        mask |= HIPCodegenLLVMImpl::
                            llvm_ray_query_observes_committed_hit;
                        break;
                    case xir::RayQueryObjectReadOp::
                        RAY_QUERY_OBJECT_WORLD_SPACE_RAY:
                        mask |= HIPCodegenLLVMImpl::
                            llvm_ray_query_observes_world_ray;
                        break;
                    case xir::RayQueryObjectReadOp::
                        RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY:
                        mask |= HIPCodegenLLVMImpl::
                            llvm_ray_query_observes_object_ray;
                        break;
                    default: break;
                }
            }
            // Generated handlers have a closed direct-call graph. Following
            // every function operand computes a conservative union: reading a
            // different query may retain an unnecessary export, but can never
            // suppress one required by the active query.
            for (auto operand_use : instruction->operand_uses()) {
                if (auto operand = operand_use->value();
                    operand != nullptr && operand->isa<xir::Function>()) {
                    accumulate_ray_query_handler_observations(
                        static_cast<const xir::Function *>(operand),
                        visited, mask);
                }
            }
        });
}

struct RayQueryHandlerObservationMasks {
    uint32_t surface;
    uint32_t procedural;

    [[nodiscard]] constexpr uint32_t any() const noexcept {
        return surface | procedural;
    }

    [[nodiscard]] constexpr uint32_t encoded() const noexcept {
        return surface |
               (procedural << HIPCodegenLLVMImpl::
                    llvm_ray_query_procedural_observation_shift);
    }
};

static_assert(
    ((HIPCodegenLLVMImpl::llvm_ray_query_handler_observation_mask << HIPCodegenLLVMImpl::llvm_ray_query_procedural_observation_shift) &
     HIPCodegenLLVMImpl::llvm_ray_query_handler_observation_mask) ==
    0u);

[[nodiscard]] uint32_t ray_query_handler_observation_mask(
    const xir::Function *handler) noexcept {
    llvm::DenseSet<const xir::Function *> visited;
    auto mask = 0u;
    accumulate_ray_query_handler_observations(handler, visited, mask);
    return mask;
}

[[nodiscard]] RayQueryHandlerObservationMasks
ray_query_handler_observation_masks(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    return {
        .surface = ray_query_handler_observation_mask(
            pipeline->on_surface_function()),
        .procedural = ray_query_handler_observation_mask(
            pipeline->on_procedural_function())};
}

[[nodiscard]] constexpr bool
ray_query_observation_requires_distinct_ray_states(
    uint32_t mask) noexcept {
    return (mask &
            HIPCodegenLLVMImpl::llvm_ray_query_observes_world_ray) != 0u &&
           (mask &
            HIPCodegenLLVMImpl::llvm_ray_query_observes_object_ray) != 0u;
}

// The compact object-ray quotient represents candidate state plus exactly one
// candidate-dependent ray. A committed-hit or immutable world-ray observation
// therefore requires the ordinary public-state transaction; object-ray-only
// observation does not.
[[nodiscard]] constexpr bool
ray_query_observation_requires_full_state(uint32_t mask) noexcept {
    return (mask &
            (HIPCodegenLLVMImpl::llvm_ray_query_observes_committed_hit |
             HIPCodegenLLVMImpl::llvm_ray_query_observes_world_ray)) != 0u;
}

[[nodiscard]] bool ray_query_value_has_function_local_state(
    const xir::Value *value,
    llvm::DenseSet<const xir::Value *> &active) noexcept {
    if (value == nullptr ||
        (value->type() != Type::of<RayQueryAll>() &&
         value->type() != Type::of<RayQueryAny>())) {
        return false;
    }
    if (value->isa<xir::ResourceQueryInst>()) {
        switch (static_cast<const xir::ResourceQueryInst *>(value)->op()) {
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
                return true;
            default: return false;
        }
    }
    if (value->isa<xir::LoadInst>()) {
        return ray_query_value_has_function_local_state(
            static_cast<const xir::LoadInst *>(value)->variable(), active);
    }
    if (value->isa<xir::PhiInst>()) {
        if (!active.insert(value).second) { return false; }
        auto local = true;
        for (auto operand_use :
             static_cast<const xir::PhiInst *>(value)->operand_uses()) {
            local &= ray_query_value_has_function_local_state(
                operand_use->value(), active);
        }
        active.erase(value);
        return local;
    }
    if (!value->isa<xir::AllocaInst>() ||
        !static_cast<const xir::AllocaInst *>(value)->is_local() ||
        !active.insert(value).second) {
        return false;
    }

    // A local query variable denotes the function's singleton traversal state
    // exactly when every possible whole-object definition is itself local and
    // no other use can rebind or escape the variable. This is a closed use-def
    // proof, independent of block traversal order. Multiple local definitions
    // remain equivalent because all query construction in one function uses
    // the same non-reentrant state allocation. Cycles or an unknown definition
    // fail closed to the encoded query-object ABI.
    auto has_definition = false;
    auto local = true;
    for (auto use : value->use_list()) {
        auto user = use->user();
        if (user == nullptr || !user->isa<xir::Instruction>()) {
            local = false;
            break;
        }
        auto instruction = static_cast<const xir::Instruction *>(user);
        if (instruction->isa<xir::StoreInst>()) {
            auto store = static_cast<const xir::StoreInst *>(instruction);
            if (store->variable() != value) {
                local = false;
                break;
            }
            has_definition = true;
            if (!ray_query_value_has_function_local_state(
                    store->value(), active)) {
                local = false;
                break;
            }
            continue;
        }
        if (instruction->isa<xir::LoadInst>() &&
            static_cast<const xir::LoadInst *>(instruction)->variable() ==
                value) {
            continue;
        }
        if (instruction->isa<xir::RayQueryPipelineInst>() &&
            static_cast<const xir::RayQueryPipelineInst *>(instruction)
                    ->query_object() == value) {
            continue;
        }
        if (instruction->isa<xir::RayQueryDispatchInst>() ||
            instruction->isa<xir::RayQueryObjectReadInst>() ||
            instruction->isa<xir::RayQueryObjectWriteInst>()) {
            continue;
        }
        local = false;
        break;
    }
    active.erase(value);
    return local && has_definition;
}

[[nodiscard]] bool ray_query_value_has_function_local_state(
    const xir::Value *value) noexcept {
    llvm::DenseSet<const xir::Value *> active;
    return ray_query_value_has_function_local_state(value, active);
}

// A synchronous pipeline mutates one query object and then returns to its
// parent function. Its post-state is proven dead under this closed, local use
// criterion:
//
//   1. the object has function-local storage;
//   2. every non-defining use is the pipeline being classified.
//
// Whole-object stores are definitions and cannot observe the previous state.
// Every other use (including a load, read, write, another pipeline, or an
// unknown escape) may observe the post-state and therefore fails closed. The
// proof intentionally does not depend on instruction order: accepting fewer
// handler-only pipelines is safe, while accepting an escaping object is not.
[[nodiscard]] bool ray_query_pipeline_post_state_is_observed(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    auto query_object = pipeline->query_object();
    if (query_object == nullptr || !query_object->isa<xir::AllocaInst>() ||
        !static_cast<const xir::AllocaInst *>(query_object)->is_local()) {
        return true;
    }
    for (auto use : query_object->use_list()) {
        auto user = use->user();
        if (user == pipeline &&
            use == pipeline->operand_use(
                       xir::RayQueryPipelineInst::
                           operand_index_query_object)) {
            continue;
        }
        if (user != nullptr && user->isa<xir::StoreInst>() &&
            static_cast<const xir::StoreInst *>(user)->variable() ==
                query_object) {
            continue;
        }
        return true;
    }
    return false;
}

// A native closest traversal is an order-insensitive reduction only when the
// parent observes no query state other than the final committed hit. Whole-
// object stores are definitions, and a load is accepted only when every use of
// the loaded query is exactly COMMITTED_HIT. Unknown uses fail closed. This is
// deliberately a use-graph property rather than a source-shape match.
[[nodiscard]] bool
ray_query_pipeline_post_state_is_committed_hit_only(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    auto query_object = pipeline->query_object();
    if (query_object == nullptr || !query_object->isa<xir::AllocaInst>() ||
        !static_cast<const xir::AllocaInst *>(query_object)->is_local()) {
        if (query_object != nullptr && query_object->isa<xir::Instruction>()) {
            auto instruction =
                static_cast<const xir::Instruction *>(query_object);
            LUISA_VERBOSE(
                "HIP native closest reduction requires local alloca query "
                "post-state, got '{}' ('{}') in function '{}'.",
                xir::to_string(instruction->derived_instruction_tag()),
                instruction->name().value_or("<unnamed>"),
                instruction->parent_function()->name().value_or("<unnamed>"));
        }
        return false;
    }
    auto observes_committed_hit = false;
    auto is_committed_hit_read = [&](const xir::User *user,
                                     const xir::Use *use) noexcept {
        if (user == nullptr || !user->isa<xir::RayQueryObjectReadInst>()) {
            return false;
        }
        auto read = static_cast<const xir::RayQueryObjectReadInst *>(user);
        if (read->op() != xir::RayQueryObjectReadOp::
                              RAY_QUERY_OBJECT_COMMITTED_HIT ||
            use != read->operand_use(0u)) {
            return false;
        }
        observes_committed_hit = true;
        return true;
    };
    for (auto use : query_object->use_list()) {
        auto user = use->user();
        if (user == pipeline &&
            use == pipeline->operand_use(
                       xir::RayQueryPipelineInst::
                           operand_index_query_object)) {
            continue;
        }
        if (user != nullptr && user->isa<xir::StoreInst>() &&
            static_cast<const xir::StoreInst *>(user)->variable() ==
                query_object) {
            continue;
        }
        if (is_committed_hit_read(user, use)) { continue; }
        if (user != nullptr && user->isa<xir::LoadInst>() &&
            static_cast<const xir::LoadInst *>(user)->variable() ==
                query_object) {
            auto load = static_cast<const xir::LoadInst *>(user);
            auto only_committed_reads = true;
            for (auto load_use : load->use_list()) {
                only_committed_reads &=
                    is_committed_hit_read(load_use->user(), load_use);
            }
            if (only_committed_reads) { continue; }
        }
        if (user != nullptr && user->isa<xir::Instruction>()) {
            auto instruction = static_cast<const xir::Instruction *>(user);
            LUISA_VERBOSE(
                "HIP native closest reduction rejected post-state use '{}' "
                "('{}') in function '{}'.",
                xir::to_string(instruction->derived_instruction_tag()),
                instruction->name().value_or("<unnamed>"),
                instruction->parent_function()->name().value_or("<unnamed>"));
        } else {
            LUISA_VERBOSE(
                "HIP native closest reduction rejected a non-instruction "
                "post-state use of query '{}'.",
                query_object->name().value_or("<unnamed>"));
        }
        return false;
    }
    return observes_committed_hit;
}

struct NativeClosestHandlerAnalysisContext {
    const xir::Function *function;
    luisa::vector<bool> local_reference_arguments;
    luisa::vector<bool> active_query_reference_arguments;
};

[[nodiscard]] size_t function_argument_index(
    const xir::Function *function,
    const xir::Argument *argument) noexcept {
    auto index = 0u;
    for (auto candidate : function->arguments()) {
        if (candidate == argument) { return index; }
        ++index;
    }
    return std::numeric_limits<size_t>::max();
}

// Prove that a store target belongs to the current handler invocation. GEP
// preserves provenance and a PHI is local iff every incoming value is local.
// Cycles fail closed. Reference arguments are local only when the caller proved
// their actual argument local and propagated that fact. CastInst is
// deliberately absent: XIR casts are rvalues, not address-preserving lvalues.
[[nodiscard]] bool native_closest_pointer_is_local(
    const xir::Value *value,
    const NativeClosestHandlerAnalysisContext &context,
    llvm::DenseSet<const xir::Value *> &active) noexcept {
    if (value == nullptr || !active.insert(value).second) { return false; }
    auto finish = [&](bool result) noexcept {
        active.erase(value);
        return result;
    };
    if (value->isa<xir::AllocaInst>()) {
        auto alloca = static_cast<const xir::AllocaInst *>(value);
        return finish(alloca->is_local() &&
                      alloca->parent_function() == context.function);
    }
    if (value->isa<xir::Argument>()) {
        auto argument = static_cast<const xir::Argument *>(value);
        auto index = function_argument_index(context.function, argument);
        return finish(
            index < context.local_reference_arguments.size() &&
            context.local_reference_arguments[index]);
    }
    if (value->isa<xir::GEPInst>()) {
        return finish(native_closest_pointer_is_local(
            static_cast<const xir::GEPInst *>(value)->base(),
            context, active));
    }
    if (value->isa<xir::PhiInst>()) {
        auto phi = static_cast<const xir::PhiInst *>(value);
        auto local = phi->incoming_count() != 0u;
        for (auto i = 0u; i < phi->incoming_count() && local; ++i) {
            local &= native_closest_pointer_is_local(
                phi->incoming(i).value, context, active);
        }
        return finish(local);
    }
    return finish(false);
}

// The native callback may mutate exactly the query whose candidate it is
// evaluating. Query identity is a reference-provenance property, propagated
// through direct Callable arguments. It is intentionally distinct from
// locality: another local RayQuery is still not the active traversal.
[[nodiscard]] bool native_closest_pointer_is_active_query(
    const xir::Value *value,
    const NativeClosestHandlerAnalysisContext &context) noexcept {
    if (value == nullptr || !value->isa<xir::Argument>()) { return false; }
    auto argument = static_cast<const xir::Argument *>(value);
    auto index = function_argument_index(context.function, argument);
    return index < context.active_query_reference_arguments.size() &&
           context.active_query_reference_arguments[index];
}

[[nodiscard]] bool native_closest_pointer_is_local(
    const xir::Value *value,
    const NativeClosestHandlerAnalysisContext &context) noexcept {
    llvm::DenseSet<const xir::Value *> active;
    return native_closest_pointer_is_local(value, context, active);
}

[[nodiscard]] bool resource_query_is_read_only_for_native_closest(
    xir::ResourceQueryOp op) noexcept {
    switch (op) {
        case xir::ResourceQueryOp::BUFFER_SIZE:
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE:
        case xir::ResourceQueryOp::TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER:
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER:
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS:
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX:
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT:
            return true;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
            return false;
    }
    // Deliberately fail closed if ResourceQueryOp gains a new enumerator.
    return false;
}

// The native closest route is a reduction over candidate acceptance. Every
// reachable handler operation must therefore be read-only outside the
// invocation, with the sole exceptions of COMMIT_TRIANGLE and
// COMMIT_PROCEDURAL on the active query. Local scratch stores are permitted;
// resource/atomic writes, explicit termination, nested traversal, and unknown
// calls are rejected. The analysis is context-sensitive for reference
// arguments so helpers may mutate caller-local scratch without making the
// transaction externally effectful.
//
// HIPRT's closest traversal may batch/reorder the two members of a triangle
// packet, so a surface handler may not observe either Ray value: both contain
// the mutable t_max frontier even though their origins/directions use different
// coordinate spaces.
// A procedural leaf has one custom candidate. For those leaves, the closest
// and resumable-any-hit templates execute the same SceneTraversal frontier:
// rejection advances the same leaf state, while acceptance changes the next
// active max from m to t by `ray.maxT = t` (closest) or
// `contractRayMaxT(t)` (resumable). Induction over that shared frontier proves
// equal world-ray state at every procedural callback. This is the exact case
// required by curve/ribbon intersection and does not generalize to surface
// candidates.
[[nodiscard]] bool native_closest_handler_is_reduction(
    const xir::Function *function,
    luisa::vector<bool> local_reference_arguments,
    luisa::vector<bool> active_query_reference_arguments,
    bool allow_mutable_ray_state,
    llvm::DenseSet<const xir::Function *> &active_functions) noexcept {
    if (function == nullptr || function->definition() == nullptr ||
        !active_functions.insert(function).second) {
        return false;
    }
    NativeClosestHandlerAnalysisContext context{
        function, std::move(local_reference_arguments),
        std::move(active_query_reference_arguments)};
    auto valid = true;
    function->definition()->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            if (!valid) { return; }
            const auto was_valid = valid;
            switch (instruction->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::STORE: {
                    auto store = static_cast<const xir::StoreInst *>(instruction);
                    // A literal load/store round trip through the same pointer
                    // is observationally the identity even for an external
                    // reference. This pattern is emitted for untouched outlined
                    // PHI state and carries no candidate-order information.
                    auto value = store->value();
                    if (value != nullptr && value->isa<xir::LoadInst>() &&
                        static_cast<const xir::LoadInst *>(value)->variable() ==
                            store->variable()) {
                        break;
                    }
                    valid = native_closest_pointer_is_local(
                        store->variable(), context);
                    break;
                }
                case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                    valid = resource_query_is_read_only_for_native_closest(
                        static_cast<const xir::ResourceQueryInst *>(instruction)
                            ->op());
                    break;
                }
                case xir::DerivedInstructionTag::RESOURCE_WRITE:
                case xir::DerivedInstructionTag::ATOMIC:
                case xir::DerivedInstructionTag::THREAD_GROUP:
                case xir::DerivedInstructionTag::PRINT:
                case xir::DerivedInstructionTag::CLOCK:
                case xir::DerivedInstructionTag::DEBUG_BREAK:
                case xir::DerivedInstructionTag::ASSERT:
                case xir::DerivedInstructionTag::RASTER_DISCARD:
                case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
                case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC:
                case xir::DerivedInstructionTag::CORO_SUSPEND:
                case xir::DerivedInstructionTag::CORO_RESUME:
                case xir::DerivedInstructionTag::CORO_TERMINATE:
                case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE:
                    valid = false;
                    break;
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: {
                    auto read = static_cast<
                        const xir::RayQueryObjectReadInst *>(instruction);
                    // COMMITTED_HIT observes reduction state before the
                    // reduction is complete. Both ray representations contain
                    // the mutable t_max frontier and can make the acceptance
                    // predicate enumeration-order dependent. These reads
                    // therefore require the exact resumable transaction unless
                    // the procedural-frontier proof applies.
                    valid = read->operand_count() != 0u &&
                            native_closest_pointer_is_active_query(
                                read->operand(0u), context) &&
                            read->op() != xir::RayQueryObjectReadOp::
                                              RAY_QUERY_OBJECT_COMMITTED_HIT &&
                            (allow_mutable_ray_state ||
                             (read->op() != xir::RayQueryObjectReadOp::
                                                RAY_QUERY_OBJECT_WORLD_SPACE_RAY &&
                              read->op() != xir::RayQueryObjectReadOp::
                                                RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY));
                    break;
                }
                case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: {
                    auto write = static_cast<
                        const xir::RayQueryObjectWriteInst *>(instruction);
                    valid = write->operand_count() != 0u &&
                            native_closest_pointer_is_active_query(
                                write->operand(0u), context) &&
                            (write->op() == xir::RayQueryObjectWriteOp::
                                                RAY_QUERY_OBJECT_COMMIT_TRIANGLE ||
                             write->op() == xir::RayQueryObjectWriteOp::
                                                RAY_QUERY_OBJECT_COMMIT_PROCEDURAL);
                    break;
                }
                case xir::DerivedInstructionTag::CALL: {
                    auto call = static_cast<const xir::CallInst *>(instruction);
                    auto callee = call->callee();
                    if (callee == nullptr || callee->definition() == nullptr ||
                        call->argument_count() !=
                            callee->arguments().count_size()) {
                        valid = false;
                        break;
                    }
                    luisa::vector<bool> callee_local_arguments(
                        call->argument_count(), false);
                    luisa::vector<bool> callee_active_query_arguments(
                        call->argument_count(), false);
                    auto i = 0u;
                    for (auto argument : callee->arguments()) {
                        if (argument->is_reference()) {
                            callee_local_arguments[i] =
                                native_closest_pointer_is_local(
                                    call->argument(i), context);
                            callee_active_query_arguments[i] =
                                native_closest_pointer_is_active_query(
                                    call->argument(i), context);
                        }
                        ++i;
                    }
                    valid = native_closest_handler_is_reduction(
                        callee, std::move(callee_local_arguments),
                        std::move(callee_active_query_arguments),
                        allow_mutable_ray_state,
                        active_functions);
                    break;
                }
                default: break;
            }
            if (was_valid && !valid) {
                LUISA_VERBOSE(
                    "HIP native closest reduction rejected handler "
                    "instruction '{}' ('{}') in function '{}'.",
                    xir::to_string(
                        instruction->derived_instruction_tag()),
                    instruction->name().value_or("<unnamed>"),
                    function->name().value_or("<unnamed>"));
            }
        });
    active_functions.erase(function);
    return valid;
}

[[nodiscard]] bool native_closest_handler_is_reduction(
    const xir::Function *function,
    bool allow_mutable_ray_state) noexcept {
    llvm::DenseSet<const xir::Function *> active_functions;
    auto argument_count = function == nullptr ? 0u :
                                                function->arguments().count_size();
    if (function == nullptr || argument_count == 0u) { return false; }
    auto query_argument = function->arguments().front();
    if (!query_argument->is_reference() ||
        query_argument->type() != Type::of<RayQueryAll>()) {
        return false;
    }
    luisa::vector<bool> active_query_arguments(argument_count, false);
    active_query_arguments.front() = true;
    return native_closest_handler_is_reduction(
        function, luisa::vector<bool>(argument_count, false),
        std::move(active_query_arguments),
        allow_mutable_ray_state,
        active_functions);
}

[[nodiscard]] bool ray_query_pipeline_admits_native_closest_reduction(
    const xir::RayQueryPipelineInst *pipeline) noexcept {
    if (pipeline == nullptr || pipeline->query_object() == nullptr ||
        pipeline->query_object()->type() != Type::of<RayQueryAll>()) {
        return false;
    }
    const auto committed_post_state_only =
        ray_query_pipeline_post_state_is_committed_hit_only(pipeline);
    const auto surface_is_reduction = native_closest_handler_is_reduction(
        pipeline->on_surface_function(), false);
    const auto procedural_is_reduction = native_closest_handler_is_reduction(
        pipeline->on_procedural_function(), true);
    const auto observation_masks =
        ray_query_handler_observation_masks(pipeline);
    const auto observes_both_ray_spaces =
        ray_query_observation_requires_distinct_ray_states(
            observation_masks.surface) ||
        ray_query_observation_requires_distinct_ray_states(
            observation_masks.procedural);
    LUISA_VERBOSE(
        "HIP native closest reduction proof: committed-post-state-only = "
        "{}, surface = {} (observations=0x{:x}), procedural = {} "
        "(observations=0x{:x}), joint-ray-handler = {}.",
        committed_post_state_only, surface_is_reduction,
        observation_masks.surface, procedural_is_reduction,
        observation_masks.procedural, observes_both_ray_spaces);
    if (std::getenv("LUISA_HIP_DUMP_NATIVE_CLOSEST_PROOF") != nullptr &&
        !(committed_post_state_only && surface_is_reduction &&
          procedural_is_reduction && !observes_both_ray_spaces)) {
        luisa::string dump;
        auto &printer = xir::XIRDebugPrinter::global();
        printer.emit_function(dump, pipeline->parent_function());
        printer.emit_function(dump, pipeline->on_surface_function());
        printer.emit_function(dump, pipeline->on_procedural_function());
        LUISA_INFO("HIP native closest rejected XIR:\n{}", dump);
    }
    return committed_post_state_only && surface_is_reduction &&
           procedural_is_reduction && !observes_both_ray_spaces;
}

}// namespace

HIPCodegenLLVMImpl::RayQueryPipelineProjectionInfo
HIPCodegenLLVMImpl::_finalize_ray_query_pipeline_contexts() noexcept {
    RayQueryPipelineProjectionInfo projection;
    size_t projected_argument_count = 0u;
    size_t projected_aggregate_leaf_count = 0u;
    size_t separated_query_argument_count = 0u;
    size_t scalarized_context_count = 0u;
    size_t original_context_bytes = 0u;
    size_t projected_context_bytes = 0u;

    // Compute the least fixed point of interprocedural primitive-leaf demand
    // over the local generated-Callable graph. For every formal aggregate A,
    // flatten(A) is the finite set of paths to non-aggregate leaves. A pure
    // extractvalue projects that set, while forwarding a projected value from
    // caller leaf (f, i, p) to local callee leaf (g, j, q) contributes
    //
    //   live(f, i, p.q) |= live(g, j, q).
    //
    // Every other use observes the complete currently projected subtree and
    // seeds its leaves. The resulting monotone Boolean equations are solved
    // from BOTTOM, so forwarding-only SCCs remain dead and one observation
    // propagates through the complete call cycle. Unknown, external, typed-
    // attribute, or structurally mismatched uses fail closed to the whole
    // subtree. This is the aggregate analogue of the former whole-argument
    // analysis; it removes descriptor sizes only when their absence is proved
    // over the complete direct-call graph.
    using AggregateLeafPath = llvm::SmallVector<unsigned, 4>;
    struct ArgumentLeafInfo {
        const llvm::Argument *argument;
        luisa::vector<AggregateLeafPath> paths;
        size_t global_offset;
    };
    luisa::vector<ArgumentLeafInfo> argument_infos;
    llvm::DenseMap<const llvm::Argument *, size_t> argument_info_indices;
    auto flatten_type = [&](auto &&self, llvm::Type *type,
                            AggregateLeafPath &path,
                            luisa::vector<AggregateLeafPath> &paths) noexcept
        -> void {
        if (auto structure = llvm::dyn_cast<llvm::StructType>(type);
            structure != nullptr && !structure->isOpaque()) {
            for (auto i = 0u; i < structure->getNumElements(); ++i) {
                path.emplace_back(i);
                self(self, structure->getElementType(i), path, paths);
                path.pop_back();
            }
            return;
        }
        if (auto array = llvm::dyn_cast<llvm::ArrayType>(type)) {
            for (auto i = 0u; i < array->getNumElements(); ++i) {
                path.emplace_back(i);
                self(self, array->getElementType(), path, paths);
                path.pop_back();
            }
            return;
        }
        // Fixed vectors are one target register value in this analysis. LLVM
        // extractelement is not an aggregate projection and therefore makes
        // the complete vector leaf observable below.
        paths.emplace_back(path);
    };
    size_t argument_leaf_count = 0u;
    for (auto &function : *_llvm_module) {
        if (function.isDeclaration()) { continue; }
        for (auto &argument : function.args()) {
            auto index = argument_infos.size();
            auto &info = argument_infos.emplace_back(
                ArgumentLeafInfo{.argument = &argument,
                                 .global_offset = argument_leaf_count});
            auto path = AggregateLeafPath{};
            flatten_type(flatten_type, argument.getType(), path, info.paths);
            argument_leaf_count += info.paths.size();
            argument_info_indices.try_emplace(&argument, index);
        }
    }
    luisa::vector<luisa::vector<size_t>> reverse_dependencies(
        argument_leaf_count);
    luisa::vector<bool> live_leaves(argument_leaf_count, false);
    luisa::vector<size_t> live_worklist;
    live_worklist.reserve(argument_leaf_count);
    auto path_has_prefix = [](const AggregateLeafPath &path,
                              const AggregateLeafPath &prefix) noexcept {
        return prefix.size() <= path.size() &&
               std::equal(prefix.begin(), prefix.end(), path.begin());
    };
    auto find_leaf = [](const ArgumentLeafInfo &info,
                        const AggregateLeafPath &path) noexcept {
        for (auto i = 0u; i < info.paths.size(); ++i) {
            if (info.paths[i] == path) {
                return static_cast<size_t>(i);
            }
        }
        return std::numeric_limits<size_t>::max();
    };
    auto seed_subtree = [&](const ArgumentLeafInfo &root,
                            const AggregateLeafPath &prefix) noexcept {
        for (auto i = 0u; i < root.paths.size(); ++i) {
            if (!path_has_prefix(root.paths[i], prefix)) { continue; }
            auto global_index = root.global_offset + i;
            if (!live_leaves[global_index]) {
                live_leaves[global_index] = true;
                live_worklist.emplace_back(global_index);
            }
        }
    };
    for (auto root_index = 0u;
         root_index < argument_infos.size(); ++root_index) {
        auto &root = argument_infos[root_index];
        if (root.argument->getParent()->getAttributes().hasParamAttrs(
                root.argument->getArgNo())) {
            seed_subtree(root, {});
            continue;
        }
        llvm::SmallPtrSet<llvm::Value *, 16> active_values;
        auto collect_uses = [&](auto &&self, llvm::Value *value,
                                AggregateLeafPath prefix) noexcept -> void {
            if (value->use_empty()) { return; }
            if (!active_values.insert(value).second) {
                // A cyclic derived-value graph requires a richer value-PHI
                // equation. Such graphs are not descriptor projections; fail
                // closed instead of assuming the cycle is dead.
                seed_subtree(root, prefix);
                return;
            }
            for (auto &use : value->uses()) {
                auto user = use.getUser();
                if (auto extract = llvm::dyn_cast<llvm::ExtractValueInst>(user)) {
                    if (extract->getAggregateOperand() != value) {
                        seed_subtree(root, prefix);
                        continue;
                    }
                    if (extract->use_empty()) {
                        // extractvalue is pure. A dead descriptor-size extract
                        // is not an observation even before ordinary DCE.
                        continue;
                    }
                    auto nested = prefix;
                    nested.insert(nested.end(), extract->idx_begin(),
                                  extract->idx_end());
                    self(self, extract, std::move(nested));
                    continue;
                }
                if (auto freeze = llvm::dyn_cast<llvm::FreezeInst>(user)) {
                    self(self, freeze, prefix);
                    continue;
                }
                if (auto phi = llvm::dyn_cast<llvm::PHINode>(user)) {
                    self(self, phi, prefix);
                    continue;
                }
                if (auto select = llvm::dyn_cast<llvm::SelectInst>(user)) {
                    if (select->getTrueValue() == value ||
                        select->getFalseValue() == value) {
                        self(self, select, prefix);
                        continue;
                    }
                }
                auto call = llvm::dyn_cast<llvm::CallBase>(user);
                if (call != nullptr && call->isArgOperand(&use)) {
                    auto callee = call->getCalledFunction();
                    auto callee_argument_index =
                        call->getArgOperandNo(&use);
                    if (call->getAttributes().hasParamAttrs(
                            callee_argument_index) ||
                        callee == nullptr || callee->isDeclaration() ||
                        !callee->hasLocalLinkage() ||
                        callee_argument_index >= callee->arg_size()) {
                        seed_subtree(root, prefix);
                        continue;
                    }
                    auto callee_argument =
                        callee->getArg(callee_argument_index);
                    auto callee_info_iter =
                        argument_info_indices.find(callee_argument);
                    if (callee_info_iter == argument_info_indices.end() ||
                        callee_argument->getType() != value->getType()) {
                        seed_subtree(root, prefix);
                        continue;
                    }
                    auto &callee_info =
                        argument_infos[callee_info_iter->second];
                    auto mapped_all_leaves = true;
                    for (auto callee_leaf = 0u;
                         callee_leaf < callee_info.paths.size();
                         ++callee_leaf) {
                        auto caller_path = prefix;
                        caller_path.insert(
                            caller_path.end(),
                            callee_info.paths[callee_leaf].begin(),
                            callee_info.paths[callee_leaf].end());
                        auto caller_leaf = find_leaf(root, caller_path);
                        if (caller_leaf ==
                            std::numeric_limits<size_t>::max()) {
                            mapped_all_leaves = false;
                            break;
                        }
                        reverse_dependencies[callee_info.global_offset + callee_leaf]
                            .emplace_back(root.global_offset + caller_leaf);
                    }
                    if (!mapped_all_leaves) {
                        seed_subtree(root, prefix);
                    }
                    continue;
                }
                seed_subtree(root, prefix);
            }
            active_values.erase(value);
        };
        collect_uses(collect_uses,
                     const_cast<llvm::Argument *>(root.argument), {});
    }
    for (auto cursor = 0u; cursor < live_worklist.size(); ++cursor) {
        auto live_index = live_worklist[cursor];
        for (auto dependent : reverse_dependencies[live_index]) {
            if (!live_leaves[dependent]) {
                live_leaves[dependent] = true;
                live_worklist.emplace_back(dependent);
            }
        }
    }
    auto argument_live_leaf_paths = [&](const llvm::Argument *argument) noexcept {
        auto iter = argument_info_indices.find(argument);
        LUISA_ASSERT(
            iter != argument_info_indices.end(),
            "Missing HIP generated-Callable argument demand state.");
        auto &info = argument_infos[iter->second];
        auto paths = luisa::vector<AggregateLeafPath>{};
        for (auto i = 0u; i < info.paths.size(); ++i) {
            if (live_leaves[info.global_offset + i]) {
                paths.emplace_back(info.paths[i]);
            }
        }
        return paths;
    };

    for (auto &context : _llvm_ray_query_pipeline_contexts) {
        if (context.distinct_ray_states_required) {
            auto &domains = projection.exact_state_required_functions;
            if (std::find(domains.begin(), domains.end(),
                          context.parent_function) == domains.end()) {
                domains.emplace_back(context.parent_function);
            }
        }
        auto argument_count = context.stores.size();
        LUISA_ASSERT(
            argument_count != 0u &&
                context.loads.size() == argument_count &&
                context.compact_loads.size() == argument_count &&
                context.compact_object_ray_loads.size() == argument_count &&
                context.on_surface->arg_size() == argument_count &&
                context.on_procedural->arg_size() == argument_count,
            "Malformed HIP synchronous ray-query callback environment.");
        LUISA_ASSERT(
            !context.on_surface->isDeclaration() &&
                !context.on_procedural->isDeclaration(),
            "HIP ray-query callback environment projection requires "
            "translated candidate handlers.");

        // Let A_i be callback ABI argument i. A_0 is the query reference: it is
        // intrinsic traversal identity and reaches the dispatcher through its
        // dedicated argument, never through user capture storage. For i > 0,
        // the environment stores A_i only when either handler demands its
        // corresponding formal argument. If both demand bits are false,
        // replacing both call operands with poison is semantics-preserving
        // under the fixed-point equations above. Taking the union is necessary
        // because candidate kind is selected dynamically inside traversal.
        struct RetainedComponent {
            uint32_t argument_index;
            bool whole_argument;
            AggregateLeafPath path;
            llvm::Type *type;
        };
        llvm::SmallVector<RetainedComponent, 16> retained_components;
        luisa::vector<luisa::vector<uint32_t>> component_indices(
            argument_count);
        for (auto i = 1u; i < argument_count; ++i) {
            auto surface_arg = context.on_surface->getArg(i);
            auto procedural_arg = context.on_procedural->getArg(i);
            LUISA_ASSERT(
                surface_arg->getType() == procedural_arg->getType(),
                "HIP RayQuery handler argument type mismatch.");
            auto surface_paths =
                argument_live_leaf_paths(surface_arg);
            auto procedural_paths =
                argument_live_leaf_paths(procedural_arg);
            for (auto &path : procedural_paths) {
                if (std::find(surface_paths.begin(), surface_paths.end(),
                              path) == surface_paths.end()) {
                    surface_paths.emplace_back(path);
                }
            }
            std::sort(surface_paths.begin(), surface_paths.end());
            auto all_paths = argument_infos[argument_info_indices.find(surface_arg)->second]
                                 .paths;
            if (surface_paths.empty()) {
                projected_argument_count++;
                projected_aggregate_leaf_count += all_paths.size();
                continue;
            }
            auto *argument_type = surface_arg->getType();
            if (!argument_type->isAggregateType() ||
                surface_paths.size() == all_paths.size()) {
                auto component_index = static_cast<uint32_t>(
                    retained_components.size());
                retained_components.emplace_back(RetainedComponent{
                    .argument_index = static_cast<uint32_t>(i),
                    .whole_argument = true,
                    .type = argument_type});
                component_indices[i].emplace_back(component_index);
                continue;
            }
            projected_aggregate_leaf_count +=
                all_paths.size() - surface_paths.size();
            for (auto &path : surface_paths) {
                auto *leaf_type = llvm::ExtractValueInst::getIndexedType(
                    argument_type, path);
                LUISA_ASSERT(
                    leaf_type != nullptr && !leaf_type->isAggregateType(),
                    "HIP RayQuery callback leaf projection produced an "
                    "invalid aggregate path.");
                auto component_index = static_cast<uint32_t>(
                    retained_components.size());
                retained_components.emplace_back(RetainedComponent{
                    .argument_index = static_cast<uint32_t>(i),
                    .whole_argument = false,
                    .path = path,
                    .type = leaf_type});
                component_indices[i].emplace_back(component_index);
            }
        }

        auto original_type = llvm::cast<llvm::StructType>(
            context.storage->getAllocatedType());
        auto original_bytes =
            _data_layout->getTypeAllocSize(original_type).getFixedValue();
        original_context_bytes += original_bytes;

        auto for_each_context_load = [&](auto i, auto &&visitor) noexcept {
            visitor(context.loads[i]);
            if (auto compact_load = context.compact_loads[i]) {
                visitor(compact_load);
            }
            if (auto compact_object_ray_load =
                    context.compact_object_ray_loads[i]) {
                visitor(compact_object_ray_load);
            }
        };
        auto erase_original_field = [&](auto i) noexcept {
            auto old_store = context.stores[i];
            auto old_store_gep = llvm::cast<llvm::GetElementPtrInst>(
                old_store->getPointerOperand());
            old_store->eraseFromParent();
            for_each_context_load(i, [&](auto old_load) noexcept {
                auto old_load_gep = llvm::cast<llvm::GetElementPtrInst>(
                    old_load->getPointerOperand());
                old_load->eraseFromParent();
                LUISA_ASSERT(
                    old_load_gep->use_empty(),
                    "HIP ray-query callback environment address escaped.");
                old_load_gep->eraseFromParent();
            });
            LUISA_ASSERT(old_store_gep->use_empty(),
                         "HIP ray-query callback environment address escaped.");
            old_store_gep->eraseFromParent();
        };
        auto erase_original_storage = [&]() noexcept {
            if (context.generic_storage != context.storage &&
                context.generic_storage->use_empty()) {
                llvm::cast<llvm::Instruction>(context.generic_storage)
                    ->eraseFromParent();
            }
            LUISA_ASSERT(
                context.storage->use_empty(),
                "HIP ray-query callback environment storage escaped.");
            context.storage->eraseFromParent();
        };

        auto dispatch_query =
            _llvm_ray_query_pipeline_dispatch->getArg(0u);
        auto dispatch_context =
            _llvm_ray_query_pipeline_dispatch->getArg(1u);
        LUISA_ASSERT(
            context.stores[0u]->getValueOperand()->getType() ==
                    dispatch_query->getType() &&
                context.loads[0u]->getType() == dispatch_query->getType() &&
                context.on_surface->getArg(0u)->getType() ==
                    dispatch_query->getType() &&
                context.on_procedural->getArg(0u)->getType() ==
                    dispatch_query->getType(),
            "HIP RayQuery callback query-reference ABI mismatch.");
        context.loads[0u]->replaceAllUsesWith(dispatch_query);
        separated_query_argument_count++;

        // An empty user environment is represented by null. The traversal
        // transports this value opaquely and the dispatcher has no remaining
        // load from it, so no zero-sized object or dummy capture is required.
        if (retained_components.empty()) {
            context.trace_call->setArgOperand(
                1u, llvm::ConstantPointerNull::get(
                        llvm::cast<llvm::PointerType>(
                            context.trace_call->getArgOperand(1u)->getType())));
            for (auto i = 0u; i < argument_count; ++i) {
                if (i != 0u) {
                    for_each_context_load(i, [](auto old_load) noexcept {
                        old_load->replaceAllUsesWith(
                            llvm::PoisonValue::get(old_load->getType()));
                    });
                }
                erase_original_field(i);
            }
            erase_original_storage();
            continue;
        }

        // The native traversal treats callback_context as an opaque value and
        // returns it unchanged to the generated dispatcher. If the projected
        // product consists of one generic pointer, use that pointer itself as
        // the context. This is the exact one-field-product isomorphism
        //   {p : ptr} stored behind &env  <->  p
        // and eliminates both private storage and its load without merging the
        // lifetime of p with any other captured object. A non-pointer scalar is
        // deliberately not encoded into a pointer: that would invent address
        // semantics and would not be representation-preserving.
        if (retained_components.size() == 1u) {
            auto &component = retained_components.front();
            auto retained_index = component.argument_index;
            auto retained_value =
                context.stores[retained_index]->getValueOperand();
            if (!component.whole_argument) {
                IB extract_b{context.stores[retained_index]};
                retained_value = extract_b.CreateExtractValue(
                    retained_value, component.path,
                    "ray.query.context.scalar.leaf");
            }
            if (retained_value->getType()->isPointerTy() &&
                retained_value->getType()->getPointerAddressSpace() == 0u) {
                LUISA_ASSERT(
                    dispatch_context->getType() == retained_value->getType() &&
                        context.loads[retained_index]->getType() ==
                            context.stores[retained_index]
                                ->getValueOperand()
                                ->getType(),
                    "HIP scalar callback context pointer type mismatch.");
                context.trace_call->setArgOperand(1u, retained_value);
                for (auto i = 0u; i < argument_count; ++i) {
                    if (i == retained_index) {
                        for_each_context_load(i, [&](auto old_load) noexcept {
                            auto old_load_gep = llvm::cast<
                                llvm::GetElementPtrInst>(
                                old_load->getPointerOperand());
                            auto replacement = static_cast<llvm::Value *>(
                                old_load_gep->getPointerOperand());
                            if (!component.whole_argument) {
                                IB rebuild_b{old_load};
                                replacement = rebuild_b.CreateInsertValue(
                                    llvm::PoisonValue::get(
                                        old_load->getType()),
                                    replacement, component.path,
                                    "ray.query.context.scalar.rebuild");
                            }
                            old_load->replaceAllUsesWith(replacement);
                        });
                    } else if (i != 0u) {
                        for_each_context_load(i, [](auto old_load) noexcept {
                            old_load->replaceAllUsesWith(
                                llvm::PoisonValue::get(old_load->getType()));
                        });
                    }
                    erase_original_field(i);
                }
                erase_original_storage();
                scalarized_context_count++;
                continue;
            }
        }

        llvm::SmallVector<llvm::Type *, 16> retained_types;
        retained_types.reserve(retained_components.size());
        for (auto &component : retained_components) {
            auto original_index = component.argument_index;
            auto original_type = context.stores[original_index]
                                     ->getValueOperand()
                                     ->getType();
            LUISA_ASSERT(
                original_type == context.loads[original_index]->getType() &&
                    original_type == context.on_surface->getArg(original_index)->getType() &&
                    original_type == context.on_procedural->getArg(original_index)->getType(),
                "HIP ray-query callback environment argument type mismatch.");
            retained_types.emplace_back(component.type);
        }

        auto projected_type = llvm::StructType::get(
            _llvm_context, retained_types, false);
        auto current_projected_context_bytes =
            _data_layout->getTypeAllocSize(projected_type).getFixedValue();
        projected_context_bytes += current_projected_context_bytes;
        projection.maximum_context_bytes = std::max(
            projection.maximum_context_bytes,
            current_projected_context_bytes);
        // A large synchronous environment is admissible only when both sides
        // of the callback boundary project to compact products:
        //
        //   parent result  = handler side effects (post-state is dead), and
        //   candidate input = {kind, instance, primitive, bary/t}.
        //
        // If the parent reads the final query state, or a handler reads the
        // committed hit/world ray, the exact query transaction remains live.
        // In either case the environment is reloaded across that exact hot
        // boundary and remains subject to the ordinary native-size budget.
        if (!context.native_closest_reduction &&
            (context.post_state_observed ||
             context.full_candidate_state_observed)) {
            projection.maximum_budget_constrained_context_bytes = std::max(
                projection.maximum_budget_constrained_context_bytes,
                current_projected_context_bytes);
            if (!hip_synchronous_ray_query_environment_is_profitable(
                    current_projected_context_bytes)) {
                auto &domains = projection
                                    .oversized_budget_constrained_state_functions;
                if (std::find(
                        domains.begin(), domains.end(),
                        context.parent_function) == domains.end()) {
                    domains.emplace_back(context.parent_function);
                }
            }
        } else if (current_projected_context_bytes >
                   hip_synchronous_ray_query_environment_budget) {
            projection.oversized_compact_handler_only_pipeline_count++;
        }
        IB alloca_b{context.storage};
        auto projected_storage = alloca_b.CreateAlloca(
            projected_type, nullptr,
            context.storage->getName() + ".projected");
        projected_storage->setAlignment(
            _data_layout->getABITypeAlign(projected_type));

        IB trace_b{context.trace_call};
        llvm::Value *projected_generic_storage = projected_storage;
        if (projected_storage->getType()->getPointerAddressSpace() != 0u) {
            projected_generic_storage = trace_b.CreateAddrSpaceCast(
                projected_storage, trace_b.getPtrTy(0),
                "ray.query.context.projected.generic");
        }
        context.trace_call->setArgOperand(
            1u, projected_generic_storage);

        for (auto i = 0u; i < argument_count; ++i) {
            auto old_store = context.stores[i];
            if (i == 0u) {
                // The query-reference load was replaced by dispatch_query
                // above; only its obsolete environment field remains.
            } else if (!component_indices[i].empty()) {
                for (auto component_index : component_indices[i]) {
                    auto &component =
                        retained_components[component_index];
                    auto value = old_store->getValueOperand();
                    IB store_b{old_store};
                    if (!component.whole_argument) {
                        value = store_b.CreateExtractValue(
                            value, component.path,
                            "ray.query.context.projected.leaf");
                    }
                    LUISA_ASSERT(
                        value->getType() == component.type,
                        "HIP RayQuery projected producer leaf type "
                        "mismatch.");
                    auto projected_store_gep = store_b.CreateStructGEP(
                        projected_type, projected_storage,
                        component_index,
                        "ray.query.context.projected.field");
                    auto projected_store = store_b.CreateStore(
                        value, projected_store_gep);
                    projected_store->setAlignment(
                        _data_layout->getABITypeAlign(component.type));
                }
                for_each_context_load(i, [&](auto old_load) noexcept {
                    auto old_load_gep = llvm::cast<
                        llvm::GetElementPtrInst>(
                        old_load->getPointerOperand());
                    auto old_context = old_load_gep->getPointerOperand();
                    IB load_b{old_load};
                    llvm::Value *reconstructed =
                        llvm::PoisonValue::get(old_load->getType());
                    for (auto component_index : component_indices[i]) {
                        auto &component =
                            retained_components[component_index];
                        auto projected_load_gep =
                            load_b.CreateStructGEP(
                                projected_type, old_context,
                                component_index,
                                "ray.query.context.projected.field");
                        auto projected_load = load_b.CreateLoad(
                            component.type, projected_load_gep,
                            "ray.query.context.projected.value");
                        projected_load->setAlignment(
                            _data_layout->getABITypeAlign(component.type));
                        reconstructed = component.whole_argument ?
                                            static_cast<llvm::Value *>(
                                                projected_load) :
                                            load_b.CreateInsertValue(
                                                reconstructed,
                                                projected_load,
                                                component.path,
                                                "ray.query.context.rebuild");
                    }
                    old_load->replaceAllUsesWith(reconstructed);
                });
            } else {
                for_each_context_load(i, [](auto old_load) noexcept {
                    old_load->replaceAllUsesWith(
                        llvm::PoisonValue::get(old_load->getType()));
                });
            }
            erase_original_field(i);
        }
        erase_original_storage();
    }

    if (projected_argument_count != 0u ||
        projected_aggregate_leaf_count != 0u ||
        separated_query_argument_count != 0u ||
        scalarized_context_count != 0u) {
        LUISA_VERBOSE(
            "Separated {} HIP RayQuery identity argument(s), projected {} "
            "unused callback ABI argument(s) and {} unused aggregate "
            "leaf/leaves, and scalarized {} "
            "one-pointer environment(s), "
            "shrinking static environments from {} to {} bytes.",
            separated_query_argument_count,
            projected_argument_count,
            projected_aggregate_leaf_count,
            scalarized_context_count,
            original_context_bytes,
            projected_context_bytes);
    }
    _llvm_ray_query_pipeline_contexts.clear();
    return projection;
}

void HIPCodegenLLVMImpl::_translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept {
    b.GetInsertBlock()->setName("ray.query.loop");
    auto llvm_dispatch_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->dispatch_block());
    llvm_dispatch_block->setName("ray.query.dispatch");
    b.CreateBr(llvm_dispatch_block);
}

void HIPCodegenLLVMImpl::_translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept {
    // luisa.ray.query.proceed();
    // switch (luisa.ray.query.state()) {
    //    case surface: br surface_block
    //    case procedural: br procedural_block
    //    default: br exit_block
    // }
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->query_object());
    auto llvm_state = _advance_ray_query(b, llvm_state_ptr);
    auto llvm_exit_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->exit_block());
    llvm_exit_block->setName("ray.query.exit");
    auto llvm_surface_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_surface_candidate_block());
    llvm_surface_block->setName("ray.query.on.surface.candidate");
    auto llvm_procedural_block = func_ctx.get_local_value<llvm::BasicBlock>(inst->on_procedural_candidate_block());
    llvm_procedural_block->setName("ray.query.on.procedural.candidate");
    auto llvm_dispatch = b.CreateSwitch(llvm_state, llvm_exit_block, 2);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_surface_candidate), llvm_surface_block);
    llvm_dispatch->addCase(b.getInt8(llvm_ray_query_state_procedural_candidate), llvm_procedural_block);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1);
    auto op = inst->op();
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->operand(0));

    switch (op) {
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_surface_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_procedural_candidate, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED:
            return _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_is_terminated, b.getInt1Ty(), {});
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY: {
            auto ox = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_x, b.getFloatTy(), {});
            auto oy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_y, b.getFloatTy(), {});
            auto oz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_origin_z, b.getFloatTy(), {});
            auto tmin = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_tmin, b.getFloatTy(), {});
            auto dx = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_x, b.getFloatTy(), {});
            auto dy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_y, b.getFloatTy(), {});
            auto dz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_direction_z, b.getFloatTy(), {});
            auto tmax = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_ray_tmax, b.getFloatTy(), {});
            auto llvm_f32x3_array_type = llvm::ArrayType::get(b.getFloatTy(), 3);
            auto origin = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            origin = b.CreateInsertValue(origin, ox, 0);
            origin = b.CreateInsertValue(origin, oy, 1);
            origin = b.CreateInsertValue(origin, oz, 2);
            auto direction = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            direction = b.CreateInsertValue(direction, dx, 0);
            direction = b.CreateInsertValue(direction, dy, 1);
            direction = b.CreateInsertValue(direction, dz, 2);
            auto result_type = _get_llvm_ray_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, origin, llvm_ray_type_origin_index);
            result = b.CreateInsertValue(result, tmin, llvm_ray_type_t_min_index);
            result = b.CreateInsertValue(result, direction, llvm_ray_type_direction_index);
            result = b.CreateInsertValue(result, tmax, llvm_ray_type_t_max_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_CANDIDATE_OBJECT_SPACE_RAY: {
            auto ox = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_origin_x, b.getFloatTy(), {});
            auto oy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_origin_y, b.getFloatTy(), {});
            auto oz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_origin_z, b.getFloatTy(), {});
            auto tmin = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_tmin, b.getFloatTy(), {});
            auto dx = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_direction_x, b.getFloatTy(), {});
            auto dy = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_direction_y, b.getFloatTy(), {});
            auto dz = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_direction_z, b.getFloatTy(), {});
            auto tmax = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_object_ray_tmax, b.getFloatTy(), {});
            auto llvm_f32x3_array_type = llvm::ArrayType::get(b.getFloatTy(), 3);
            auto origin = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            origin = b.CreateInsertValue(origin, ox, 0);
            origin = b.CreateInsertValue(origin, oy, 1);
            origin = b.CreateInsertValue(origin, oz, 2);
            auto direction = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_f32x3_array_type));
            direction = b.CreateInsertValue(direction, dx, 0);
            direction = b.CreateInsertValue(direction, dy, 1);
            direction = b.CreateInsertValue(direction, dz, 2);
            auto result_type = _get_llvm_ray_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, origin, llvm_ray_type_origin_index);
            result = b.CreateInsertValue(result, tmin, llvm_ray_type_t_min_index);
            result = b.CreateInsertValue(result, direction, llvm_ray_type_direction_index);
            result = b.CreateInsertValue(result, tmax, llvm_ray_type_t_max_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto u = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_bary_u, b.getFloatTy(), {});
            auto v = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_bary_v, b.getFloatTy(), {});
            auto t = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_hit_t, b.getFloatTy(), {});
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_surface_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_surface_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_surface_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_surface_hit_type_bary_index);
            result = b.CreateInsertValue(result, t, llvm_surface_hit_type_t_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_candidate_prim_id, b.getInt32Ty(), {});
            auto result_type = _get_llvm_procedural_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_procedural_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_procedural_hit_type_prim_id_index);
            return result;
        }
        case xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT: {
            auto inst_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_inst_id, b.getInt32Ty(), {});
            auto prim_id = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_prim_id, b.getInt32Ty(), {});
            auto hit_kind = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_hit_kind, b.getInt32Ty(), {});
            // Read committed hit float fields through wrapper function calls,
            // then pass each result through an inline asm barrier to make
            // the value opaque to the optimizer.  Without this barrier, the
            // LLVM O2 pipeline (specifically FunctionAttrs + downstream DCE)
            // eliminates the entire barycentric-interpolation → hit-position
            // → shadow-ray computation chain because the float values only
            // feed into FP math (no observable memory side-effects), unlike
            // the integer fields which are used for buffer GEP+load.
            auto u_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_bary_u, b.getFloatTy(), {});
            auto u = _create_opaque_float_barrier(b, u_raw, "committed.bary.u");
            auto v_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_bary_v, b.getFloatTy(), {});
            auto v = _create_opaque_float_barrier(b, v_raw, "committed.bary.v");
            auto t_raw = _call_ray_query_intrinsic(b, llvm_state_ptr, llvm_ray_query_intrinsic_name_committed_hit_t, b.getFloatTy(), {});
            auto t = _create_opaque_float_barrier(b, t_raw, "committed.hit.t");
            auto bary = _create_llvm_vector(b, {u, v});
            auto result_type = _get_llvm_committed_hit_type();
            auto result = static_cast<llvm::Value *>(llvm::PoisonValue::get(result_type));
            result = b.CreateInsertValue(result, inst_id, llvm_committed_hit_type_inst_id_index);
            result = b.CreateInsertValue(result, prim_id, llvm_committed_hit_type_prim_id_index);
            result = b.CreateInsertValue(result, bary, llvm_committed_hit_type_bary_index);
            result = b.CreateInsertValue(result, hit_kind, llvm_committed_hit_type_hit_kind_index);
            result = b.CreateInsertValue(result, t, llvm_committed_hit_type_t_index);
            return result;
        }
        default: break;
    }
    LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectReadInst.", luisa::to_underlying(op));
}

void HIPCodegenLLVMImpl::_translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept {
    auto intrinsic = [op = inst->op()] {
        switch (op) {
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE: return llvm_ray_query_intrinsic_name_commit_surface_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL: return llvm_ray_query_intrinsic_name_commit_procedural_hit;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE: return llvm_ray_query_intrinsic_name_terminate;
            case xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED: return llvm_ray_query_intrinsic_name_proceed;
            default: break;
        }
        LUISA_ERROR("Invalid op (code = {}) for RayQueryObjectWriteInst.", luisa::to_underlying(op));
    }();
    LUISA_DEBUG_ASSERT(inst->type() == nullptr);
    LUISA_DEBUG_ASSERT(inst->operand_count() == 1 || inst->operand_count() == 2);
    auto llvm_state_ptr = _get_ray_query_state_pointer(
        b, func_ctx, inst->operand(0));
    llvm::SmallVector<llvm::Value *, 2> llvm_args;
    for (auto &&op_use : inst->operand_uses().subspan(1) /* skip the query object */) {
        llvm_args.emplace_back(_get_llvm_value(b, func_ctx, op_use->value()));
    }
    if (inst->op() ==
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED) {
        LUISA_DEBUG_ASSERT(llvm_args.empty());
        if (_uses_hardware_rt_stack) {
            (void)_advance_ray_query(b, llvm_state_ptr);
        } else {
            (void)_call_ray_query_intrinsic(
                b, llvm_state_ptr, intrinsic, b.getVoidTy(), llvm_args);
        }
    } else {
        (void)_call_ray_query_intrinsic(
            b, llvm_state_ptr, intrinsic, b.getVoidTy(), llvm_args);
    }
}

void HIPCodegenLLVMImpl::_translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept {
    auto query_object = inst->query_object();
    LUISA_ASSERT(
        query_object != nullptr &&
            (query_object->type() == Type::of<RayQueryAll>() ||
             query_object->type() == Type::of<RayQueryAny>()),
        "Invalid HIP ray-query pipeline object.");

    auto llvm_on_surface = _get_or_declare_llvm_function(
        inst->on_surface_function());
    auto llvm_on_procedural = _get_or_declare_llvm_function(
        inst->on_procedural_function());
    const auto observation_masks =
        ray_query_handler_observation_masks(inst);
    const auto post_state_observed =
        ray_query_pipeline_post_state_is_observed(inst);
    const auto native_closest_reduction =
        ray_query_pipeline_admits_native_closest_reduction(inst);
    LUISA_ASSERT(
        _ray_query_pipeline_count <=
            std::numeric_limits<uint32_t>::max(),
        "HIP RayQuery pipeline index overflow.");
    const auto pipeline_index = static_cast<uint32_t>(
        _ray_query_pipeline_count++);
    const auto pipeline_is_resumable =
        _function_uses_resumable_ray_query_state(
            inst->parent_function());
    const auto pipeline_is_synchronous =
        _uses_synchronous_ray_query_pipeline ||
        (_uses_mixed_ray_query_pipeline && !pipeline_is_resumable);
    LUISA_VERBOSE(
        "HIP ray-query pipeline codegen {}: synchronous = {}, pipeline = {}.",
        static_cast<const void *>(this),
        pipeline_is_synchronous,
        pipeline_index);
    LUISA_ASSERT(
        llvm_on_surface->getReturnType()->isVoidTy() &&
            llvm_on_procedural->getReturnType()->isVoidTy(),
        "HIP ray-query candidate handlers must return void.");

    // Candidate handlers take the query object by reference. Materialize an
    // rvalue query if necessary; lowered AST ray queries normally already use
    // an alloca here, but accepting both forms keeps RayQueryPipelineInst's
    // documented operand contract intact.
    auto llvm_query_object = _get_llvm_value(b, func_ctx, query_object);
    llvm::Value *llvm_query_pointer;
    if (llvm_query_object->getType()->isPointerTy()) {
        llvm_query_pointer = llvm_query_object;
    } else {
        llvm_query_pointer = _create_temp_in_alloca_block(
            func_ctx, _get_llvm_type(query_object->type())->mem_type,
            _get_type_alignment(query_object->type()));
        _store_llvm_value(
            b, llvm_query_pointer, llvm_query_object,
            query_object->type());
    }
    if (llvm_query_pointer->getType()->getPointerAddressSpace() != 0u) {
        llvm_query_pointer = b.CreateAddrSpaceCast(
            llvm_query_pointer, b.getPtrTy(0),
            "ray.query.object.generic");
    }

    // Form the exact ordinary-callable ABI used by _translate_call_inst:
    // (query-ref, captures..., print?, dispatch-size, kernel-id, rt-stack...).
    llvm::SmallVector<llvm::Value *, 16> llvm_callback_args;
    llvm_callback_args.reserve(inst->captured_argument_count() + 8u);
    llvm_callback_args.emplace_back(llvm_query_pointer);
    for (auto captured_use : inst->captured_argument_uses()) {
        auto llvm_arg = _get_llvm_value(
            b, func_ctx, captured_use->value());
        if (llvm_arg->getType()->isPointerTy() &&
            llvm_arg->getType()->getPointerAddressSpace() != 0u) {
            llvm_arg = b.CreateAddrSpaceCast(
                llvm_arg, b.getPtrTy(0),
                "ray.query.capture.generic");
        }
        llvm_callback_args.emplace_back(llvm_arg);
    }
    if (_config.requires_printing) {
        auto llvm_print_buffer = static_cast<llvm::Value *>(
            llvm::PoisonValue::get(_get_llvm_print_buffer_type()));
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer,
            func_ctx.llvm_print_buffer_capacity, 0u);
        llvm_print_buffer = b.CreateInsertValue(
            llvm_print_buffer,
            func_ctx.llvm_print_buffer_content, 1u);
        llvm_callback_args.emplace_back(llvm_print_buffer);
    }
    llvm_callback_args.emplace_back(_read_dispatch_size(b, func_ctx));
    llvm_callback_args.emplace_back(_read_kernel_id(b, func_ctx));
    if (_rt_analysis.uses_ray_tracing) {
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_size);
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_count);
        llvm_callback_args.emplace_back(func_ctx.llvm_rt_stack_data);
    }

    llvm::SmallVector<llvm::Type *, 16> llvm_callback_arg_types;
    llvm_callback_arg_types.reserve(llvm_callback_args.size());
    for (auto llvm_arg : llvm_callback_args) {
        llvm_callback_arg_types.emplace_back(llvm_arg->getType());
    }
    auto llvm_pipeline_type = llvm::FunctionType::get(
        b.getVoidTy(), llvm_callback_arg_types, false);
    LUISA_ASSERT(
        llvm_on_surface->getFunctionType() == llvm_pipeline_type &&
            llvm_on_procedural->getFunctionType() == llvm_pipeline_type,
        "HIP ray-query pipeline callback ABI mismatch.");

    if (pipeline_is_synchronous) {
        // The reduction proof describes the handler semantics, not the
        // selected execution route. A module that requires reentrant ray
        // queries lowers every pipeline to the ordinary in-kernel loop; in
        // that route no native wrapper or callback dispatcher is reachable.
        // Keep the module feature bit equal to the disjunction of *lowered*
        // native calls, otherwise post-processing links a callback wrapper
        // whose dispatcher was never emitted.
        // A proven reduction is semantically one closest traversal: each
        // candidate contributes only reject, commit(t), or terminate, and the
        // final observable state is the minimum committed t. Keep that
        // transaction inside HIPRT's one-shot closest traversal on every
        // architecture. Pre-gfx12 uses HIPRT's dynamically assigned spill
        // stack. Gfx12 uses the same traversal with a statically indexed global
        // spill stack: the host sizes it from the physical launch domain, so
        // it needs neither the dynamic stack lock nor a resumable per-candidate
        // frontier. This is a semantic lowering selected from the handler
        // proof, not a renderer- or scene-specific policy.
        const auto use_static_global_hiprt_closest =
            native_closest_reduction && _uses_hardware_rt_stack;
        const auto use_native_hiprt_closest =
            native_closest_reduction && !_uses_hardware_rt_stack;
        LUISA_VERBOSE(
            "HIP native closest route: reduction = {}, hardware-stack = {}, "
            "motion-blur = {}, dynamic-global = {}, static-global = {}.",
            native_closest_reduction, _uses_hardware_rt_stack,
            _rt_analysis.uses_motion_blur,
            use_native_hiprt_closest,
            use_static_global_hiprt_closest);
        _uses_native_closest_ray_query_pipeline |=
            use_native_hiprt_closest ||
            use_static_global_hiprt_closest;
        _uses_static_global_rt_stack |=
            use_static_global_hiprt_closest;
        _uses_iterative_synchronous_ray_query_pipeline |=
            !(use_native_hiprt_closest ||
              use_static_global_hiprt_closest);
        // Materialize the exact callback environment once. The native HIPRT
        // filter/intersection callbacks receive only an opaque context pointer;
        // this typed struct restores the ordinary Callable ABI without an
        // indirect device-function call or callback-specific backend pattern.
        auto llvm_context_type = llvm::StructType::get(
            _llvm_context, llvm_callback_arg_types, false);
        auto llvm_context_pointer = _create_temp_in_alloca_block(
            func_ctx, llvm_context_type,
            _data_layout->getABITypeAlign(llvm_context_type).value());
        luisa::vector<llvm::StoreInst *> llvm_context_stores;
        llvm_context_stores.reserve(llvm_callback_args.size());
        for (auto i = 0u; i < llvm_callback_args.size(); ++i) {
            auto llvm_field = b.CreateStructGEP(
                llvm_context_type, llvm_context_pointer, i,
                "ray.query.context.field");
            llvm_context_stores.emplace_back(
                b.CreateStore(llvm_callback_args[i], llvm_field));
        }
        auto llvm_generic_context = llvm_context_pointer;
        if (llvm_generic_context->getType()->getPointerAddressSpace() != 0u) {
            llvm_generic_context = b.CreateAddrSpaceCast(
                llvm_generic_context, b.getPtrTy(0),
                "ray.query.context.generic");
        }

        // One direct switch is shared by all pipelines in the module. Each
        // case decodes its own typed context and directly invokes the two XIR
        // handlers, preserving reference captures and arbitrary side effects.
        if (_llvm_ray_query_pipeline_dispatch == nullptr) {
            auto llvm_dispatch_type = llvm::FunctionType::get(
                b.getVoidTy(),
                {b.getPtrTy(0), b.getPtrTy(0),
                 b.getInt32Ty(), b.getInt32Ty()},
                false);
            _llvm_ray_query_pipeline_dispatch = llvm::Function::Create(
                llvm_dispatch_type, llvm::Function::ExternalLinkage,
                "luisa_ray_query_pipeline_dispatch", _llvm_module.get());
            _llvm_ray_query_pipeline_dispatch->addFnAttr(
                llvm::Attribute::NoUnwind);
            _llvm_ray_query_pipeline_dispatch->getArg(2u)->addAttr(
                llvm::Attribute::get(
                    _llvm_context,
                    llvm_constant_argument_specialization_attribute));
            auto llvm_dispatch_entry = llvm::BasicBlock::Create(
                _llvm_context, "entry", _llvm_ray_query_pipeline_dispatch);
            auto llvm_dispatch_invalid = llvm::BasicBlock::Create(
                _llvm_context, "invalid", _llvm_ray_query_pipeline_dispatch);
            IB dispatch_b{llvm_dispatch_entry};
            _llvm_ray_query_pipeline_switch = dispatch_b.CreateSwitch(
                _llvm_ray_query_pipeline_dispatch->getArg(2),
                llvm_dispatch_invalid, 0u);
            dispatch_b.SetInsertPoint(llvm_dispatch_invalid);
            dispatch_b.CreateUnreachable();
        }

        // Candidate handlers admit two useful quotients of public RayQuery
        // state. Q0 carries only candidate identity/attributes. Qo additionally
        // carries the active object-space ray, but deliberately not the world
        // ray or committed hit. These are distinct dispatchers so the common
        // Q0 surface callback never pays Qo's eight-float call ABI. Both
        // reconstruct the ordinary query-reference contract in a temporary
        // state that LLVM can scalar-replace and return only
        // {commit, terminate, committed_t}.
        auto create_compact_dispatch = [&]<bool carries_object_ray>(
                                           llvm::Function *&dispatcher,
                                           llvm::SwitchInst *&dispatcher_switch,
                                           llvm::Value *&query,
                                           llvm::BasicBlock *&finish) noexcept {
            if (dispatcher != nullptr) { return; }
            llvm::SmallVector<llvm::Type *, 17> argument_types{
                b.getPtrTy(0), b.getInt32Ty(), b.getInt32Ty(),
                b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(),
                b.getFloatTy(), b.getFloatTy(), b.getFloatTy()};
            if constexpr (carries_object_ray) {
                argument_types.append(
                    {b.getFloatTy(), b.getFloatTy(), b.getFloatTy(),
                     b.getFloatTy(), b.getFloatTy(), b.getFloatTy(),
                     b.getFloatTy(), b.getFloatTy()});
            }
            auto dispatcher_type = llvm::FunctionType::get(
                b.getInt64Ty(), argument_types, false);
            constexpr auto dispatcher_name = carries_object_ray ?
                                                 "luisa_pipeline_ray_query_dispatch_compact_object_ray" :
                                                 "luisa_pipeline_ray_query_dispatch_compact";
            dispatcher = llvm::Function::Create(
                dispatcher_type, llvm::Function::ExternalLinkage,
                dispatcher_name, _llvm_module.get());
            dispatcher->addFnAttr(llvm::Attribute::NoUnwind);
            dispatcher->getArg(1u)->addAttr(
                llvm::Attribute::get(
                    _llvm_context,
                    llvm_constant_argument_specialization_attribute));

            auto entry = llvm::BasicBlock::Create(
                _llvm_context, "entry", dispatcher);
            auto invalid = llvm::BasicBlock::Create(
                _llvm_context, "invalid", dispatcher);
            finish = llvm::BasicBlock::Create(
                _llvm_context, "finish", dispatcher);

            IB compact_b{entry};
            auto compact_state_type = llvm::ArrayType::get(
                compact_b.getInt8Ty(),
                hip_synchronous_ray_query_state_size);
            auto compact_state = compact_b.CreateAlloca(
                compact_state_type, nullptr,
                carries_object_ray ?
                    "ray.query.compact.object.state" :
                    "ray.query.compact.state");
            compact_state->setAlignment(llvm::Align{16u});
            auto compact_field = [&](uint32_t offset,
                                     const llvm::Twine &name) noexcept {
                return compact_b.CreateInBoundsGEP(
                    compact_state_type, compact_state,
                    {compact_b.getInt32(0u), compact_b.getInt32(offset)},
                    name);
            };
            auto compact_store = [&](llvm::Value *value, uint32_t offset,
                                     llvm::Align alignment,
                                     const llvm::Twine &name) noexcept {
                auto field = compact_field(offset, name);
                auto store = compact_b.CreateStore(value, field);
                store->setAlignment(alignment);
            };
            auto compact_arg = [&](uint32_t index) noexcept -> llvm::Argument * {
                return dispatcher->getArg(index);
            };
            if constexpr (carries_object_ray) {
                constexpr std::array ray_fields{
                    compact_query_layout::ray_origin_x,
                    compact_query_layout::ray_origin_y,
                    compact_query_layout::ray_origin_z,
                    compact_query_layout::ray_t_min,
                    compact_query_layout::ray_direction_x,
                    compact_query_layout::ray_direction_y,
                    compact_query_layout::ray_direction_z,
                    compact_query_layout::ray_t_max};
                for (auto i = 0u; i < ray_fields.size(); ++i) {
                    compact_store(
                        compact_arg(9u + i), ray_fields[i],
                        llvm::Align{4u},
                        llvm::Twine{"ray.query.compact.object.ray."} +
                            llvm::Twine{i});
                }
            } else {
                compact_store(
                    compact_arg(8u), compact_query_layout::ray_t_max,
                    llvm::Align{4u}, "ray.query.compact.ray.tmax");
            }
            compact_store(
                compact_arg(4u), compact_query_layout::candidate_instance,
                llvm::Align{16u}, "ray.query.compact.candidate.instance");
            compact_store(
                compact_arg(5u), compact_query_layout::candidate_primitive,
                llvm::Align{4u}, "ray.query.compact.candidate.primitive");
            compact_store(
                compact_arg(6u), compact_query_layout::candidate_u,
                llvm::Align{8u}, "ray.query.compact.candidate.u");
            compact_store(
                compact_arg(7u), compact_query_layout::candidate_v,
                llvm::Align{4u}, "ray.query.compact.candidate.v");
            compact_store(
                compact_arg(8u), compact_query_layout::candidate_t,
                llvm::Align{16u}, "ray.query.compact.candidate.t");
            auto compact_state_address = compact_b.CreatePtrToInt(
                compact_state, compact_b.getInt32Ty(),
                "ray.query.compact.state.address");
            compact_store(
                compact_state_address,
                compact_query_layout::query_address,
                llvm::Align{8u}, "ray.query.compact.query.address");
            compact_store(
                compact_arg(3u), compact_query_layout::flags,
                llvm::Align{16u}, "ray.query.compact.flags");
            compact_store(
                compact_b.CreateTrunc(
                    compact_arg(2u), compact_b.getInt8Ty()),
                compact_query_layout::state,
                llvm::Align{8u}, "ray.query.compact.kind");
            compact_store(
                compact_b.getInt8(0u),
                compact_query_layout::candidate_committed,
                llvm::Align{2u}, "ray.query.compact.committed");
            compact_store(
                compact_b.getInt8(0u),
                compact_query_layout::terminated,
                llvm::Align{1u}, "ray.query.compact.terminated");

            query = compact_field(
                compact_query_layout::query_address,
                "ray.query.compact.query");
            if (query->getType()->getPointerAddressSpace() != 0u) {
                query = compact_b.CreateAddrSpaceCast(
                    query, compact_b.getPtrTy(0),
                    "ray.query.compact.query.generic");
            }
            dispatcher_switch = compact_b.CreateSwitch(
                compact_arg(1u), invalid, 0u);

            compact_b.SetInsertPoint(invalid);
            compact_b.CreateUnreachable();

            compact_b.SetInsertPoint(finish);
            auto compact_load = [&](llvm::Type *type, uint32_t offset,
                                    llvm::Align alignment,
                                    const llvm::Twine &name) noexcept {
                auto field = compact_b.CreateInBoundsGEP(
                    compact_state_type, compact_state,
                    {compact_b.getInt32(0u), compact_b.getInt32(offset)},
                    name + ".address");
                auto load = compact_b.CreateLoad(type, field, name);
                load->setAlignment(alignment);
                return load;
            };
            auto compact_committed = compact_load(
                compact_b.getInt8Ty(),
                compact_query_layout::candidate_committed,
                llvm::Align{2u}, "ray.query.compact.result.committed");
            auto compact_terminated = compact_load(
                compact_b.getInt8Ty(),
                compact_query_layout::terminated,
                llvm::Align{1u}, "ray.query.compact.result.terminated");
            auto compact_distance = compact_load(
                compact_b.getFloatTy(),
                compact_query_layout::ray_t_max,
                llvm::Align{4u}, "ray.query.compact.result.distance");
            auto compact_result_flags = compact_b.CreateOr(
                compact_b.CreateZExt(
                    compact_committed, compact_b.getInt32Ty()),
                compact_b.CreateShl(
                    compact_b.CreateZExt(
                        compact_terminated, compact_b.getInt32Ty()),
                    compact_b.getInt32(1u)));
            auto compact_distance_bits = compact_b.CreateBitCast(
                compact_distance, compact_b.getInt32Ty());
            auto compact_result = compact_b.CreateOr(
                compact_b.CreateZExt(
                    compact_result_flags, compact_b.getInt64Ty()),
                compact_b.CreateShl(
                    compact_b.CreateZExt(
                        compact_distance_bits, compact_b.getInt64Ty()),
                    compact_b.getInt64(32u)));
            compact_b.CreateRet(compact_result);
        };
        create_compact_dispatch.template operator()<false>(
            _llvm_ray_query_pipeline_compact_dispatch,
            _llvm_ray_query_pipeline_compact_switch,
            _llvm_ray_query_pipeline_compact_query,
            _llvm_ray_query_pipeline_compact_finish);
        create_compact_dispatch.template operator()<true>(
            _llvm_ray_query_pipeline_compact_object_ray_dispatch,
            _llvm_ray_query_pipeline_compact_object_ray_switch,
            _llvm_ray_query_pipeline_compact_object_ray_query,
            _llvm_ray_query_pipeline_compact_object_ray_finish);

        auto llvm_pipeline_block = llvm::BasicBlock::Create(
            _llvm_context,
            llvm::Twine{"pipeline."} + llvm::Twine{pipeline_index},
            _llvm_ray_query_pipeline_dispatch);
        _llvm_ray_query_pipeline_switch->addCase(
            b.getInt32(pipeline_index), llvm_pipeline_block);
        IB dispatch_b{llvm_pipeline_block};
        auto llvm_dispatch_context =
            _llvm_ray_query_pipeline_dispatch->getArg(1);
        llvm::SmallVector<llvm::Value *, 16> llvm_decoded_args;
        llvm_decoded_args.reserve(llvm_callback_arg_types.size());
        luisa::vector<llvm::LoadInst *> llvm_context_loads;
        llvm_context_loads.reserve(llvm_callback_arg_types.size());
        for (auto i = 0u; i < llvm_callback_arg_types.size(); ++i) {
            auto llvm_field = dispatch_b.CreateStructGEP(
                llvm_context_type, llvm_dispatch_context, i,
                "ray.query.context.field");
            auto llvm_value = dispatch_b.CreateLoad(
                llvm_callback_arg_types[i], llvm_field,
                "ray.query.context.value");
            llvm_decoded_args.emplace_back(llvm_value);
            llvm_context_loads.emplace_back(llvm_value);
        }
        auto llvm_surface_block = llvm::BasicBlock::Create(
            _llvm_context, "surface", _llvm_ray_query_pipeline_dispatch);
        auto llvm_procedural_block = llvm::BasicBlock::Create(
            _llvm_context, "procedural", _llvm_ray_query_pipeline_dispatch);
        auto llvm_invalid_kind_block = llvm::BasicBlock::Create(
            _llvm_context, "invalid.kind", _llvm_ray_query_pipeline_dispatch);
        auto llvm_kind_switch = dispatch_b.CreateSwitch(
            _llvm_ray_query_pipeline_dispatch->getArg(3),
            llvm_invalid_kind_block, 2u);
        llvm_kind_switch->addCase(
            dispatch_b.getInt32(llvm_ray_query_state_surface_candidate),
            llvm_surface_block);
        llvm_kind_switch->addCase(
            dispatch_b.getInt32(llvm_ray_query_state_procedural_candidate),
            llvm_procedural_block);

        dispatch_b.SetInsertPoint(llvm_surface_block);
        auto llvm_surface_call = dispatch_b.CreateCall(
            llvm_on_surface, llvm_decoded_args);
        llvm_surface_call->setCallingConv(
            llvm_on_surface->getCallingConv());
        dispatch_b.CreateRetVoid();

        dispatch_b.SetInsertPoint(llvm_procedural_block);
        auto llvm_procedural_call = dispatch_b.CreateCall(
            llvm_on_procedural, llvm_decoded_args);
        llvm_procedural_call->setCallingConv(
            llvm_on_procedural->getCallingConv());
        dispatch_b.CreateRetVoid();

        dispatch_b.SetInsertPoint(llvm_invalid_kind_block);
        dispatch_b.CreateUnreachable();

        auto append_compact_pipeline = [&](llvm::Function *dispatcher,
                                           llvm::SwitchInst *dispatcher_switch,
                                           llvm::Value *compact_query,
                                           llvm::BasicBlock *finish,
                                           const llvm::Twine &suffix) noexcept {
            auto pipeline_block = llvm::BasicBlock::Create(
                _llvm_context,
                llvm::Twine{"pipeline."} + llvm::Twine{pipeline_index},
                dispatcher);
            dispatcher_switch->addCase(
                b.getInt32(pipeline_index), pipeline_block);
            IB compact_b{pipeline_block};
            auto compact_context = dispatcher->getArg(0u);
            llvm::SmallVector<llvm::Value *, 16> decoded_args;
            decoded_args.reserve(llvm_callback_arg_types.size());
            decoded_args.emplace_back(compact_query);
            luisa::vector<llvm::LoadInst *> context_loads;
            context_loads.reserve(llvm_callback_arg_types.size());
            context_loads.emplace_back(nullptr);
            for (auto i = 1u; i < llvm_callback_arg_types.size(); ++i) {
                auto field = compact_b.CreateStructGEP(
                    llvm_context_type, compact_context, i,
                    llvm::Twine{"ray.query.compact.context.field"} + suffix);
                auto value = compact_b.CreateLoad(
                    llvm_callback_arg_types[i], field,
                    llvm::Twine{"ray.query.compact.context.value"} + suffix);
                decoded_args.emplace_back(value);
                context_loads.emplace_back(value);
            }
            auto surface_block = llvm::BasicBlock::Create(
                _llvm_context, "surface", dispatcher);
            auto procedural_block = llvm::BasicBlock::Create(
                _llvm_context, "procedural", dispatcher);
            auto invalid_kind_block = llvm::BasicBlock::Create(
                _llvm_context, "invalid.kind", dispatcher);
            auto kind_switch = compact_b.CreateSwitch(
                dispatcher->getArg(2u), invalid_kind_block, 2u);
            kind_switch->addCase(
                compact_b.getInt32(
                    llvm_ray_query_state_surface_candidate),
                surface_block);
            kind_switch->addCase(
                compact_b.getInt32(
                    llvm_ray_query_state_procedural_candidate),
                procedural_block);

            compact_b.SetInsertPoint(surface_block);
            auto surface_call = compact_b.CreateCall(
                llvm_on_surface, decoded_args);
            surface_call->setCallingConv(llvm_on_surface->getCallingConv());
            compact_b.CreateBr(finish);

            compact_b.SetInsertPoint(procedural_block);
            auto procedural_call = compact_b.CreateCall(
                llvm_on_procedural, decoded_args);
            procedural_call->setCallingConv(
                llvm_on_procedural->getCallingConv());
            compact_b.CreateBr(finish);

            compact_b.SetInsertPoint(invalid_kind_block);
            compact_b.CreateUnreachable();
            return context_loads;
        };
        auto llvm_compact_context_loads = append_compact_pipeline(
            _llvm_ray_query_pipeline_compact_dispatch,
            _llvm_ray_query_pipeline_compact_switch,
            _llvm_ray_query_pipeline_compact_query,
            _llvm_ray_query_pipeline_compact_finish, "");
        auto llvm_compact_object_ray_context_loads = append_compact_pipeline(
            _llvm_ray_query_pipeline_compact_object_ray_dispatch,
            _llvm_ray_query_pipeline_compact_object_ray_switch,
            _llvm_ray_query_pipeline_compact_object_ray_query,
            _llvm_ray_query_pipeline_compact_object_ray_finish,
            ".object.ray");

        auto llvm_state_pointer = _get_ray_query_state_pointer(
            b, func_ctx, query_object);
        const auto requires_full_candidate_state =
            ray_query_observation_requires_full_state(
                observation_masks.surface) ||
            ray_query_observation_requires_full_state(
                observation_masks.procedural);
        if (requires_full_candidate_state) {
            // Only a full-dispatch callback transaction exports query
            // identity. Candidate-only and object-ray-only native closest
            // reductions use compact quotients and recover every observable
            // value from explicit callback arguments.
            auto llvm_query_address_field = b.CreateInBoundsGEP(
                b.getInt8Ty(), llvm_state_pointer,
                b.getInt32(compact_query_layout::query_address),
                "ray.query.identity.field");
            auto llvm_state_address = b.CreatePtrToInt(
                llvm_state_pointer, b.getInt32Ty(),
                "ray.query.identity.address");
            auto llvm_identity_store = b.CreateStore(
                llvm_state_address, llvm_query_address_field);
            llvm_identity_store->setAlignment(llvm::Align{8u});
        }
        if (llvm_state_pointer->getType()->getPointerAddressSpace() != 0u) {
            llvm_state_pointer = b.CreateAddrSpaceCast(
                llvm_state_pointer, b.getPtrTy(0),
                "ray.query.state.generic");
        }
        LUISA_ASSERT(
            func_ctx.llvm_rt_stack_size != nullptr &&
                func_ctx.llvm_rt_stack_count != nullptr &&
                func_ctx.llvm_rt_stack_data != nullptr,
            "Synchronous HIP ray query requires an RT stack buffer.");
        auto llvm_trace_type = llvm::FunctionType::get(
            b.getVoidTy(),
            {b.getPtrTy(0), b.getPtrTy(0), b.getInt32Ty(),
             b.getInt32Ty(), b.getInt32Ty(), b.getInt32Ty(),
             b.getPtrTy(0)},
            false);
        auto llvm_trace_name = use_static_global_hiprt_closest ?
                                   (_rt_analysis.writes_instance_opacity ?
                                        "luisa_pipeline_ray_query_trace_all_native_closest_global_stack" :
                                        "luisa_pipeline_ray_query_trace_all_native_closest_global_stack_stable_opacity") :
                               use_native_hiprt_closest ?
                                   (_rt_analysis.writes_instance_opacity ?
                                        "luisa_pipeline_ray_query_trace_all_native_closest" :
                                        "luisa_pipeline_ray_query_trace_all_native_closest_stable_opacity") :
                                   (query_object->type() == Type::of<RayQueryAny>() ?
                                        (_rt_analysis.writes_instance_opacity ?
                                             "luisa_pipeline_ray_query_trace_any" :
                                             "luisa_pipeline_ray_query_trace_any_stable_opacity") :
                                        (_rt_analysis.writes_instance_opacity ?
                                             "luisa_pipeline_ray_query_trace_all" :
                                             "luisa_pipeline_ray_query_trace_all_stable_opacity"));
        auto llvm_trace = _llvm_module->getFunction(
            llvm_trace_name);
        if (llvm_trace == nullptr) {
            llvm_trace = llvm::Function::Create(
                llvm_trace_type, llvm::Function::ExternalLinkage,
                llvm_trace_name, _llvm_module.get());
        } else {
            LUISA_ASSERT(llvm_trace->getFunctionType() == llvm_trace_type,
                         "HIP synchronous ray-query trace ABI mismatch.");
        }
        auto llvm_trace_call = b.CreateCall(
            llvm_trace,
            {llvm_state_pointer, llvm_generic_context,
             b.getInt32(pipeline_index),
             b.getInt32(observation_masks.encoded()),
             func_ctx.llvm_rt_stack_size,
             func_ctx.llvm_rt_stack_count, func_ctx.llvm_rt_stack_data});
        _llvm_ray_query_pipeline_contexts.emplace_back(
            RayQueryPipelineContext{
                pipeline_index,
                inst->parent_function(),
                llvm::cast<llvm::AllocaInst>(llvm_context_pointer),
                llvm_generic_context,
                llvm_trace_call,
                llvm_on_surface,
                llvm_on_procedural,
                post_state_observed,
                requires_full_candidate_state,
                native_closest_reduction,
                ray_query_observation_requires_distinct_ray_states(
                    observation_masks.surface) ||
                    ray_query_observation_requires_distinct_ray_states(
                        observation_masks.procedural),
                std::move(llvm_context_stores),
                std::move(llvm_context_loads),
                std::move(llvm_compact_context_loads),
                std::move(llvm_compact_object_ray_context_loads)});
        return;
    }

    // Keep the pipeline's control flow inside a private helper. Expanding it
    // directly in the containing LLVM block would invalidate XIR PHI incoming
    // block mappings whenever the pipeline precedes a branch.
    auto llvm_pipeline = llvm::Function::Create(
        llvm_pipeline_type, llvm::Function::PrivateLinkage,
        llvm::Twine{"luisa.ray.query.pipeline."} +
            llvm::Twine{pipeline_index},
        _llvm_module.get());
    llvm_pipeline->addFnAttr(llvm::Attribute::AlwaysInline);
    llvm_pipeline->addFnAttr(llvm::Attribute::NoUnwind);

    auto llvm_entry = llvm::BasicBlock::Create(
        _llvm_context, "entry", llvm_pipeline);
    auto llvm_dispatch = llvm::BasicBlock::Create(
        _llvm_context, "dispatch", llvm_pipeline);
    auto llvm_surface = llvm::BasicBlock::Create(
        _llvm_context, "surface", llvm_pipeline);
    auto llvm_procedural = llvm::BasicBlock::Create(
        _llvm_context, "procedural", llvm_pipeline);
    auto llvm_exit = llvm::BasicBlock::Create(
        _llvm_context, "exit", llvm_pipeline);

    llvm::SmallVector<llvm::Value *, 16> llvm_pipeline_args;
    llvm_pipeline_args.reserve(llvm_pipeline->arg_size());
    for (auto &llvm_arg : llvm_pipeline->args()) {
        llvm_pipeline_args.emplace_back(&llvm_arg);
    }

    IB pipeline_b{llvm_entry};
    auto llvm_state_address = pipeline_b.CreateAlignedLoad(
        _get_llvm_ray_query_type(), llvm_pipeline_args.front(),
        llvm::Align{_get_type_alignment(query_object->type())},
        "ray.query.object");
    auto llvm_state_pointer = pipeline_b.CreateIntToPtr(
        llvm_state_address,
        pipeline_b.getPtrTy(amdgpu_address_space_local),
        "ray.query.state");
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_dispatch);
    auto llvm_state = _advance_ray_query(
        pipeline_b, llvm_state_pointer);
    auto llvm_switch = pipeline_b.CreateSwitch(
        llvm_state, llvm_exit, 2u);
    llvm_switch->addCase(
        pipeline_b.getInt8(llvm_ray_query_state_surface_candidate),
        llvm_surface);
    llvm_switch->addCase(
        pipeline_b.getInt8(llvm_ray_query_state_procedural_candidate),
        llvm_procedural);

    pipeline_b.SetInsertPoint(llvm_surface);
    auto llvm_surface_call = pipeline_b.CreateCall(
        llvm_on_surface, llvm_pipeline_args);
    llvm_surface_call->setCallingConv(
        llvm_on_surface->getCallingConv());
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_procedural);
    auto llvm_procedural_call = pipeline_b.CreateCall(
        llvm_on_procedural, llvm_pipeline_args);
    llvm_procedural_call->setCallingConv(
        llvm_on_procedural->getCallingConv());
    pipeline_b.CreateBr(llvm_dispatch);

    pipeline_b.SetInsertPoint(llvm_exit);
    pipeline_b.CreateRetVoid();

    auto llvm_call = b.CreateCall(
        llvm_pipeline, llvm_callback_args);
    llvm_call->setCallingConv(llvm_pipeline->getCallingConv());
}

llvm::Value *HIPCodegenLLVMImpl::_get_ray_query_state_pointer(
    IB &b, const FunctionContext &func_ctx,
    const xir::Value *query_object) noexcept {
    LUISA_ASSERT(
        query_object != nullptr &&
            (query_object->type() == Type::of<RayQueryAll>() ||
             query_object->type() == Type::of<RayQueryAny>()),
        "Invalid HIP ray-query object operand.");
    if (ray_query_value_has_function_local_state(query_object)) {
        LUISA_ASSERT(func_ctx.llvm_rq_state != nullptr,
                     "Missing HIP state for function-local RayQuery.");
        return func_ctx.llvm_rq_state;
    }
    auto llvm_query = _get_llvm_value(b, func_ctx, query_object);
    if (llvm_query->getType()->isPointerTy()) {
        llvm_query = _load_llvm_value(
            b, llvm_query, query_object->type());
    }
    LUISA_ASSERT(
        llvm_query->getType() == _get_llvm_ray_query_type(),
        "Invalid HIP ray-query LLVM object type.");
    return b.CreateIntToPtr(
        llvm_query,
        b.getPtrTy(amdgpu_address_space_local),
        "ray.query.state");
}

llvm::Value *HIPCodegenLLVMImpl::_advance_ray_query(
    IB &b, llvm::Value *llvm_state_ptr) noexcept {
    if (_uses_hardware_rt_stack) {
        return _call_ray_query_intrinsic(
            b, llvm_state_ptr,
            llvm_ray_query_intrinsic_name_advance,
            b.getInt8Ty(), {});
    }
    (void)_call_ray_query_intrinsic(
        b, llvm_state_ptr,
        llvm_ray_query_intrinsic_name_proceed,
        b.getVoidTy(), {});
    return _call_ray_query_intrinsic(
        b, llvm_state_ptr,
        llvm_ray_query_intrinsic_name_state,
        b.getInt8Ty(), {});
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(
    IB &b, llvm::Value *llvm_state_ptr, llvm::StringRef name,
    llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args,
    bool use_pipeline_abi) noexcept {
    LUISA_ASSERT(
        llvm_state_ptr != nullptr &&
            llvm_state_ptr->getType()->isPointerTy(),
        "Invalid HIP ray-query state pointer.");
    if (!_uses_hardware_rt_stack || use_pipeline_abi) {
        if (llvm_state_ptr->getType()->getPointerAddressSpace() != 0u) {
            llvm_state_ptr = b.CreateAddrSpaceCast(
                llvm_state_ptr, b.getPtrTy(0),
                "rq.state.generic");
        }
    }
    llvm::SmallVector<llvm::Value *, 8> augmented_args;
    augmented_args.push_back(llvm_state_ptr);
    augmented_args.append(args.begin(), args.end());
    std::string motion_name;
    auto wrapper_name = name;
    if (use_pipeline_abi) {
        static constexpr std::string_view prefix{"luisa_ray_query_"};
        LUISA_ASSERT(name.starts_with(prefix),
                     "Invalid HIP ray-query wrapper name '{}'.", name.str());
        motion_name = "luisa_pipeline_ray_query_";
        motion_name.append(name.drop_front(prefix.size()).str());
        wrapper_name = motion_name;
    } else if (_supports_hardware_rt_stack &&
               !_uses_hardware_rt_stack) {
        // On gfx12 the generic DynamicStack implementation is emitted under
        // the historical motion-query symbol family. It is also the required
        // reentrant path when a static query handler performs a nested trace;
        // selecting by the actual stack plan keeps those two reasons unified.
        static constexpr std::string_view prefix{"luisa_ray_query_"};
        LUISA_ASSERT(name.starts_with(prefix),
                     "Invalid HIP ray-query wrapper name '{}'.", name.str());
        motion_name = "luisa_motion_ray_query_";
        motion_name.append(name.drop_front(prefix.size()).str());
        wrapper_name = motion_name;
    }
    auto func = _llvm_module->getFunction(wrapper_name);
    if (func == nullptr) {
        llvm::SmallVector<llvm::Type *, 8> arg_types;
        for (auto arg : augmented_args) { arg_types.push_back(arg->getType()); }
        auto func_type = llvm::FunctionType::get(ret, arg_types, false);
        func = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            wrapper_name, _llvm_module.get());
    }
    return b.CreateCall(func, augmented_args);
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(
    IB &b, llvm::Value *llvm_state_ptr, llvm::StringRef name,
    llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    // Pointer-based operations may execute in an outlined candidate handler.
    // In an all-synchronous module the pointer necessarily names the compact
    // pipeline state. In a mixed module the same handler body can be shared by
    // either function-level ABI, so its generic wrapper accesses only the
    // representation-identical common prefix.
    return _call_ray_query_intrinsic(
        b, llvm_state_ptr, name, ret, args,
        _uses_synchronous_ray_query_pipeline);
}

llvm::Value *HIPCodegenLLVMImpl::_call_ray_query_intrinsic(
    IB &b, FunctionContext &func_ctx, llvm::StringRef name,
    llvm::Type *ret, llvm::ArrayRef<llvm::Value *> args) noexcept {
    return _call_ray_query_intrinsic(
        b, func_ctx.llvm_rq_state, name, ret, args,
        !func_ctx.llvm_rq_state_uses_resumable_abi);
}

llvm::Value *HIPCodegenLLVMImpl::_create_opaque_float_barrier(IB &b, llvm::Value *val, const llvm::Twine &name) noexcept {
    auto *float_ty = b.getFloatTy();
    auto *asm_func_ty = llvm::FunctionType::get(float_ty, {float_ty}, false);
    // The asm keeps a consumed value opaque to LLVM, but it has no observable
    // side effect of its own. Marking it side-effecting retained committed-hit
    // loads even when a query only inspected miss(), as traverse_any does.
    auto *ia = llvm::InlineAsm::get(asm_func_ty, "v_mov_b32 $0, $1", "=v,v", /*hasSideEffects=*/false);
    return b.CreateCall(asm_func_ty, ia, {val}, name);
}

}// namespace luisa::compute::hip
