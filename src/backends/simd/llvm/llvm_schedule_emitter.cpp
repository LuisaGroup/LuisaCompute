#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <unordered_set>

#include "../../common/env_flag.h"

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_ray_query_type(const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           (type == Type::custom("LC_RayQueryAll") ||
            type == Type::custom("LC_RayQueryAny"));
}

[[nodiscard]] bool is_indirect_dispatch_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type == Type::custom("LC_IndirectDispatchBuffer");
}

[[nodiscard]] constexpr bool is_power_of_two(uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] constexpr uint32_t exact_log2(uint32_t value) noexcept {
    auto result = uint32_t{0u};
    while (value > 1u) {
        value >>= 1u;
        result++;
    }
    return result;
}

}// namespace

ScheduleEmitter::ScheduleEmitter(
    ::llvm::Module &module, const schedule::Function &source, uint32_t width,
    std::string_view entry_name, bool enable_fast_math,
    std::array<uint32_t, 3u> static_block_size,
    bool enable_uniform_buffer_broadcast,
    bool enable_lane_affine_buffer,
    bool enable_paired_leaf_gather,
    uint32_t dispatch_worker_count,
    bool enable_native_predicated_loop,
    bool enable_runtime_packet_geometry,
    bool enable_linear_1d_packet_tail_narrowing)
    : _module{module},
      _source{source},
      _width{width},
      _entry_name{entry_name},
      _enable_fast_math{enable_fast_math},
      _static_block_size{static_block_size},
      _enable_uniform_buffer_broadcast{enable_uniform_buffer_broadcast},
      _enable_lane_affine_buffer{enable_lane_affine_buffer},
      _enable_paired_leaf_gather{enable_paired_leaf_gather},
      _dispatch_worker_count{std::max(dispatch_worker_count, 1u)},
      _enable_native_predicated_loop{enable_native_predicated_loop},
      _enable_runtime_packet_geometry{enable_runtime_packet_geometry},
      _enable_linear_1d_packet_tail_narrowing{
          enable_linear_1d_packet_tail_narrowing},
      _layout{module.getContext(), width},
      _collectives{width},
      _builder{module.getContext()} {}

void ScheduleEmitter::_fail(std::string message) {
    if (_result.error.empty()) { _result.error = std::move(message); }
}

[[nodiscard]] bool ScheduleEmitter::_failed() const noexcept {
    return !_result.error.empty();
}

[[nodiscard]] size_t ScheduleEmitter::_align_up(size_t value,
                                                size_t alignment) noexcept {
    return alignment == 0u ? value :
                             (value + alignment - 1u) & ~(alignment - 1u);
}

[[nodiscard]] bool ScheduleEmitter::_is_scalar_data(const Type *type) noexcept {
    return type != nullptr &&
           ((type->is_scalar() && !type->is_float8()) ||
            is_ray_query_type(type));
}

[[nodiscard]] bool ScheduleEmitter::_is_data(const Type *type) noexcept {
    if (type == nullptr || type->is_resource() ||
        (type->is_custom() && !is_ray_query_type(type)) ||
        type->is_cooperative_vector() ||
        type->is_cooperative_vector_ref() ||
        type->is_cooperative_matrix_ref()) {
        return false;
    }
    if (is_ray_query_type(type)) { return true; }
    if (type->is_scalar()) { return !type->is_float8(); }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return _is_data(type->element());
    }
    if (type->is_structure()) {
        return std::all_of(
            type->members().begin(), type->members().end(),
            [](auto *member) noexcept { return _is_data(member); });
    }
    return false;
}

[[nodiscard]] size_t ScheduleEmitter::_abi_size(const Type *type) noexcept {
    if (is_ray_query_type(type)) { return sizeof(void *); }
    if (is_indirect_dispatch_type(type)) {
        return sizeof(SIMDHostBufferView);
    }
    if (type != nullptr && type->is_buffer()) {
        return sizeof(SIMDHostBufferView);
    }
    if (type != nullptr && type->is_texture()) {
        return sizeof(SIMDHostTextureView);
    }
    if (type != nullptr && type->is_bindless_array()) {
        return sizeof(SIMDHostBindlessArrayView);
    }
    if (type != nullptr && type->is_accel()) {
        return sizeof(SIMDHostAccelView);
    }
    return type == nullptr ? 0u : type->size();
}

[[nodiscard]] size_t ScheduleEmitter::_abi_alignment(const Type *type) noexcept {
    if (is_ray_query_type(type)) { return alignof(void *); }
    if (is_indirect_dispatch_type(type)) {
        return alignof(SIMDHostBufferView);
    }
    if (type != nullptr && type->is_buffer()) {
        return alignof(SIMDHostBufferView);
    }
    if (type != nullptr && type->is_texture()) {
        return alignof(SIMDHostTextureView);
    }
    if (type != nullptr && type->is_bindless_array()) {
        return alignof(SIMDHostBindlessArrayView);
    }
    return type == nullptr ? 1u : type->alignment();
}

[[nodiscard]] uint32_t ScheduleEmitter::_child_count(const Type *type) noexcept {
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return type->dimension();
    }
    if (type->is_structure()) {
        return static_cast<uint32_t>(type->members().size());
    }
    return 0u;
}

[[nodiscard]] const Type *ScheduleEmitter::_child_type(
    const Type *type, uint32_t index) noexcept {
    if (type->is_vector() || type->is_array()) {
        return type->element();
    }
    if (type->is_matrix()) {
        return Type::vector(type->element(), type->dimension());
    }
    if (type->is_structure()) { return type->members()[index]; }
    return nullptr;
}

[[nodiscard]] size_t ScheduleEmitter::_child_offset(
    const Type *type, uint32_t index) noexcept {
    if (type->is_vector()) {
        return static_cast<size_t>(index) * type->element()->size();
    }
    if (type->is_matrix() || type->is_array()) {
        return static_cast<size_t>(index) *
               (type->size() / type->dimension());
    }
    if (type->is_structure()) {
        auto offset = size_t{0u};
        for (auto i = uint32_t{0u}; i <= index; i++) {
            auto *member = type->members()[i];
            offset = _align_up(offset, member->alignment());
            if (i == index) { return offset; }
            offset += member->size();
        }
    }
    return 0u;
}

[[nodiscard]] ::llvm::Type *ScheduleEmitter::_data_type(
    const Type *type, bool varying) {
    auto *result = _layout.expression_type(schedule::Value{
        .value_class = varying ? schedule::ValueClass::varying :
                                 schedule::ValueClass::warp_uniform,
        .type = type,
    });
    if (result == nullptr) { _fail(_layout.error()); }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_extract_child(
    ::llvm::Value *aggregate, const Type *type, uint32_t index,
    bool varying) {
    if (type->is_vector() && !varying) {
        return _builder.CreateExtractElement(aggregate, index);
    }
    return _builder.CreateExtractValue(aggregate, {index});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_insert_child(
    ::llvm::Value *aggregate, ::llvm::Value *child,
    const Type *type, uint32_t index, bool varying) {
    if (type->is_vector() && !varying) {
        return _builder.CreateInsertElement(aggregate, child, index);
    }
    return _builder.CreateInsertValue(aggregate, child, {index});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_assemble(
    const Type *type, bool varying,
    const std::function<::llvm::Value *(uint32_t)> &child) {
    auto *llvm_type = _data_type(type, varying);
    if (llvm_type == nullptr) { return nullptr; }
    auto *result = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(llvm_type));
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        auto *value = child(i);
        if (value == nullptr) { return nullptr; }
        result = _insert_child(result, value, type, i, varying);
    }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_splat_data(
    ::llvm::Value *value, const Type *type) {
    if (_is_scalar_data(type)) {
        return _builder.CreateVectorSplat(_width, value);
    }
    if (!_is_data(type)) {
        _fail("cannot splat a non-data Schedule IR value");
        return nullptr;
    }
    return _assemble(type, true, [&](uint32_t i) {
        return _splat_data(
            _extract_child(value, type, i, false),
            _child_type(type, i));
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_extract_lane(
    ::llvm::Value *value, const Type *type, ::llvm::Value *lane) {
    if (_is_scalar_data(type)) {
        return _builder.CreateExtractElement(value, lane);
    }
    return _assemble(type, false, [&](uint32_t i) {
        return _extract_lane(
            _extract_child(value, type, i, true),
            _child_type(type, i), lane);
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_masked_merge(
    ::llvm::Value *new_value, ::llvm::Value *old_value,
    const Type *type, ::llvm::Value *mask) {
    if (_is_scalar_data(type)) {
        return _builder.CreateSelect(mask, new_value, old_value);
    }
    return _assemble(type, true, [&](uint32_t i) {
        auto *child_type = _child_type(type, i);
        return _masked_merge(
            _extract_child(new_value, type, i, true),
            _extract_child(old_value, type, i, true),
            child_type, mask);
    });
}

[[nodiscard]] ::llvm::StructType *ScheduleEmitter::_local_handle_type() {
    auto *pointer = ::llvm::PointerType::getUnqual(
        _module.getContext());
    return ::llvm::StructType::get(
        _module.getContext(),
        {::llvm::FixedVectorType::get(pointer, _width),
         ::llvm::FixedVectorType::get(
             _builder.getInt64Ty(), _width)});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_handle(
    ::llvm::Value *base, ::llvm::Value *offsets) {
    auto *handle = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(_local_handle_type()));
    handle = _builder.CreateInsertValue(handle, base, {0u});
    return _builder.CreateInsertValue(handle, offsets, {1u});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_base(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *handle) {
    return builder.CreateExtractValue(handle, {0u});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_offsets(
    ::llvm::IRBuilder<> &builder, ::llvm::Value *handle) {
    return builder.CreateExtractValue(handle, {1u});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_merge_local_handles(
    ::llvm::Value *new_handle, ::llvm::Value *old_handle,
    ::llvm::Value *mask) {
    auto *base = _builder.CreateSelect(
        mask,
        _local_base(_builder, new_handle),
        _local_base(_builder, old_handle));
    auto *offsets = _builder.CreateSelect(
        mask,
        _local_offsets(_builder, new_handle),
        _local_offsets(_builder, old_handle));
    return _local_handle(base, offsets);
}

[[nodiscard]] bool ScheduleEmitter::_is_local_lvalue(
    schedule::ValueId id) const noexcept {
    return id.value < _local_lvalue_values.size() &&
           _local_lvalue_values[id.value] != 0u;
}

[[nodiscard]] bool ScheduleEmitter::_is_shared_lvalue(
    schedule::ValueId id) const noexcept {
    return id.value < _shared_lvalue_values.size() &&
           _shared_lvalue_values[id.value] != 0u;
}

void ScheduleEmitter::_for_each_assignment(
    const schedule::BasicBlock &block,
    const std::function<void(schedule::EdgeAssignment)> &visit) {
    auto visit_edge = [&](const schedule::ControlEdge &edge) {
        for (auto assignment : edge.assignments) { visit(assignment); }
    };
    std::visit(
        [&](const auto &terminator) {
            using T = std::decay_t<decltype(terminator)>;
            if constexpr (std::is_same_v<
                              T, schedule::BranchTerminator>) {
                visit_edge(terminator.edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SplitTerminator>) {
                visit_edge(terminator.true_edge);
                visit_edge(terminator.false_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::SwitchTerminator>) {
                for (auto &&item : terminator.cases) {
                    visit_edge(item.edge);
                }
                visit_edge(terminator.default_edge);
            } else if constexpr (std::is_same_v<
                                     T, schedule::JoinTerminator> ||
                                 std::is_same_v<
                                     T, schedule::LoopBackTerminator>) {
                for (auto assignment : terminator.assignments) {
                    visit(assignment);
                }
            } else if constexpr (std::is_same_v<
                                     T,
                                     schedule::BlockBarrierTerminator>) {
                visit_edge(terminator.resume_edge);
            }
        },
        block.terminator);
}

void ScheduleEmitter::_analyze_local_lvalues() {
    _local_lvalue_values.assign(
        _source.values().size(), uint8_t{0u});
    _shared_lvalue_values.assign(
        _source.values().size(), uint8_t{0u});
    std::vector<std::vector<schedule::ValueId>> dependents(
        _source.values().size());
    struct PendingLValue {
        schedule::ValueId value{};
        bool shared{false};
    };
    std::vector<PendingLValue> ready;
    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::alloca) {
                if (!instruction.result || !instruction.source_op) {
                    _fail("allocation is missing its result or address space");
                    return;
                }
                auto op = static_cast<xir::AllocaOp>(
                    *instruction.source_op);
                if (op != xir::AllocaOp::LOCAL &&
                    op != xir::AllocaOp::SHARED) {
                    _fail("packet allocation has an unsupported address space");
                    return;
                }
                _local_lvalue_values[instruction.result->value] = 1u;
                auto shared = op == xir::AllocaOp::SHARED;
                _shared_lvalue_values[instruction.result->value] =
                    static_cast<uint8_t>(shared);
                ready.emplace_back(PendingLValue{
                    .value = *instruction.result,
                    .shared = shared,
                });
            } else if (instruction.opcode == schedule::Opcode::gep &&
                       instruction.result &&
                       !instruction.operands.empty()) {
                dependents[instruction.operands.front().value]
                    .emplace_back(*instruction.result);
            }
        }
        _for_each_assignment(
            block, [&](schedule::EdgeAssignment assignment) {
                dependents[assignment.source.value]
                    .emplace_back(assignment.destination);
            });
    }
    for (auto i = size_t{0u}; i < ready.size(); i++) {
        for (auto dependent : dependents[ready[i].value.value]) {
            if (!_is_local_lvalue(dependent)) {
                _local_lvalue_values[dependent.value] = 1u;
                _shared_lvalue_values[dependent.value] =
                    static_cast<uint8_t>(ready[i].shared);
                ready.emplace_back(PendingLValue{
                    .value = dependent,
                    .shared = ready[i].shared,
                });
            } else if (_is_shared_lvalue(dependent) !=
                       ready[i].shared) {
                _fail("control flow mixes local and shared references");
                return;
            }
        }
    }

    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::gep) {
                if (!instruction.result ||
                    instruction.operands.empty() ||
                    !_is_local_lvalue(instruction.operands.front())) {
                    _fail("LLVM packet codegen only supports GEPs rooted in thread-local storage");
                    return;
                }
            } else if (instruction.opcode == schedule::Opcode::load ||
                       instruction.opcode == schedule::Opcode::store) {
                if (instruction.operands.empty() ||
                    !_is_local_lvalue(instruction.operands.front())) {
                    _fail("LLVM packet codegen only supports loads and stores to thread-local storage");
                    return;
                }
            } else if ((instruction.opcode == schedule::Opcode::ray_query_read ||
                        instruction.opcode == schedule::Opcode::ray_query_write) &&
                       (instruction.operands.empty() ||
                        !_is_local_lvalue(instruction.operands.front()) ||
                        _is_shared_lvalue(instruction.operands.front()))) {
                _fail("ray-query object access requires thread-local storage");
                return;
            } else if (instruction.opcode == schedule::Opcode::atomic &&
                       !instruction.operands.empty() &&
                       _is_local_lvalue(instruction.operands.front()) &&
                       !_is_shared_lvalue(instruction.operands.front())) {
                _fail("thread-local atomics are not supported by LLVM packet codegen");
                return;
            }
        }
        _for_each_assignment(
            block, [&](schedule::EdgeAssignment assignment) {
                if (_is_local_lvalue(assignment.destination) !=
                        _is_local_lvalue(assignment.source) ||
                    _is_shared_lvalue(assignment.destination) !=
                        _is_shared_lvalue(assignment.source)) {
                    _fail("control-flow assignment mixes local references and data values");
                }
            });
        if (_failed()) { return; }
    }
}

void ScheduleEmitter::_preflight_edge(const schedule::ControlEdge &edge,
                                      bool split_edge) {
    static_cast<void>(split_edge);
    if (edge.loop_back && _source.loop(*edge.loop_back) == nullptr) {
        _fail("control edge references an invalid loop back-edge");
        return;
    }
    for (auto convergence : edge.joins) {
        auto *point = _source.convergence(convergence);
        if (point == nullptr || point->target != edge.target) {
            _fail("convergence arrival edge does not target its gate block");
            return;
        }
    }
}

void ScheduleEmitter::_preflight() {
    if (_width == 0u || _width > 128u) {
        _fail("LLVM packet specialization width must be in [1, 128]");
        return;
    }
    if (_source.logical_warp_width() != 0u &&
        _source.logical_warp_width() != _width) {
        _fail("Schedule IR warp width does not match LLVM specialization width");
        return;
    }
    for (auto size : _static_block_size) {
        if (size != 0u && !is_power_of_two(size)) {
            _fail("SIMD static block-size dimensions must be powers of two");
            return;
        }
    }
    auto verification = schedule::verify(_source);
    if (!verification.succeeded()) {
        _fail("cannot lower invalid Schedule IR: " +
              verification.errors.front().message);
        return;
    }
    for (auto &&block : _source.blocks()) {
        if (std::holds_alternative<
                schedule::BlockBarrierTerminator>(block.terminator)) {
            _has_block_barrier = true;
            _result.block_barrier_count++;
        }
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode == schedule::Opcode::alloca &&
                instruction.source_op &&
                static_cast<xir::AllocaOp>(*instruction.source_op) ==
                    xir::AllocaOp::SHARED) {
                _has_shared_memory = true;
            }
        }
    }
    _result.block_barrier_loop_epochs.resize(
        _result.block_barrier_count);
    _cooperative_loop_epoch_indices.assign(
        _source.loops().size(), -1);
    for (auto &&block : _source.blocks()) {
        auto *barrier = std::get_if<
            schedule::BlockBarrierTerminator>(&block.terminator);
        if (barrier == nullptr) { continue; }
        if (barrier->barrier_id >=
            _result.block_barrier_loop_epochs.size()) {
            _fail("block barrier ID exceeds the cooperative epoch table");
            return;
        }
        auto &barrier_epochs =
            _result.block_barrier_loop_epochs[barrier->barrier_id];
        for (auto &&loop : _source.loops()) {
            if (std::find(
                    loop.blocks.cbegin(), loop.blocks.cend(),
                    block.id) == loop.blocks.cend()) {
                continue;
            }
            if (loop.id.value >=
                _cooperative_loop_epoch_indices.size()) {
                _fail("natural-loop ID exceeds the cooperative epoch map");
                return;
            }
            auto &epoch_index =
                _cooperative_loop_epoch_indices[loop.id.value];
            if (epoch_index < 0) {
                if (_result.block_barrier_loop_epoch_count >=
                    static_cast<size_t>(
                        std::numeric_limits<int32_t>::max())) {
                    _fail("cooperative SIMD kernel has too many barrier loops");
                    return;
                }
                epoch_index = static_cast<int32_t>(
                    _result.block_barrier_loop_epoch_count++);
            }
            barrier_epochs.emplace_back(
                static_cast<uint32_t>(epoch_index));
        }
    }
    _cooperative_block = _has_block_barrier || _has_shared_memory;
    _result.cooperative_block = _cooperative_block;
    if (_cooperative_block) {
        auto block_thread_count = uint64_t{1u};
        for (auto size : _static_block_size) {
            if (size == 0u ||
                block_thread_count >
                    std::numeric_limits<uint32_t>::max() / size) {
                _fail("cooperative SIMD kernels require a finite static block size");
                return;
            }
            block_thread_count *= size;
        }
        if (block_thread_count % _width != 0u) {
            _fail("cooperative SIMD block size must be divisible by packet width");
            return;
        }
    }
    _analyze_local_lvalues();
    if (_failed()) { return; }
    _analyze_ray_query_scratch();
    if (_source.blocks().size() >
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        _fail("Schedule IR block count exceeds the packet PC width");
        return;
    }

    std::vector<const schedule::Value *> parameters;
    for (auto &&value : _source.values()) {
        if (value.origin == schedule::ValueOrigin::parameter) {
            auto *metadata = std::get_if<schedule::ParameterValueMetadata>(
                &value.metadata);
            if (metadata == nullptr ||
                value.value_class != schedule::ValueClass::warp_uniform) {
                _fail("packet parameters must be warp-uniform and carry argument metadata");
                return;
            }
            auto argument_tag = static_cast<xir::DerivedArgumentTag>(
                metadata->argument_tag);
            if ((argument_tag == xir::DerivedArgumentTag::REFERENCE &&
                 !is_indirect_dispatch_type(value.type)) ||
                (argument_tag == xir::DerivedArgumentTag::VALUE &&
                 !_is_data(value.type)) ||
                (argument_tag == xir::DerivedArgumentTag::RESOURCE &&
                 (value.type == nullptr ||
                  (!value.type->is_buffer() &&
                   !value.type->is_texture() &&
                   !value.type->is_bindless_array() &&
                   !value.type->is_accel())))) {
                _fail("packet ABI supports data, buffer, texture, bindless, accel, and indirect-dispatch arguments only");
                return;
            }
            if (parameters.size() <= metadata->index) {
                parameters.resize(metadata->index + 1u, nullptr);
            }
            if (parameters[metadata->index] != nullptr) {
                _fail("Schedule IR contains duplicate parameter indices");
                return;
            }
            parameters[metadata->index] = &value;
        } else if (value.origin != schedule::ValueOrigin::scheduler_builtin &&
                   value.type != nullptr && !_is_data(value.type) &&
                   !value.type->is_buffer() &&
                   !value.type->is_texture() &&
                   !value.type->is_bindless_array() &&
                   !value.type->is_accel()) {
            _fail("packet codegen encountered an unsupported Schedule IR value type");
            return;
        }
    }
    _parameter_offsets.resize(parameters.size());
    auto offset = size_t{0u};
    for (auto index = size_t{0u}; index < parameters.size(); index++) {
        if (parameters[index] == nullptr) {
            _fail("Schedule IR parameter indices must be dense");
            return;
        }
        auto *type = parameters[index]->type;
        offset = _align_up(offset, 16u);
        _parameter_offsets[index] = offset;
        offset += _abi_size(type);
        offset = _align_up(offset, 16u);
    }
    _result.argument_buffer_size = offset;

    for (auto &&block : _source.blocks()) {
        for (auto &&instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::arithmetic &&
                instruction.opcode != schedule::Opcode::cast &&
                instruction.opcode != schedule::Opcode::alloca &&
                instruction.opcode != schedule::Opcode::load &&
                instruction.opcode != schedule::Opcode::store &&
                instruction.opcode != schedule::Opcode::gep &&
                instruction.opcode != schedule::Opcode::resource_query &&
                instruction.opcode != schedule::Opcode::resource_read &&
                instruction.opcode != schedule::Opcode::resource_write &&
                instruction.opcode != schedule::Opcode::ray_query_read &&
                instruction.opcode != schedule::Opcode::ray_query_write &&
                instruction.opcode != schedule::Opcode::atomic &&
                instruction.opcode != schedule::Opcode::warp_collective &&
                instruction.opcode != schedule::Opcode::print &&
                instruction.opcode != schedule::Opcode::assert_ &&
                instruction.opcode != schedule::Opcode::clock) {
                auto message = std::string{
                                   "LLVM packet codegen encountered unsupported Schedule IR opcode '"} +
                               schedule::to_string(instruction.opcode) + "' in block '" + block.name + "'";
                if (instruction.source_op) {
                    message += " (source op " +
                               std::to_string(*instruction.source_op) + ")";
                }
                _fail(std::move(message));
                return;
            }
        }
        std::visit(
            [&](const auto &terminator) {
                using T = std::decay_t<decltype(terminator)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    _fail("Schedule IR block has no terminator");
                } else if constexpr (
                    std::is_same_v<T, schedule::BranchTerminator>) {
                    _preflight_edge(terminator.edge, false);
                } else if constexpr (
                    std::is_same_v<T, schedule::SplitTerminator>) {
                    _preflight_edge(terminator.true_edge, true);
                    _preflight_edge(terminator.false_edge, true);
                } else if constexpr (
                    std::is_same_v<T, schedule::SwitchTerminator>) {
                    auto *selector = _source.value(terminator.selector);
                    if (selector == nullptr ||
                        !_is_scalar_data(selector->type) ||
                        selector->type->is_float()) {
                        _fail("switch selector must be an integer scalar");
                        return;
                    }
                    auto bit_width = selector->type->is_bool() ?
                                         1u :
                                         static_cast<uint32_t>(selector->type->size() * 8u);
                    std::unordered_set<uint64_t> labels;
                    for (auto &&item : terminator.cases) {
                        if ((bit_width < 64u &&
                             (item.value >> bit_width) != 0u) ||
                            !labels.emplace(item.value).second) {
                            _fail("switch case labels must be canonical and unique");
                            return;
                        }
                        _preflight_edge(item.edge, true);
                    }
                    _preflight_edge(terminator.default_edge, true);
                } else if constexpr (
                    std::is_same_v<T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(
                        terminator.convergence);
                    if (point == nullptr) {
                        _fail("join references an invalid convergence point");
                    }
                } else if constexpr (
                    std::is_same_v<
                        T, schedule::BlockBarrierTerminator>) {
                    _preflight_edge(terminator.resume_edge, false);
                } else if constexpr (
                    std::is_same_v<T, schedule::ReturnTerminator>) {
                    if (terminator.value) {
                        auto *value = _source.value(*terminator.value);
                        if (value == nullptr ||
                            !_is_scalar_data(value->type) ||
                            value->type->is_bool()) {
                            _fail("Phase-2 packet ABI supports non-bool scalar returns only");
                        }
                    }
                } else if constexpr (!std::is_same_v<
                                         T,
                                         schedule::UnreachableTerminator>) {
                    _fail("Phase-2 LLVM packet dispatcher encountered an unsupported terminator");
                }
            },
            block.terminator);
        if (_failed()) { return; }
    }
}

[[nodiscard]] ::llvm::Constant *ScheduleEmitter::_scalar_constant(
    const Type *type, const std::byte *bytes) {
    if (!_is_scalar_data(type) || bytes == nullptr) {
        _fail("invalid scalar constant payload");
        return nullptr;
    }
    uint64_t raw = 0u;
    std::memcpy(&raw, bytes, std::min(type->size(), sizeof(raw)));
    auto *llvm_type = _layout.expression_type(schedule::Value{
        .value_class = schedule::ValueClass::warp_uniform,
        .type = type,
    });
    if (llvm_type == nullptr) {
        _fail(_layout.error());
        return nullptr;
    }
    if (type->is_bool()) {
        return ::llvm::ConstantInt::get(llvm_type, raw != 0u);
    }
    if (auto *integer = ::llvm::dyn_cast<::llvm::IntegerType>(llvm_type)) {
        return ::llvm::ConstantInt::get(integer, raw);
    }
    auto *integer = ::llvm::IntegerType::get(
        _module.getContext(), static_cast<unsigned>(type->size() * 8u));
    auto *bits = ::llvm::ConstantInt::get(integer, raw);
    return ::llvm::ConstantExpr::getBitCast(bits, llvm_type);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_lane_ids() {
    std::vector<::llvm::Constant *> lanes;
    lanes.reserve(_width);
    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        lanes.emplace_back(_builder.getInt32(lane));
    }
    return ::llvm::ConstantVector::get(lanes);
}

[[nodiscard]] ::llvm::Constant *ScheduleEmitter::_constant_data(
    const Type *type, const std::byte *bytes, size_t offset) {
    if (_is_scalar_data(type)) {
        return _scalar_constant(type, bytes + offset);
    }
    if (!_is_data(type)) {
        _fail("invalid aggregate constant payload");
        return nullptr;
    }
    auto *llvm_type = _data_type(type, false);
    if (llvm_type == nullptr) { return nullptr; }
    std::vector<::llvm::Constant *> children;
    children.reserve(_child_count(type));
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        auto *child = _constant_data(
            _child_type(type, i), bytes,
            offset + _child_offset(type, i));
        if (child == nullptr) { return nullptr; }
        children.emplace_back(child);
    }
    if (type->is_vector()) {
        return ::llvm::ConstantVector::get(children);
    }
    if (type->is_structure()) {
        return ::llvm::ConstantStruct::get(
            ::llvm::cast<::llvm::StructType>(llvm_type), children);
    }
    return ::llvm::ConstantArray::get(
        ::llvm::cast<::llvm::ArrayType>(llvm_type), children);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_byte_pointer(
    ::llvm::Value *base, size_t offset) {
    return offset == 0u ? base : _builder.CreateGEP(_builder.getInt8Ty(), base, _builder.getInt64(offset));
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_uniform_data(
    ::llvm::Value *base, const Type *type, size_t offset) {
    if (_is_scalar_data(type)) {
        auto *pointer = _byte_pointer(base, offset);
        if (type->is_bool()) {
            auto *load = _builder.CreateLoad(
                _builder.getInt8Ty(), pointer, "bool.arg");
            load->setAlignment(::llvm::Align{1u});
            return _builder.CreateICmpNE(load, _builder.getInt8(0u));
        }
        auto *llvm_type = _data_type(type, false);
        auto *load = _builder.CreateLoad(llvm_type, pointer);
        load->setAlignment(::llvm::Align{type->alignment()});
        return load;
    }
    return _assemble(type, false, [&](uint32_t i) {
        return _load_uniform_data(
            base, _child_type(type, i),
            offset + _child_offset(type, i));
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_buffer_view(
    ::llvm::Value *base) {
    auto *type = _layout.expression_type(schedule::Value{
        .value_class = schedule::ValueClass::warp_uniform,
        .type = Type::buffer(nullptr),
    });
    // The concrete buffer element type does not affect the descriptor.
    // Some Type registries do not expose buffer<byte>, so fall back to the
    // canonical literal LLVM descriptor when needed.
    if (type == nullptr) {
        type = ::llvm::StructType::get(
            _module.getContext(),
            {::llvm::PointerType::getUnqual(_module.getContext()),
             _builder.getInt64Ty()});
    }
    auto *result = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(type));
    auto *pointer = _builder.CreateLoad(
        ::llvm::PointerType::getUnqual(_module.getContext()), base);
    pointer->setAlignment(::llvm::Align{alignof(SIMDHostBufferView)});
    auto *size_pointer = _byte_pointer(
        base, offsetof(SIMDHostBufferView, size_bytes));
    auto *size = _builder.CreateLoad(_builder.getInt64Ty(), size_pointer);
    size->setAlignment(::llvm::Align{alignof(size_t)});
    result = _builder.CreateInsertValue(result, pointer, {0u});
    return _builder.CreateInsertValue(result, size, {1u});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_texture_view(
    ::llvm::Value *base) {
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *type = ::llvm::StructType::get(
        context,
        {pointer_type, pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type,
         _builder.getInt32Ty(), _builder.getInt32Ty(), pointer_type});
    auto *result = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(type));
    constexpr std::array pointer_offsets{
        offsetof(SIMDHostTextureView, texture),
        offsetof(SIMDHostTextureView, read_float),
        offsetof(SIMDHostTextureView, read_uint),
        offsetof(SIMDHostTextureView, write_float),
        offsetof(SIMDHostTextureView, write_uint),
        offsetof(SIMDHostTextureView, size),
    };
    for (auto i = uint32_t{0u}; i < pointer_offsets.size(); i++) {
        auto *field = _builder.CreateLoad(
            pointer_type, _byte_pointer(base, pointer_offsets[i]));
        field->setAlignment(::llvm::Align{alignof(void *)});
        result = _builder.CreateInsertValue(result, field, {i});
    }
    constexpr std::array u32_offsets{
        offsetof(SIMDHostTextureView, level),
        offsetof(SIMDHostTextureView, dimension),
    };
    for (auto i = uint32_t{0u}; i < u32_offsets.size(); i++) {
        auto *field = _builder.CreateLoad(
            _builder.getInt32Ty(),
            _byte_pointer(base, u32_offsets[i]));
        field->setAlignment(::llvm::Align{alignof(uint32_t)});
        result = _builder.CreateInsertValue(result, field, {i + 6u});
    }
    auto *sample = _builder.CreateLoad(
        pointer_type,
        _byte_pointer(
            base, offsetof(SIMDHostTextureView, sample_float)));
    sample->setAlignment(::llvm::Align{alignof(void *)});
    return _builder.CreateInsertValue(result, sample, {8u});
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_bindless_view(
    ::llvm::Value *base) {
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *type = ::llvm::StructType::get(
        context,
        {pointer_type, _builder.getInt64Ty(),
         pointer_type, pointer_type, pointer_type});
    auto *result = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(type));
    auto *slots = _builder.CreateLoad(
        pointer_type, base);
    slots->setAlignment(::llvm::Align{alignof(void *)});
    auto *size_pointer = _byte_pointer(
        base, offsetof(SIMDHostBindlessArrayView, size));
    auto *size = _builder.CreateLoad(
        _builder.getInt64Ty(), size_pointer);
    size->setAlignment(::llvm::Align{alignof(size_t)});
    result = _builder.CreateInsertValue(result, slots, {0u});
    result = _builder.CreateInsertValue(result, size, {1u});
    constexpr std::array callback_offsets{
        offsetof(SIMDHostBindlessArrayView, sample_texture),
        offsetof(SIMDHostBindlessArrayView, read_texture),
        offsetof(SIMDHostBindlessArrayView, size_texture),
    };
    for (auto i = uint32_t{0u}; i < callback_offsets.size(); i++) {
        auto *callback = _builder.CreateLoad(
            pointer_type, _byte_pointer(base, callback_offsets[i]));
        callback->setAlignment(::llvm::Align{alignof(void *)});
        result = _builder.CreateInsertValue(result, callback, {i + 2u});
    }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_accel_view(
    ::llvm::Value *base) {
    auto &context = _module.getContext();
    auto *pointer_type = ::llvm::PointerType::getUnqual(context);
    auto *type = ::llvm::StructType::get(
        context,
        {pointer_type, pointer_type, pointer_type,
         pointer_type, pointer_type, pointer_type});
    auto *result = static_cast<::llvm::Value *>(
        ::llvm::PoisonValue::get(type));
    constexpr std::array offsets{
        offsetof(SIMDHostAccelView, accel),
        offsetof(SIMDHostAccelView, trace_closest),
        offsetof(SIMDHostAccelView, trace_any),
        offsetof(SIMDHostAccelView, instances),
        offsetof(SIMDHostAccelView, ray_query_proceed),
        offsetof(SIMDHostAccelView, ray_query_proceed_wide),
    };
    for (auto i = uint32_t{0u}; i < offsets.size(); i++) {
        auto *field = _builder.CreateLoad(
            pointer_type, _byte_pointer(base, offsets[i]));
        field->setAlignment(::llvm::Align{alignof(void *)});
        result = _builder.CreateInsertValue(result, field, {i});
    }
    return result;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_launch_u32(size_t offset) {
    auto *pointer = _byte_pointer(_launch_config, offset);
    auto *load = _builder.CreateLoad(_builder.getInt32Ty(), pointer);
    load->setAlignment(::llvm::Align{alignof(uint32_t)});
    return load;
}

void ScheduleEmitter::_ensure_launch_vectors() {
    if (_block_size[0u] != nullptr || _failed()) { return; }
    for (auto i = uint32_t{0u}; i < 3u; i++) {
        _block_id[i] = _load_launch_u32(
            offsetof(SIMDPacketLaunchConfig, block_id) +
            sizeof(uint32_t) * i);
        _dispatch_size[i] = _load_launch_u32(
            offsetof(SIMDPacketLaunchConfig, dispatch_size) +
            sizeof(uint32_t) * i);
        _block_size[i] = _static_block_size[i] == 0u ?
                             _load_launch_u32(
                                 offsetof(SIMDPacketLaunchConfig, block_size) +
                                 sizeof(uint32_t) * i) :
                             _builder.getInt32(_static_block_size[i]);
    }
    auto *first = _load_launch_u32(
        offsetof(SIMDPacketLaunchConfig, thread_index));
    _linear_thread_indices = _builder.CreateAdd(
        _builder.CreateVectorSplat(_width, first), _lane_ids());
    auto static_power_of_two = true;
    for (auto size : _static_block_size) {
        static_power_of_two &= is_power_of_two(size);
    }
    if (static_power_of_two) {
        auto splat_u32 = [&](uint32_t value) noexcept {
            return _builder.CreateVectorSplat(
                _width, _builder.getInt32(value));
        };
        auto linear_1d =
            _enable_runtime_packet_geometry &&
            _static_block_size[1u] == 1u &&
            _static_block_size[2u] == 1u &&
            !luisa::compute::detail::env_flag(
                "LUISA_SIMD_DISABLE_LINEAR_1D_THREAD_ID");
        if (linear_1d) {
            // Runtime packet batching starts at thread zero and emits exactly
            // the statically known packets of one block. For a 1D block the
            // linear thread index is therefore already thread_id.x and the
            // other components are identically zero. Keep the standalone
            // packet ABI on the general decomposition because its caller may
            // deliberately supply an arbitrary thread index; decomposing its
            // overflow into y/z preserves block-boundary lane masking.
            _result.linear_1d_thread_id_count++;
            _thread_id[0u] = _linear_thread_indices;
            _thread_id[1u] = splat_u32(0u);
            _thread_id[2u] = _thread_id[1u];
        } else {
            _thread_id[0u] = _builder.CreateAnd(
                _linear_thread_indices,
                splat_u32(_static_block_size[0u] - 1u),
                "thread.id.x");
            auto *yz = _builder.CreateLShr(
                _linear_thread_indices,
                splat_u32(exact_log2(_static_block_size[0u])),
                "thread.id.yz");
            _thread_id[1u] = _builder.CreateAnd(
                yz, splat_u32(_static_block_size[1u] - 1u),
                "thread.id.y");
            _thread_id[2u] = _builder.CreateLShr(
                yz, splat_u32(exact_log2(_static_block_size[1u])),
                "thread.id.z");
        }
    } else {
        auto *x_size = _builder.CreateVectorSplat(
            _width, _block_size[0u]);
        auto *y_size = _builder.CreateVectorSplat(
            _width, _block_size[1u]);
        _thread_id[0u] = _builder.CreateURem(
            _linear_thread_indices, x_size);
        auto *yz = _builder.CreateUDiv(
            _linear_thread_indices, x_size);
        _thread_id[1u] = _builder.CreateURem(yz, y_size);
        _thread_id[2u] = _builder.CreateUDiv(yz, y_size);
    }
    for (auto i = uint32_t{0u}; i < 3u; i++) {
        auto *base = _builder.CreateMul(
            _block_id[i], _block_size[i]);
        _dispatch_id[i] = _builder.CreateAdd(
            _builder.CreateVectorSplat(_width, base),
            _thread_id[i]);
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_triplet(
    const Type *type, const std::array<::llvm::Value *, 3u> &values,
    bool varying) {
    return _assemble(type, varying, [&](uint32_t i) {
        return values[i];
    });
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_special_register(
    const schedule::Value &value,
    xir::DerivedSpecialRegisterTag tag) {
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID:
            return _lane_ids();
        case xir::DerivedSpecialRegisterTag::WARP_SIZE:
            return _builder.getInt32(_width);
        case xir::DerivedSpecialRegisterTag::KERNEL_ID:
            return _load_launch_u32(
                offsetof(SIMDPacketLaunchConfig, kernel_id));
        case xir::DerivedSpecialRegisterTag::THREAD_ID:
        case xir::DerivedSpecialRegisterTag::BLOCK_ID:
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID:
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE:
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE:
            _ensure_launch_vectors();
            break;
        default:
            _fail("packet ABI does not provide this special register");
            return nullptr;
    }
    switch (tag) {
        case xir::DerivedSpecialRegisterTag::THREAD_ID:
            return _triplet(value.type, _thread_id, true);
        case xir::DerivedSpecialRegisterTag::BLOCK_ID:
            return _triplet(value.type, _block_id, false);
        case xir::DerivedSpecialRegisterTag::DISPATCH_ID:
            return _triplet(value.type, _dispatch_id, true);
        case xir::DerivedSpecialRegisterTag::BLOCK_SIZE:
            return _triplet(value.type, _block_size, false);
        case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE:
            return _triplet(value.type, _dispatch_size, false);
        default: break;
    }
    return nullptr;
}

void ScheduleEmitter::_create_external_values() {
    _external_values.resize(_source.values().size(), nullptr);
    auto *i8 = _builder.getInt8Ty();
    for (auto &&value : _source.values()) {
        ::llvm::Value *llvm_value = nullptr;
        switch (value.origin) {
            case schedule::ValueOrigin::parameter: {
                auto *metadata = std::get_if<
                    schedule::ParameterValueMetadata>(&value.metadata);
                auto offset = _parameter_offsets[metadata->index];
                auto *pointer = _builder.CreateGEP(
                    i8, _argument_buffer, _builder.getInt64(offset));
                auto tag = static_cast<xir::DerivedArgumentTag>(
                    metadata->argument_tag);
                if (is_indirect_dispatch_type(value.type)) {
                    llvm_value = _load_buffer_view(pointer);
                } else if (tag == xir::DerivedArgumentTag::RESOURCE) {
                    if (value.type->is_buffer()) {
                        llvm_value = _load_buffer_view(pointer);
                    } else if (value.type->is_texture()) {
                        llvm_value = _load_texture_view(pointer);
                    } else if (value.type->is_bindless_array()) {
                        llvm_value = _load_bindless_view(pointer);
                    } else {
                        llvm_value = _load_accel_view(pointer);
                    }
                } else {
                    llvm_value = _load_uniform_data(pointer, value.type);
                }
                break;
            }
            case schedule::ValueOrigin::constant: {
                auto *metadata = std::get_if<
                    schedule::ConstantValueMetadata>(&value.metadata);
                llvm_value = _constant_data(
                    value.type, metadata->bytes.data());
                break;
            }
            case schedule::ValueOrigin::special_register: {
                auto *metadata = std::get_if<
                    schedule::SpecialRegisterValueMetadata>(&value.metadata);
                auto tag = static_cast<xir::DerivedSpecialRegisterTag>(
                    metadata->tag);
                llvm_value = _special_register(value, tag);
                break;
            }
            case schedule::ValueOrigin::scheduler_builtin:
            case schedule::ValueOrigin::instruction:
            case schedule::ValueOrigin::state_slot: break;
        }
        _external_values[value.id.value] = llvm_value;
        if (_failed()) { return; }
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_value(schedule::ValueId id) {
    auto *value = _source.value(id);
    if (value == nullptr) {
        _fail("LLVM packet codegen references an invalid value");
        return nullptr;
    }
    if (value->origin == schedule::ValueOrigin::scheduler_builtin) {
        return _active_mask;
    }
    if (value->origin == schedule::ValueOrigin::parameter ||
        value->origin == schedule::ValueOrigin::constant ||
        value->origin == schedule::ValueOrigin::special_register) {
        return _external_values[id.value];
    }
    if (value->origin == schedule::ValueOrigin::state_slot) {
        auto *state = _builder.CreateLoad(
            _state_slots[id.value]->getAllocatedType(),
            _state_slots[id.value], value->name + ".state");
        if (value->value_class == schedule::ValueClass::cohort_uniform &&
            !_direct_control_flow) {
            return _extract_lane(
                state, value->type,
                _safe_first_lane(_active_mask));
        }
        return state;
    }
    if (auto iter = _locals.find(id.value); iter != _locals.end()) {
        return iter->second;
    }
    if (id.value < _spilled_instruction_values.size() &&
        _spilled_instruction_values[id.value] != 0u) {
        auto *state = _builder.CreateLoad(
            _state_slots[id.value]->getAllocatedType(),
            _state_slots[id.value], value->name + ".spill.load");
        if (value->value_class == schedule::ValueClass::cohort_uniform &&
            !_direct_control_flow) {
            return _extract_lane(
                state, value->type,
                _safe_first_lane(_active_mask));
        }
        return state;
    }
    _fail("instruction value '" + value->name + "' (#" +
          std::to_string(id.value) +
          ") is not available in the current Schedule IR block");
    return nullptr;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_as_lane_vector(
    ::llvm::Value *value, const schedule::Value &schedule_value) {
    if (value == nullptr) { return nullptr; }
    if (schedule_value.value_class == schedule::ValueClass::varying ||
        schedule_value.value_class == schedule::ValueClass::mask) {
        return value;
    }
    if (!_is_data(schedule_value.type)) {
        _fail("cannot splat a non-data Schedule IR value");
        return nullptr;
    }
    return _splat_data(value, schedule_value.type);
}

[[nodiscard]] LLVMScheduleCodegenResult ScheduleEmitter::run() {
    _preflight();
    if (!_failed()) { _build(); }
    if (!_failed() && _entry != nullptr) {
        std::string verification_error;
        ::llvm::raw_string_ostream stream{verification_error};
        if (::llvm::verifyFunction(*_entry, &stream)) {
            stream.flush();
            _fail("generated invalid LLVM IR: " + verification_error);
        }
    }
    if (_failed() && _entry != nullptr) {
        _entry->eraseFromParent();
        _entry = nullptr;
    }
    _result.entry = _entry;
    return std::move(_result);
}

}// namespace luisa::compute::simd::detail
