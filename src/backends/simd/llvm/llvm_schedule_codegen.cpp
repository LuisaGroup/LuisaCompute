#include "llvm_schedule_codegen.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/APInt.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/op.h>
#include <luisa/xir/special_register.h>

#include "llvm_value_layout.h"
#include "llvm_warp_collectives.h"
#include "../schedule/schedule_ir.h"

namespace luisa::compute::simd {

namespace {

class ScheduleEmitter {

private:
    ::llvm::Module &_module;
    const schedule::Function &_source;
    uint32_t _width;
    std::string _entry_name;
    LLVMScheduleCodegenResult _result{};
    LLVMValueLayout _layout;
    LLVMWarpCollectives _collectives;
    ::llvm::IRBuilder<> _builder;
    ::llvm::Function *_entry{nullptr};
    ::llvm::Value *_argument_buffer{nullptr};
    ::llvm::Value *_return_buffer{nullptr};
    ::llvm::Value *_launch_config{nullptr};
    ::llvm::Value *_active_lane_count{nullptr};
    ::llvm::BasicBlock *_scheduler_loop{nullptr};
    ::llvm::AllocaInst *_live_mask{nullptr};
    ::llvm::AllocaInst *_runnable_mask{nullptr};
    ::llvm::AllocaInst *_pc_state{nullptr};
    ::llvm::AllocaInst *_token_state{nullptr};
    ::llvm::AllocaInst *_frame_active{nullptr};
    ::llvm::AllocaInst *_frame_static_id{nullptr};
    ::llvm::AllocaInst *_frame_parent_token{nullptr};
    ::llvm::AllocaInst *_frame_expected{nullptr};
    ::llvm::AllocaInst *_frame_arrived{nullptr};
    std::vector<::llvm::AllocaInst *> _loop_epochs{};
    std::vector<std::vector<schedule::LoopId>> _block_loops{};
    std::vector<::llvm::AllocaInst *> _state_slots{};
    std::vector<uint8_t> _spilled_instruction_values{};
    std::vector<::llvm::Value *> _external_values{};
    std::vector<size_t> _parameter_offsets{};
    std::unordered_map<uint32_t, ::llvm::Value *> _locals{};
    ::llvm::Value *_active_mask{nullptr};
    ::llvm::Value *_seed_lane{nullptr};
    ::llvm::Value *_linear_thread_indices{nullptr};
    std::array<::llvm::Value *, 3u> _block_id{};
    std::array<::llvm::Value *, 3u> _dispatch_size{};
    std::array<::llvm::Value *, 3u> _block_size{};
    std::array<::llvm::Value *, 3u> _thread_id{};
    std::array<::llvm::Value *, 3u> _dispatch_id{};

private:
    void _fail(std::string message) {
        if (_result.error.empty()) { _result.error = std::move(message); }
    }

    [[nodiscard]] bool _failed() const noexcept {
        return !_result.error.empty();
    }

    [[nodiscard]] static size_t _align_up(size_t value,
                                          size_t alignment) noexcept {
        return alignment == 0u ? value :
                                (value + alignment - 1u) & ~(alignment - 1u);
    }

    [[nodiscard]] static bool _is_scalar_data(const Type *type) noexcept {
        return type != nullptr && type->is_scalar() && !type->is_float8();
    }

    [[nodiscard]] static bool _is_data(const Type *type) noexcept {
        if (type == nullptr || type->is_resource() || type->is_custom() ||
            type->is_cooperative_vector() ||
            type->is_cooperative_vector_ref() ||
            type->is_cooperative_matrix_ref()) {
            return false;
        }
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

    [[nodiscard]] static size_t _abi_size(const Type *type) noexcept {
        if (type != nullptr && type->is_buffer()) {
            return sizeof(SIMDHostBufferView);
        }
        if (type != nullptr && type->is_texture()) {
            return sizeof(SIMDHostTextureView);
        }
        return type == nullptr ? 0u : type->size();
    }

    [[nodiscard]] static size_t _abi_alignment(const Type *type) noexcept {
        if (type != nullptr && type->is_buffer()) {
            return alignof(SIMDHostBufferView);
        }
        if (type != nullptr && type->is_texture()) {
            return alignof(SIMDHostTextureView);
        }
        return type == nullptr ? 1u : type->alignment();
    }

    [[nodiscard]] static uint32_t _child_count(const Type *type) noexcept {
        if (type->is_vector() || type->is_matrix() || type->is_array()) {
            return type->dimension();
        }
        if (type->is_structure()) {
            return static_cast<uint32_t>(type->members().size());
        }
        return 0u;
    }

    [[nodiscard]] static const Type *_child_type(
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

    [[nodiscard]] static size_t _child_offset(
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

    [[nodiscard]] ::llvm::Type *_data_type(
        const Type *type, bool varying) {
        auto *result = _layout.expression_type(schedule::Value{
            .value_class = varying ? schedule::ValueClass::varying :
                                     schedule::ValueClass::warp_uniform,
            .type = type,
        });
        if (result == nullptr) { _fail(_layout.error()); }
        return result;
    }

    [[nodiscard]] ::llvm::Value *_extract_child(
        ::llvm::Value *aggregate, const Type *type, uint32_t index,
        bool varying) {
        if (type->is_vector() && !varying) {
            return _builder.CreateExtractElement(aggregate, index);
        }
        return _builder.CreateExtractValue(aggregate, {index});
    }

    [[nodiscard]] ::llvm::Value *_insert_child(
        ::llvm::Value *aggregate, ::llvm::Value *child,
        const Type *type, uint32_t index, bool varying) {
        if (type->is_vector() && !varying) {
            return _builder.CreateInsertElement(aggregate, child, index);
        }
        return _builder.CreateInsertValue(aggregate, child, {index});
    }

    [[nodiscard]] ::llvm::Value *_assemble(
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

    [[nodiscard]] ::llvm::Value *_splat_data(
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

    [[nodiscard]] ::llvm::Value *_extract_lane(
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

    [[nodiscard]] ::llvm::Value *_masked_merge(
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

    void _preflight_edge(const schedule::ControlEdge &edge,
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

    void _preflight() {
        if (_width == 0u || _width > 128u) {
            _fail("LLVM packet specialization width must be in [1, 128]");
            return;
        }
        if (_source.logical_warp_width() != 0u &&
            _source.logical_warp_width() != _width) {
            _fail("Schedule IR warp width does not match LLVM specialization width");
            return;
        }
        auto verification = schedule::verify(_source);
        if (!verification.succeeded()) {
            _fail("cannot lower invalid Schedule IR: " +
                  verification.errors.front().message);
            return;
        }
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
                if (argument_tag == xir::DerivedArgumentTag::REFERENCE ||
                    (argument_tag == xir::DerivedArgumentTag::VALUE &&
                     !_is_data(value.type)) ||
                    (argument_tag == xir::DerivedArgumentTag::RESOURCE &&
                     (value.type == nullptr ||
                      (!value.type->is_buffer() &&
                       !value.type->is_texture())))) {
                    _fail("packet ABI supports data, buffer, and texture arguments only");
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
                       !value.type->is_texture()) {
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
                    instruction.opcode != schedule::Opcode::resource_query &&
                    instruction.opcode != schedule::Opcode::resource_read &&
                    instruction.opcode != schedule::Opcode::resource_write &&
                    instruction.opcode != schedule::Opcode::atomic &&
                    instruction.opcode != schedule::Opcode::warp_collective) {
                    _fail("LLVM packet codegen encountered an unsupported Schedule IR instruction");
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
                        std::is_same_v<T, schedule::JoinTerminator>) {
                        auto *point = _source.convergence(
                            terminator.convergence);
                        if (point == nullptr) {
                            _fail("join references an invalid convergence point");
                        }
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

    [[nodiscard]] ::llvm::Constant *_scalar_constant(
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

    [[nodiscard]] ::llvm::Value *_lane_ids() {
        std::vector<::llvm::Constant *> lanes;
        lanes.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            lanes.emplace_back(_builder.getInt32(lane));
        }
        return ::llvm::ConstantVector::get(lanes);
    }

    [[nodiscard]] ::llvm::Constant *_constant_data(
        const Type *type, const std::byte *bytes, size_t offset = 0u) {
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

    [[nodiscard]] ::llvm::Value *_byte_pointer(
        ::llvm::Value *base, size_t offset) {
        return offset == 0u ? base : _builder.CreateGEP(
            _builder.getInt8Ty(), base, _builder.getInt64(offset));
    }

    [[nodiscard]] ::llvm::Value *_load_uniform_data(
        ::llvm::Value *base, const Type *type, size_t offset = 0u) {
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

    [[nodiscard]] ::llvm::Value *_load_buffer_view(
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

    [[nodiscard]] ::llvm::Value *_load_texture_view(
        ::llvm::Value *base) {
        auto &context = _module.getContext();
        auto *pointer_type = ::llvm::PointerType::getUnqual(context);
        auto *type = ::llvm::StructType::get(
            context,
            {pointer_type, pointer_type, pointer_type, pointer_type,
             pointer_type, pointer_type,
             _builder.getInt32Ty(), _builder.getInt32Ty()});
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
        return result;
    }

    [[nodiscard]] ::llvm::Value *_load_launch_u32(size_t offset) {
        auto *pointer = _byte_pointer(_launch_config, offset);
        auto *load = _builder.CreateLoad(_builder.getInt32Ty(), pointer);
        load->setAlignment(::llvm::Align{alignof(uint32_t)});
        return load;
    }

    void _ensure_launch_vectors() {
        if (_block_size[0u] != nullptr || _failed()) { return; }
        for (auto i = uint32_t{0u}; i < 3u; i++) {
            _block_id[i] = _load_launch_u32(
                offsetof(SIMDPacketLaunchConfig, block_id) +
                sizeof(uint32_t) * i);
            _dispatch_size[i] = _load_launch_u32(
                offsetof(SIMDPacketLaunchConfig, dispatch_size) +
                sizeof(uint32_t) * i);
            _block_size[i] = _load_launch_u32(
                offsetof(SIMDPacketLaunchConfig, block_size) +
                sizeof(uint32_t) * i);
        }
        auto *first = _load_launch_u32(
            offsetof(SIMDPacketLaunchConfig, thread_index));
        _linear_thread_indices = _builder.CreateAdd(
            _builder.CreateVectorSplat(_width, first), _lane_ids());
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
        for (auto i = uint32_t{0u}; i < 3u; i++) {
            auto *base = _builder.CreateMul(
                _block_id[i], _block_size[i]);
            _dispatch_id[i] = _builder.CreateAdd(
                _builder.CreateVectorSplat(_width, base),
                _thread_id[i]);
        }
    }

    [[nodiscard]] ::llvm::Value *_triplet(
        const Type *type, const std::array<::llvm::Value *, 3u> &values,
        bool varying) {
        return _assemble(type, varying, [&](uint32_t i) {
            return values[i];
        });
    }

    [[nodiscard]] ::llvm::Value *_special_register(
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

    void _create_external_values() {
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
                    llvm_value = tag == xir::DerivedArgumentTag::RESOURCE ?
                        value.type->is_buffer() ?
                            _load_buffer_view(pointer) :
                            _load_texture_view(pointer) :
                        _load_uniform_data(pointer, value.type);
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

    [[nodiscard]] ::llvm::Value *_load_value(schedule::ValueId id) {
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
            if (value->value_class == schedule::ValueClass::cohort_uniform) {
                auto *first = _collectives.first_active_lane(
                    _builder, _active_mask);
                auto *any = _builder.CreateOrReduce(_active_mask);
                auto *safe = _builder.CreateSelect(
                    any, first, _builder.getInt32(0u));
                return _extract_lane(state, value->type, safe);
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
            if (value->value_class == schedule::ValueClass::cohort_uniform) {
                auto *first = _collectives.first_active_lane(
                    _builder, _active_mask);
                auto *any = _builder.CreateOrReduce(_active_mask);
                auto *safe = _builder.CreateSelect(
                    any, first, _builder.getInt32(0u));
                return _extract_lane(state, value->type, safe);
            }
            return state;
        }
        _fail("instruction value '" + value->name + "' (#" +
              std::to_string(id.value) +
              ") is not available in the current Schedule IR block");
        return nullptr;
    }

    [[nodiscard]] ::llvm::Value *_as_lane_vector(
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

    [[nodiscard]] ::llvm::Value *_select_data(
        ::llvm::Value *condition, ::llvm::Value *true_value,
        ::llvm::Value *false_value, const Type *type, bool varying) {
        if (_is_scalar_data(type)) {
            return _builder.CreateSelect(
                condition, true_value, false_value);
        }
        return _assemble(type, varying, [&](uint32_t i) {
            return _select_data(
                condition,
                _extract_child(true_value, type, i, varying),
                _extract_child(false_value, type, i, varying),
                _child_type(type, i), varying);
        });
    }

    using UnaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, const Type *)>;
    using BinaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, ::llvm::Value *, const Type *, const Type *)>;
    using TernaryLeaf = std::function<::llvm::Value *(
        ::llvm::Value *, ::llvm::Value *, ::llvm::Value *,
        const Type *, const Type *, const Type *)>;

    [[nodiscard]] ::llvm::Value *_componentwise_unary(
        const Type *result_type, ::llvm::Value *operand,
        const Type *operand_type, bool varying, const UnaryLeaf &leaf) {
        if (_is_scalar_data(result_type)) {
            return leaf(operand, operand_type);
        }
        return _assemble(result_type, varying, [&](uint32_t i) {
            auto scalar_operand = _is_scalar_data(operand_type);
            auto *child_operand_type = scalar_operand ? operand_type :
                                                      _child_type(operand_type, i);
            auto *child_operand = scalar_operand ? operand :
                _extract_child(operand, operand_type, i, varying);
            return _componentwise_unary(
                _child_type(result_type, i), child_operand,
                child_operand_type, varying, leaf);
        });
    }

    [[nodiscard]] ::llvm::Value *_componentwise_binary(
        const Type *result_type, ::llvm::Value *lhs, const Type *lhs_type,
        ::llvm::Value *rhs, const Type *rhs_type, bool varying,
        const BinaryLeaf &leaf) {
        if (_is_scalar_data(result_type)) {
            return leaf(lhs, rhs, lhs_type, rhs_type);
        }
        return _assemble(result_type, varying, [&](uint32_t i) {
            auto lhs_scalar = _is_scalar_data(lhs_type);
            auto rhs_scalar = _is_scalar_data(rhs_type);
            auto *lhs_child_type = lhs_scalar ? lhs_type :
                                               _child_type(lhs_type, i);
            auto *rhs_child_type = rhs_scalar ? rhs_type :
                                               _child_type(rhs_type, i);
            auto *lhs_child = lhs_scalar ? lhs :
                _extract_child(lhs, lhs_type, i, varying);
            auto *rhs_child = rhs_scalar ? rhs :
                _extract_child(rhs, rhs_type, i, varying);
            return _componentwise_binary(
                _child_type(result_type, i), lhs_child, lhs_child_type,
                rhs_child, rhs_child_type, varying, leaf);
        });
    }

    [[nodiscard]] ::llvm::Value *_componentwise_ternary(
        const Type *result_type,
        ::llvm::Value *a, const Type *a_type,
        ::llvm::Value *b, const Type *b_type,
        ::llvm::Value *c, const Type *c_type, bool varying,
        const TernaryLeaf &leaf) {
        if (_is_scalar_data(result_type)) {
            return leaf(a, b, c, a_type, b_type, c_type);
        }
        return _assemble(result_type, varying, [&](uint32_t i) {
            auto a_scalar = _is_scalar_data(a_type);
            auto b_scalar = _is_scalar_data(b_type);
            auto c_scalar = _is_scalar_data(c_type);
            auto *a_child_type = a_scalar ? a_type :
                                                _child_type(a_type, i);
            auto *b_child_type = b_scalar ? b_type :
                                                _child_type(b_type, i);
            auto *c_child_type = c_scalar ? c_type :
                                                _child_type(c_type, i);
            auto *a_child = a_scalar ? a :
                _extract_child(a, a_type, i, varying);
            auto *b_child = b_scalar ? b :
                _extract_child(b, b_type, i, varying);
            auto *c_child = c_scalar ? c :
                _extract_child(c, c_type, i, varying);
            return _componentwise_ternary(
                _child_type(result_type, i),
                a_child, a_child_type, b_child, b_child_type,
                c_child, c_child_type, varying, leaf);
        });
    }

    [[nodiscard]] ::llvm::Value *_componentwise_varying_to_uniform(
        const Type *result_type, ::llvm::Value *operand,
        const Type *operand_type, const UnaryLeaf &leaf) {
        if (_is_scalar_data(result_type)) {
            if (!_is_scalar_data(operand_type)) {
                _fail("aggregate warp collective result shape does not match its operand");
                return nullptr;
            }
            return leaf(operand, operand_type);
        }
        if (_is_scalar_data(operand_type) ||
            _child_count(result_type) != _child_count(operand_type)) {
            _fail("aggregate warp collective result shape does not match its operand");
            return nullptr;
        }
        return _assemble(result_type, false, [&](uint32_t i) {
            return _componentwise_varying_to_uniform(
                _child_type(result_type, i),
                _extract_child(operand, operand_type, i, true),
                _child_type(operand_type, i), leaf);
        });
    }

    [[nodiscard]] static std::optional<uint64_t> _constant_index(
        ::llvm::Value *value) noexcept {
        if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(value)) {
            return integer->getZExtValue();
        }
        if (auto *constant = ::llvm::dyn_cast<::llvm::Constant>(value)) {
            if (auto *splat = constant->getSplatValue()) {
                if (auto *integer = ::llvm::dyn_cast<::llvm::ConstantInt>(splat)) {
                    return integer->getZExtValue();
                }
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] ::llvm::Value *_index_constant_like(
        ::llvm::Value *index, uint64_t value) {
        auto *type = index->getType();
        if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(type)) {
            auto *element = ::llvm::cast<::llvm::IntegerType>(
                vector->getElementType());
            return _builder.CreateVectorSplat(
                vector->getElementCount(),
                ::llvm::ConstantInt::get(element, value));
        }
        return ::llvm::ConstantInt::get(
            ::llvm::cast<::llvm::IntegerType>(type), value);
    }

    [[nodiscard]] ::llvm::Value *_extract_indexed(
        ::llvm::Value *aggregate, const Type *type,
        const std::vector<::llvm::Value *> &indices, size_t depth,
        bool varying) {
        if (depth == indices.size()) { return aggregate; }
        auto *index = indices[depth];
        auto count = _child_count(type);
        if (count == 0u) {
            _fail("aggregate extract has too many indices");
            return nullptr;
        }
        ::llvm::Value *selected = nullptr;
        auto *child_type = _child_type(type, 0u);
        if (auto constant = _constant_index(index)) {
            if (*constant >= count) {
                _fail("aggregate extract index is out of range");
                return nullptr;
            }
            selected = _extract_child(
                aggregate, type, static_cast<uint32_t>(*constant), varying);
            child_type = _child_type(
                type, static_cast<uint32_t>(*constant));
        } else {
            if (type->is_structure()) {
                _fail("dynamic structure member extraction is invalid");
                return nullptr;
            }
            selected = _extract_child(aggregate, type, 0u, varying);
            for (auto i = uint32_t{1u}; i < count; i++) {
                auto *candidate = _extract_child(
                    aggregate, type, i, varying);
                auto *condition = _builder.CreateICmpEQ(
                    index, _index_constant_like(index, i));
                selected = _select_data(
                    condition, candidate, selected, child_type, varying);
            }
        }
        return _extract_indexed(
            selected, child_type, indices, depth + 1u, varying);
    }

    [[nodiscard]] ::llvm::Value *_insert_indexed(
        ::llvm::Value *aggregate, const Type *type,
        ::llvm::Value *replacement,
        const std::vector<::llvm::Value *> &indices, size_t depth,
        bool varying) {
        if (depth == indices.size()) { return replacement; }
        auto *index = indices[depth];
        auto count = _child_count(type);
        if (count == 0u) {
            _fail("aggregate insert has too many indices");
            return nullptr;
        }
        if (auto constant = _constant_index(index)) {
            if (*constant >= count) {
                _fail("aggregate insert index is out of range");
                return nullptr;
            }
            auto i = static_cast<uint32_t>(*constant);
            auto *child_type = _child_type(type, i);
            auto *old_child = _extract_child(
                aggregate, type, i, varying);
            auto *new_child = _insert_indexed(
                old_child, child_type, replacement,
                indices, depth + 1u, varying);
            return new_child == nullptr ? nullptr :
                _insert_child(aggregate, new_child, type, i, varying);
        }
        if (type->is_structure()) {
            _fail("dynamic structure member insertion is invalid");
            return nullptr;
        }
        auto *result = aggregate;
        for (auto i = uint32_t{0u}; i < count; i++) {
            auto *child_type = _child_type(type, i);
            auto *old_child = _extract_child(
                aggregate, type, i, varying);
            auto *updated = _insert_indexed(
                old_child, child_type, replacement,
                indices, depth + 1u, varying);
            if (updated == nullptr) { return nullptr; }
            auto *condition = _builder.CreateICmpEQ(
                index, _index_constant_like(index, i));
            auto *selected = _select_data(
                condition, updated, old_child, child_type, varying);
            result = _insert_child(
                result, selected, type, i, varying);
        }
        return result;
    }

    [[nodiscard]] ::llvm::Value *_aggregate_operation(
        const schedule::Value &result,
        const schedule::Instruction &instruction,
        const std::vector<::llvm::Value *> &operands, bool varying) {
        if (operands.size() != _child_count(result.type)) {
            _fail("aggregate construction operand count mismatch");
            return nullptr;
        }
        return _assemble(result.type, varying, [&](uint32_t i) {
            return operands[i];
        });
    }

    [[nodiscard]] ::llvm::Value *_arithmetic(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op) {
            _fail("arithmetic instruction is missing result or source operation");
            return nullptr;
        }
        auto *result = _source.value(*instruction.result);
        if (result == nullptr || !_is_data(result->type)) {
            _fail("arithmetic requires a supported Luisa data result type");
            return nullptr;
        }
        auto varying = result->value_class == schedule::ValueClass::varying;
        std::vector<::llvm::Value *> operands;
        std::vector<const Type *> operand_types;
        operands.reserve(instruction.operands.size());
        operand_types.reserve(instruction.operands.size());
        for (auto operand_id : instruction.operands) {
            auto *operand = _source.value(operand_id);
            auto *llvm_operand = _load_value(operand_id);
            if (varying) {
                llvm_operand = _as_lane_vector(llvm_operand, *operand);
            }
            if (llvm_operand == nullptr) { return nullptr; }
            operands.emplace_back(llvm_operand);
            operand_types.emplace_back(operand->type);
        }
        auto require = [&](size_t count) {
            if (operands.size() != count) {
                _fail("arithmetic operation has an invalid operand count");
                return false;
            }
            return true;
        };
        auto op = static_cast<xir::ArithmeticOp>(*instruction.source_op);
        if (op == xir::ArithmeticOp::AGGREGATE) {
            return _aggregate_operation(
                *result, instruction, operands, varying);
        }
        if (op == xir::ArithmeticOp::EXTRACT ||
            op == xir::ArithmeticOp::INSERT ||
            op == xir::ArithmeticOp::SHUFFLE) {
            if (operands.size() < 2u) {
                _fail("aggregate extraction requires an aggregate and indices");
                return nullptr;
            }
            if (op == xir::ArithmeticOp::SHUFFLE) {
                return _assemble(result->type, varying, [&](uint32_t i) {
                    std::vector<::llvm::Value *> index{operands[i + 1u]};
                    return _extract_indexed(
                        operands[0u], operand_types[0u], index, 0u,
                        varying);
                });
            }
            if (op == xir::ArithmeticOp::INSERT) {
                if (operands.size() < 3u) {
                    _fail("aggregate insertion requires a base, value, and indices");
                    return nullptr;
                }
                std::vector<::llvm::Value *> indices{
                    operands.begin() + 2, operands.end()};
                return _insert_indexed(
                    operands[0u], operand_types[0u], operands[1u],
                    indices, 0u, varying);
            }
            std::vector<::llvm::Value *> indices{
                operands.begin() + 1, operands.end()};
            return _extract_indexed(
                operands[0u], operand_types[0u], indices, 0u, varying);
        }

        auto unary = [&](const UnaryLeaf &leaf) -> ::llvm::Value * {
            if (!require(1u)) { return nullptr; }
            return _componentwise_unary(
                result->type, operands[0u], operand_types[0u],
                varying, leaf);
        };
        auto binary = [&](const BinaryLeaf &leaf) -> ::llvm::Value * {
            if (!require(2u)) { return nullptr; }
            return _componentwise_binary(
                result->type, operands[0u], operand_types[0u],
                operands[1u], operand_types[1u], varying, leaf);
        };
        auto ternary = [&](const TernaryLeaf &leaf) -> ::llvm::Value * {
            if (!require(3u)) { return nullptr; }
            return _componentwise_ternary(
                result->type,
                operands[0u], operand_types[0u],
                operands[1u], operand_types[1u],
                operands[2u], operand_types[2u], varying, leaf);
        };
        auto intrinsic = [&](::llvm::Intrinsic::ID id,
                             std::initializer_list<::llvm::Value *> args) {
            std::vector<::llvm::Value *> values{args};
            std::array<::llvm::Type *, 1u> overloads{
                values.front()->getType()};
#if LLVM_VERSION_MAJOR >= 22
            auto *function = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            auto *function = ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, id, overloads);
            return _builder.CreateCall(function, values);
        };
        auto unary_intrinsic = [&](::llvm::Intrinsic::ID id) {
            return unary([&](::llvm::Value *value, const Type *) {
                return intrinsic(id, {value});
            });
        };
        auto binary_intrinsic = [&](::llvm::Intrinsic::ID id) {
            return binary([&](::llvm::Value *lhs, ::llvm::Value *rhs,
                              const Type *, const Type *) {
                return intrinsic(id, {lhs, rhs});
            });
        };
        auto float_constant_like = [&](::llvm::Value *value, double x) {
            auto *scalar = ::llvm::ConstantFP::get(
                value->getType()->getScalarType(), x);
            if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(
                    value->getType())) {
                return static_cast<::llvm::Constant *>(
                    ::llvm::ConstantVector::getSplat(
                        vector->getElementCount(), scalar));
            }
            return scalar;
        };
        auto minmax_leaf = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                               const Type *type, bool maximum)
            -> ::llvm::Value * {
            if (type->is_float16() || type->is_float32() ||
                type->is_float64()) {
                return intrinsic(
                    maximum ? ::llvm::Intrinsic::maxnum :
                              ::llvm::Intrinsic::minnum,
                    {lhs, rhs});
            }
            auto predicate = maximum ?
                type->is_int() ? ::llvm::CmpInst::ICMP_SGT :
                                 ::llvm::CmpInst::ICMP_UGT :
                type->is_int() ? ::llvm::CmpInst::ICMP_SLT :
                                 ::llvm::CmpInst::ICMP_ULT;
            return _builder.CreateSelect(
                _builder.CreateICmp(predicate, lhs, rhs), lhs, rhs);
        };
        auto dot = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                       const Type *type) -> ::llvm::Value * {
            if (!type->is_vector()) {
                _fail("dot product requires vector operands");
                return nullptr;
            }
            ::llvm::Value *sum = nullptr;
            for (auto i = uint32_t{0u}; i < type->dimension(); i++) {
                auto *a = _extract_child(lhs, type, i, varying);
                auto *b = _extract_child(rhs, type, i, varying);
                auto *product = _builder.CreateFMul(a, b);
                sum = sum == nullptr ? product :
                                       _builder.CreateFAdd(sum, product);
            }
            return sum;
        };
        auto binary_leaf = [&](::llvm::Value *lhs, ::llvm::Value *rhs,
                               const Type *lhs_type,
                               const Type *) -> ::llvm::Value * {
            auto is_float = lhs_type->is_float16() ||
                            lhs_type->is_float32() ||
                            lhs_type->is_float64();
            auto is_signed = lhs_type->is_int();
            switch (op) {
                case xir::ArithmeticOp::BINARY_ADD:
                case xir::ArithmeticOp::MATRIX_COMP_ADD:
                    return is_float ? _builder.CreateFAdd(lhs, rhs) :
                                      _builder.CreateAdd(lhs, rhs);
                case xir::ArithmeticOp::BINARY_SUB:
                case xir::ArithmeticOp::MATRIX_COMP_SUB:
                    return is_float ? _builder.CreateFSub(lhs, rhs) :
                                      _builder.CreateSub(lhs, rhs);
                case xir::ArithmeticOp::BINARY_MUL:
                case xir::ArithmeticOp::MATRIX_COMP_MUL:
                    return is_float ? _builder.CreateFMul(lhs, rhs) :
                                      _builder.CreateMul(lhs, rhs);
                case xir::ArithmeticOp::BINARY_DIV:
                case xir::ArithmeticOp::MATRIX_COMP_DIV:
                    return is_float ? _builder.CreateFDiv(lhs, rhs) :
                           is_signed ? _builder.CreateSDiv(lhs, rhs) :
                                       _builder.CreateUDiv(lhs, rhs);
                case xir::ArithmeticOp::BINARY_MOD:
                    return is_float ? _builder.CreateFRem(lhs, rhs) :
                           is_signed ? _builder.CreateSRem(lhs, rhs) :
                                       _builder.CreateURem(lhs, rhs);
                case xir::ArithmeticOp::BINARY_BIT_AND:
                    return _builder.CreateAnd(lhs, rhs);
                case xir::ArithmeticOp::BINARY_BIT_OR:
                    return _builder.CreateOr(lhs, rhs);
                case xir::ArithmeticOp::BINARY_BIT_XOR:
                    return _builder.CreateXor(lhs, rhs);
                case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
                    return _builder.CreateShl(lhs, rhs);
                case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
                    return is_signed ? _builder.CreateAShr(lhs, rhs) :
                                       _builder.CreateLShr(lhs, rhs);
                case xir::ArithmeticOp::BINARY_LESS:
                case xir::ArithmeticOp::BINARY_GREATER:
                case xir::ArithmeticOp::BINARY_LESS_EQUAL:
                case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
                case xir::ArithmeticOp::BINARY_EQUAL:
                case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
                    if (is_float) {
                        auto predicate = ::llvm::CmpInst::FCMP_FALSE;
                        switch (op) {
                            case xir::ArithmeticOp::BINARY_LESS: predicate = ::llvm::CmpInst::FCMP_OLT; break;
                            case xir::ArithmeticOp::BINARY_GREATER: predicate = ::llvm::CmpInst::FCMP_OGT; break;
                            case xir::ArithmeticOp::BINARY_LESS_EQUAL: predicate = ::llvm::CmpInst::FCMP_OLE; break;
                            case xir::ArithmeticOp::BINARY_GREATER_EQUAL: predicate = ::llvm::CmpInst::FCMP_OGE; break;
                            case xir::ArithmeticOp::BINARY_EQUAL: predicate = ::llvm::CmpInst::FCMP_OEQ; break;
                            case xir::ArithmeticOp::BINARY_NOT_EQUAL: predicate = ::llvm::CmpInst::FCMP_UNE; break;
                            default: break;
                        }
                        return _builder.CreateFCmp(predicate, lhs, rhs);
                    }
                    auto predicate = ::llvm::CmpInst::BAD_ICMP_PREDICATE;
                    switch (op) {
                        case xir::ArithmeticOp::BINARY_LESS: predicate = is_signed ? ::llvm::CmpInst::ICMP_SLT : ::llvm::CmpInst::ICMP_ULT; break;
                        case xir::ArithmeticOp::BINARY_GREATER: predicate = is_signed ? ::llvm::CmpInst::ICMP_SGT : ::llvm::CmpInst::ICMP_UGT; break;
                        case xir::ArithmeticOp::BINARY_LESS_EQUAL: predicate = is_signed ? ::llvm::CmpInst::ICMP_SLE : ::llvm::CmpInst::ICMP_ULE; break;
                        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: predicate = is_signed ? ::llvm::CmpInst::ICMP_SGE : ::llvm::CmpInst::ICMP_UGE; break;
                        case xir::ArithmeticOp::BINARY_EQUAL: predicate = ::llvm::CmpInst::ICMP_EQ; break;
                        case xir::ArithmeticOp::BINARY_NOT_EQUAL: predicate = ::llvm::CmpInst::ICMP_NE; break;
                        default: break;
                    }
                    return _builder.CreateICmp(predicate, lhs, rhs);
                }
                default: return nullptr;
            }
        };
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
            case xir::ArithmeticOp::MATRIX_COMP_NEG:
                return unary([&](::llvm::Value *value, const Type *type) {
                    return type->is_float16() || type->is_float32() ||
                                   type->is_float64() ?
                               _builder.CreateFNeg(value) :
                               _builder.CreateNeg(value);
                });
            case xir::ArithmeticOp::UNARY_BIT_NOT:
                return unary([&](::llvm::Value *value, const Type *) {
                    return _builder.CreateNot(value);
                });
            case xir::ArithmeticOp::BINARY_ADD:
            case xir::ArithmeticOp::MATRIX_COMP_ADD:
            case xir::ArithmeticOp::BINARY_SUB:
            case xir::ArithmeticOp::MATRIX_COMP_SUB:
            case xir::ArithmeticOp::BINARY_MUL:
            case xir::ArithmeticOp::MATRIX_COMP_MUL:
            case xir::ArithmeticOp::BINARY_DIV:
            case xir::ArithmeticOp::MATRIX_COMP_DIV:
            case xir::ArithmeticOp::BINARY_MOD:
            case xir::ArithmeticOp::BINARY_BIT_AND:
            case xir::ArithmeticOp::BINARY_BIT_OR:
            case xir::ArithmeticOp::BINARY_BIT_XOR:
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL:
                return binary(binary_leaf);
            case xir::ArithmeticOp::SELECT:
                return ternary(
                    [&](::llvm::Value *if_false,
                        ::llvm::Value *if_true,
                        ::llvm::Value *condition,
                        const Type *, const Type *, const Type *) {
                        return _builder.CreateSelect(
                            condition, if_true, if_false);
                    });
            case xir::ArithmeticOp::CLAMP:
                return ternary(
                    [&](::llvm::Value *value, ::llvm::Value *low,
                        ::llvm::Value *high, const Type *type,
                        const Type *, const Type *) {
                        return minmax_leaf(
                            minmax_leaf(value, low, type, true),
                            high, type, false);
                    });
            case xir::ArithmeticOp::SATURATE:
                return unary([&](::llvm::Value *value, const Type *type) {
                    auto *zero = float_constant_like(value, 0.0);
                    auto *one = float_constant_like(value, 1.0);
                    return minmax_leaf(
                        minmax_leaf(value, zero, type, true),
                        one, type, false);
                });
            case xir::ArithmeticOp::LERP:
                return ternary(
                    [&](::llvm::Value *a, ::llvm::Value *b,
                        ::llvm::Value *t, const Type *,
                        const Type *, const Type *) {
                        return _builder.CreateFAdd(
                            a, _builder.CreateFMul(
                                   _builder.CreateFSub(b, a), t));
                    });
            case xir::ArithmeticOp::ABS:
                return unary([&](::llvm::Value *value, const Type *type)
                                 -> ::llvm::Value * {
                    if (type->is_float16() || type->is_float32() ||
                        type->is_float64()) {
                        return intrinsic(::llvm::Intrinsic::fabs, {value});
                    }
                    if (!type->is_int()) { return value; }
                    auto *zero = ::llvm::Constant::getNullValue(
                        value->getType());
                    return _builder.CreateSelect(
                        _builder.CreateICmpSLT(value, zero),
                        _builder.CreateNeg(value), value);
                });
            case xir::ArithmeticOp::MIN:
            case xir::ArithmeticOp::MAX:
                return binary([&](::llvm::Value *lhs, ::llvm::Value *rhs,
                                  const Type *type, const Type *) {
                    return minmax_leaf(
                        lhs, rhs, type,
                        op == xir::ArithmeticOp::MAX);
                });
            case xir::ArithmeticOp::ISNAN:
                return unary([&](::llvm::Value *value, const Type *) {
                    return _builder.CreateFCmpUNO(value, value);
                });
            case xir::ArithmeticOp::ISINF:
                return unary([&](::llvm::Value *value, const Type *) {
                    auto *absolute = intrinsic(
                        ::llvm::Intrinsic::fabs, {value});
                    auto *infinity = ::llvm::ConstantFP::getInfinity(
                        value->getType()->getScalarType());
                    if (auto *vector = ::llvm::dyn_cast<::llvm::VectorType>(
                            value->getType())) {
                        infinity = ::llvm::ConstantVector::getSplat(
                            vector->getElementCount(), infinity);
                    }
                    return _builder.CreateFCmpOEQ(absolute, infinity);
                });
            case xir::ArithmeticOp::ACOS:
                return unary_intrinsic(::llvm::Intrinsic::acos);
            case xir::ArithmeticOp::ASIN:
                return unary_intrinsic(::llvm::Intrinsic::asin);
            case xir::ArithmeticOp::ATAN:
                return unary_intrinsic(::llvm::Intrinsic::atan);
            case xir::ArithmeticOp::ATAN2:
                return binary_intrinsic(::llvm::Intrinsic::atan2);
            case xir::ArithmeticOp::COS:
                return unary_intrinsic(::llvm::Intrinsic::cos);
            case xir::ArithmeticOp::SIN:
                return unary_intrinsic(::llvm::Intrinsic::sin);
            case xir::ArithmeticOp::TAN:
                return unary_intrinsic(::llvm::Intrinsic::tan);
            case xir::ArithmeticOp::COSH:
                return unary_intrinsic(::llvm::Intrinsic::cosh);
            case xir::ArithmeticOp::SINH:
                return unary_intrinsic(::llvm::Intrinsic::sinh);
            case xir::ArithmeticOp::TANH:
                return unary_intrinsic(::llvm::Intrinsic::tanh);
            case xir::ArithmeticOp::EXP:
                return unary_intrinsic(::llvm::Intrinsic::exp);
            case xir::ArithmeticOp::EXP2:
                return unary_intrinsic(::llvm::Intrinsic::exp2);
            case xir::ArithmeticOp::EXP10:
                return unary_intrinsic(::llvm::Intrinsic::exp10);
            case xir::ArithmeticOp::LOG:
                return unary_intrinsic(::llvm::Intrinsic::log);
            case xir::ArithmeticOp::LOG2:
                return unary_intrinsic(::llvm::Intrinsic::log2);
            case xir::ArithmeticOp::LOG10:
                return unary_intrinsic(::llvm::Intrinsic::log10);
            case xir::ArithmeticOp::POW:
                return binary_intrinsic(::llvm::Intrinsic::pow);
            case xir::ArithmeticOp::SQRT:
                return unary_intrinsic(::llvm::Intrinsic::sqrt);
            case xir::ArithmeticOp::RSQRT:
                return unary([&](::llvm::Value *value, const Type *) {
                    auto *root = intrinsic(
                        ::llvm::Intrinsic::sqrt, {value});
                    return _builder.CreateFDiv(
                        float_constant_like(value, 1.0), root);
                });
            case xir::ArithmeticOp::CEIL:
                return unary_intrinsic(::llvm::Intrinsic::ceil);
            case xir::ArithmeticOp::FLOOR:
                return unary_intrinsic(::llvm::Intrinsic::floor);
            case xir::ArithmeticOp::TRUNC:
                return unary_intrinsic(::llvm::Intrinsic::trunc);
            case xir::ArithmeticOp::ROUND:
                return unary_intrinsic(::llvm::Intrinsic::round);
            case xir::ArithmeticOp::RINT:
                return unary_intrinsic(::llvm::Intrinsic::rint);
            case xir::ArithmeticOp::FRACT:
                return unary([&](::llvm::Value *value, const Type *) {
                    return _builder.CreateFSub(
                        value, intrinsic(
                                   ::llvm::Intrinsic::floor, {value}));
                });
            case xir::ArithmeticOp::FMA:
                return ternary(
                    [&](::llvm::Value *a, ::llvm::Value *b,
                        ::llvm::Value *c, const Type *,
                        const Type *, const Type *) {
                        return intrinsic(
                            ::llvm::Intrinsic::fma, {a, b, c});
                    });
            case xir::ArithmeticOp::COPYSIGN:
                return binary_intrinsic(::llvm::Intrinsic::copysign);
            case xir::ArithmeticOp::DOT:
                if (!require(2u)) { return nullptr; }
                return dot(
                    operands[0u], operands[1u], operand_types[0u]);
            case xir::ArithmeticOp::LENGTH_SQUARED:
                if (!require(1u)) { return nullptr; }
                return dot(
                    operands[0u], operands[0u], operand_types[0u]);
            case xir::ArithmeticOp::LENGTH: {
                if (!require(1u)) { return nullptr; }
                auto *squared = dot(
                    operands[0u], operands[0u], operand_types[0u]);
                return squared == nullptr ? nullptr :
                    intrinsic(::llvm::Intrinsic::sqrt, {squared});
            }
            case xir::ArithmeticOp::NORMALIZE: {
                if (!require(1u) || !operand_types[0u]->is_vector()) {
                    return nullptr;
                }
                auto *squared = dot(
                    operands[0u], operands[0u], operand_types[0u]);
                if (squared == nullptr) { return nullptr; }
                auto *length = intrinsic(
                    ::llvm::Intrinsic::sqrt, {squared});
                return _componentwise_unary(
                    result->type, operands[0u], operand_types[0u],
                    varying,
                    [&](::llvm::Value *value, const Type *) {
                        return _builder.CreateFDiv(value, length);
                    });
            }
            case xir::ArithmeticOp::CROSS: {
                if (!require(2u) || !result->type->is_vector() ||
                    result->type->dimension() != 3u) {
                    return nullptr;
                }
                std::array<::llvm::Value *, 3u> a{};
                std::array<::llvm::Value *, 3u> b{};
                for (auto i = uint32_t{0u}; i < 3u; i++) {
                    a[i] = _extract_child(
                        operands[0u], operand_types[0u], i, varying);
                    b[i] = _extract_child(
                        operands[1u], operand_types[1u], i, varying);
                }
                return _assemble(result->type, varying, [&](uint32_t i) {
                    auto j = (i + 1u) % 3u;
                    auto k = (i + 2u) % 3u;
                    return _builder.CreateFSub(
                        _builder.CreateFMul(a[j], b[k]),
                        _builder.CreateFMul(a[k], b[j]));
                });
            }
            default:
                _fail("LLVM packet codegen does not implement arithmetic operation '" +
                      std::string{xir::to_string(op)} + "' yet");
                return nullptr;
        }
    }

    [[nodiscard]] ::llvm::Value *_cast(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op ||
            instruction.operands.size() != 1u) {
            _fail("cast instruction is malformed");
            return nullptr;
        }
        auto *result = _source.value(*instruction.result);
        auto *source = _source.value(instruction.operands.front());
        auto *value = _load_value(instruction.operands.front());
        if (result == nullptr || source == nullptr || value == nullptr ||
            !_is_data(result->type) || !_is_data(source->type)) {
            _fail("cast requires supported data types");
            return nullptr;
        }
        auto varying = result->value_class == schedule::ValueClass::varying;
        if (varying) { value = _as_lane_vector(value, *source); }
        auto op = static_cast<xir::CastOp>(*instruction.source_op);
        return _componentwise_unary(
            result->type, value, source->type, varying,
            [&](::llvm::Value *scalar, const Type *source_type) {
                auto *destination_type = result->type;
                while (!_is_scalar_data(destination_type)) {
                    destination_type = destination_type->element();
                }
                if (op == xir::CastOp::BITWISE_CAST) {
                    return _builder.CreateBitCast(
                        scalar,
                        scalar->getType()->isVectorTy() ?
                            ::llvm::FixedVectorType::get(
                                _data_type(destination_type, false),
                                _width) :
                            _data_type(destination_type, false));
                }
                auto destination_is_float =
                    destination_type->is_float16() ||
                    destination_type->is_float32() ||
                    destination_type->is_float64();
                auto source_is_float = source_type->is_float16() ||
                                       source_type->is_float32() ||
                                       source_type->is_float64();
                auto *destination = scalar->getType()->isVectorTy() ?
                    static_cast<::llvm::Type *>(::llvm::FixedVectorType::get(
                        _data_type(destination_type, false), _width)) :
                    _data_type(destination_type, false);
                if (destination_type->is_bool()) {
                    auto *zero = ::llvm::Constant::getNullValue(
                        scalar->getType());
                    return source_is_float ?
                        _builder.CreateFCmpUNE(scalar, zero) :
                        _builder.CreateICmpNE(scalar, zero);
                }
                if (source_type->is_bool()) {
                    return destination_is_float ?
                        _builder.CreateUIToFP(scalar, destination) :
                        _builder.CreateZExtOrTrunc(scalar, destination);
                }
                if (source_is_float && destination_is_float) {
                    return _builder.CreateFPCast(scalar, destination);
                }
                if (source_is_float) {
                    return destination_type->is_int() ?
                        _builder.CreateFPToSI(scalar, destination) :
                        _builder.CreateFPToUI(scalar, destination);
                }
                if (destination_is_float) {
                    return source_type->is_int() ?
                        _builder.CreateSIToFP(scalar, destination) :
                        _builder.CreateUIToFP(scalar, destination);
                }
                return source_type->is_int() ?
                    _builder.CreateSExtOrTrunc(scalar, destination) :
                    _builder.CreateZExtOrTrunc(scalar, destination);
            });
    }

    [[nodiscard]] ::llvm::Value *_lane_offsets(
        ::llvm::Value *index, uint64_t stride) {
        auto *i64_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt64Ty(), _width);
        auto *extended = _builder.CreateZExtOrTrunc(index, i64_lanes);
        return stride == 1u ? extended : _builder.CreateMul(
            extended,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(stride)));
    }

    [[nodiscard]] ::llvm::Value *_atomic(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op) {
            _fail("atomic instruction is missing a result or operation");
            return nullptr;
        }
        auto op = static_cast<xir::AtomicOp>(*instruction.source_op);
        auto value_count = xir::atomic_op_value_count(op);
        if (instruction.operands.size() < 2u + value_count) {
            _fail("atomic instruction has an invalid operand count");
            return nullptr;
        }
        auto index_count =
            instruction.operands.size() - 1u - value_count;
        auto *result = _source.value(*instruction.result);
        auto *buffer_value = _source.value(instruction.operands[0u]);
        auto *index_value = _source.value(instruction.operands[1u]);
        auto *buffer = _load_value(instruction.operands[0u]);
        auto *index = index_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[1u]), *index_value);
        if (result == nullptr || buffer_value == nullptr ||
            buffer == nullptr || index == nullptr ||
            !buffer_value->type->is_buffer() || index_count != 1u ||
            !_is_scalar_data(result->type) ||
            buffer_value->type->element() != result->type) {
            _fail("LLVM packet codegen currently supports scalar direct-buffer atomics");
            return nullptr;
        }
        std::vector<::llvm::Value *> values;
        values.reserve(value_count);
        for (auto i = size_t{0u}; i < value_count; i++) {
            auto operand_id = instruction.operands[
                instruction.operands.size() - value_count + i];
            auto *operand = _source.value(operand_id);
            auto *value = operand == nullptr ? nullptr :
                _as_lane_vector(_load_value(operand_id), *operand);
            if (value == nullptr) { return nullptr; }
            values.emplace_back(value);
        }

        auto *base = _builder.CreateExtractValue(buffer, {0u});
        auto *offsets = _lane_offsets(
            index, static_cast<uint64_t>(result->type->size()));
        auto *result_type = _layout.expression_type(*result);
        if (result_type == nullptr || !result_type->isVectorTy()) {
            _fail("buffer atomic result must be lane-varying");
            return nullptr;
        }
        ::llvm::Value *old_values =
            ::llvm::Constant::getNullValue(result_type);
        auto alignment = ::llvm::MaybeAlign{result->type->alignment()};
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            auto *before = _builder.GetInsertBlock();
            auto *active = _builder.CreateExtractElement(
                _active_mask, lane);
            auto *atomic_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "atomic.lane." + std::to_string(lane), _entry);
            auto *continue_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "atomic.continue." + std::to_string(lane), _entry);
            _builder.CreateCondBr(active, atomic_block, continue_block);

            _builder.SetInsertPoint(atomic_block);
            auto *offset = _builder.CreateExtractElement(offsets, lane);
            auto *pointer = _builder.CreateGEP(
                _builder.getInt8Ty(), base, offset);
            std::vector<::llvm::Value *> lane_values;
            lane_values.reserve(values.size());
            for (auto *value : values) {
                lane_values.emplace_back(
                    _builder.CreateExtractElement(value, lane));
            }
            ::llvm::Value *old = nullptr;
            if (op == xir::AtomicOp::COMPARE_EXCHANGE) {
                auto *expected = lane_values[0u];
                auto *desired = lane_values[1u];
                auto *value_type = expected->getType();
                auto floating = value_type->isFloatingPointTy();
                if (floating) {
                    auto *integer = ::llvm::IntegerType::get(
                        _module.getContext(),
                        value_type->getPrimitiveSizeInBits());
                    expected = _builder.CreateBitCast(expected, integer);
                    desired = _builder.CreateBitCast(desired, integer);
                }
                auto *pair = _builder.CreateAtomicCmpXchg(
                    pointer, expected, desired, alignment,
                    ::llvm::AtomicOrdering::Monotonic,
                    ::llvm::AtomicOrdering::Monotonic);
                old = _builder.CreateExtractValue(pair, {0u});
                if (floating) {
                    old = _builder.CreateBitCast(old, value_type);
                }
            } else {
                auto atomic_op = ::llvm::AtomicRMWInst::BAD_BINOP;
                auto floating = result->type->is_float16() ||
                                result->type->is_float32() ||
                                result->type->is_float64();
                auto signed_integer = result->type->is_int();
                switch (op) {
                    case xir::AtomicOp::EXCHANGE:
                        atomic_op = ::llvm::AtomicRMWInst::Xchg;
                        break;
                    case xir::AtomicOp::FETCH_ADD:
                        atomic_op = floating ?
                            ::llvm::AtomicRMWInst::FAdd :
                            ::llvm::AtomicRMWInst::Add;
                        break;
                    case xir::AtomicOp::FETCH_SUB:
                        atomic_op = floating ?
                            ::llvm::AtomicRMWInst::FSub :
                            ::llvm::AtomicRMWInst::Sub;
                        break;
                    case xir::AtomicOp::FETCH_AND:
                        atomic_op = ::llvm::AtomicRMWInst::And;
                        break;
                    case xir::AtomicOp::FETCH_OR:
                        atomic_op = ::llvm::AtomicRMWInst::Or;
                        break;
                    case xir::AtomicOp::FETCH_XOR:
                        atomic_op = ::llvm::AtomicRMWInst::Xor;
                        break;
                    case xir::AtomicOp::FETCH_MIN:
                        atomic_op = floating ?
                            ::llvm::AtomicRMWInst::FMin :
                            signed_integer ?
                                ::llvm::AtomicRMWInst::Min :
                                ::llvm::AtomicRMWInst::UMin;
                        break;
                    case xir::AtomicOp::FETCH_MAX:
                        atomic_op = floating ?
                            ::llvm::AtomicRMWInst::FMax :
                            signed_integer ?
                                ::llvm::AtomicRMWInst::Max :
                                ::llvm::AtomicRMWInst::UMax;
                        break;
                    case xir::AtomicOp::COMPARE_EXCHANGE: break;
                }
                if (atomic_op == ::llvm::AtomicRMWInst::BAD_BINOP) {
                    _fail("unsupported direct-buffer atomic operation");
                    return nullptr;
                }
                old = _builder.CreateAtomicRMW(
                    atomic_op, pointer, lane_values[0u], alignment,
                    ::llvm::AtomicOrdering::Monotonic);
            }
            auto *updated = _builder.CreateInsertElement(
                old_values, old, lane);
            _builder.CreateBr(continue_block);
            auto *atomic_end = _builder.GetInsertBlock();

            _builder.SetInsertPoint(continue_block);
            auto *phi = _builder.CreatePHI(result_type, 2u);
            phi->addIncoming(old_values, before);
            phi->addIncoming(updated, atomic_end);
            old_values = phi;
        }
        return old_values;
    }

    [[nodiscard]] ::llvm::Value *_leaf_pointers(
        ::llvm::Value *base, ::llvm::Value *offsets,
        size_t leaf_offset) {
        if (leaf_offset != 0u) {
            offsets = _builder.CreateAdd(
                offsets,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(leaf_offset)));
        }
        return _builder.CreateGEP(
            _builder.getInt8Ty(), base, offsets);
    }

    [[nodiscard]] ::llvm::AllocaInst *_entry_scratch(
        ::llvm::Type *type, std::string_view name) {
        auto &entry_block = _entry->getEntryBlock();
        ::llvm::IRBuilder<> builder{
            &entry_block, entry_block.begin()};
        return builder.CreateAlloca(
            type, nullptr, ::llvm::StringRef{name.data(), name.size()});
    }

    [[nodiscard]] ::llvm::Value *_texture_read(
        const schedule::Instruction &instruction) {
        if (!instruction.result || instruction.operands.size() != 2u) {
            _fail("texture read instruction is malformed");
            return nullptr;
        }
        auto *result = _source.value(*instruction.result);
        auto *texture_value = _source.value(instruction.operands[0u]);
        auto *coordinate_value = _source.value(instruction.operands[1u]);
        auto *texture = _load_value(instruction.operands[0u]);
        auto *coordinate = coordinate_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[1u]),
                *coordinate_value);
        if (result == nullptr || texture_value == nullptr ||
            texture == nullptr || coordinate == nullptr ||
            !texture_value->type->is_texture() ||
            result->value_class != schedule::ValueClass::varying ||
            !result->type->is_vector() ||
            result->type->dimension() != 4u ||
            !coordinate_value->type->is_vector() ||
            (coordinate_value->type->dimension() != 2u &&
             coordinate_value->type->dimension() != 3u)) {
            _fail("LLVM packet codegen requires varying float4/uint4 direct texture reads");
            return nullptr;
        }
        auto *element = result->type->element();
        auto floating = element->is_float32();
        auto integer = element->is_int32() || element->is_uint32();
        if (!floating && !integer) {
            _fail("direct texture reads currently support float32 and int32 elements");
            return nullptr;
        }
        auto op = static_cast<xir::ResourceReadOp>(
            *instruction.source_op);
        auto expected_dimension =
            op == xir::ResourceReadOp::TEXTURE2D_READ ? 2u :
            op == xir::ResourceReadOp::TEXTURE3D_READ ? 3u : 0u;
        if (expected_dimension == 0u ||
            coordinate_value->type->dimension() != expected_dimension) {
            _fail("direct texture read dimension mismatch");
            return nullptr;
        }

        auto *scalar_type = _data_type(element, false);
        auto *scratch_type = ::llvm::ArrayType::get(scalar_type, 4u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "texture.read." + std::to_string(instruction.result->value));
        auto *read = _builder.CreateExtractValue(
            texture, {floating ? 1u : 2u});
        auto *object = _builder.CreateExtractValue(texture, {0u});
        auto *level = _builder.CreateExtractValue(texture, {6u});
        auto *read_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {::llvm::PointerType::getUnqual(_module.getContext()),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             ::llvm::PointerType::getUnqual(_module.getContext())},
            false);
        std::array<::llvm::Value *, 3u> coordinates{
            nullptr, nullptr,
            _builder.CreateVectorSplat(
                _width, _builder.getInt32(0u))};
        for (auto axis = uint32_t{0u}; axis < expected_dimension; axis++) {
            coordinates[axis] = _extract_child(
                coordinate, coordinate_value->type, axis, true);
        }
        auto *result_type = _data_type(result->type, true);
        ::llvm::Value *pixels =
            ::llvm::Constant::getNullValue(result_type);
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            auto *before = _builder.GetInsertBlock();
            auto *active = _builder.CreateExtractElement(
                _active_mask, lane);
            auto *read_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "texture.read.lane." + std::to_string(lane), _entry);
            auto *continue_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "texture.read.continue." + std::to_string(lane), _entry);
            _builder.CreateCondBr(active, read_block, continue_block);

            _builder.SetInsertPoint(read_block);
            auto *x = _builder.CreateExtractElement(coordinates[0u], lane);
            auto *y = _builder.CreateExtractElement(coordinates[1u], lane);
            auto *z = _builder.CreateExtractElement(coordinates[2u], lane);
            _builder.CreateCall(
                read_type, read,
                {object, level, x, y, z, scratch});
            auto *updated = pixels;
            for (auto component = uint32_t{0u}; component < 4u;
                 component++) {
                auto *component_pointer = _builder.CreateGEP(
                    scratch_type, scratch,
                    {_builder.getInt32(0u),
                     _builder.getInt32(component)});
                auto *scalar = _builder.CreateLoad(
                    scalar_type, component_pointer);
                auto *lanes = _extract_child(
                    updated, result->type, component, true);
                lanes = _builder.CreateInsertElement(
                    lanes, scalar, lane);
                updated = _insert_child(
                    updated, lanes, result->type, component, true);
            }
            _builder.CreateBr(continue_block);
            auto *read_end = _builder.GetInsertBlock();

            _builder.SetInsertPoint(continue_block);
            auto *phi = _builder.CreatePHI(result_type, 2u);
            phi->addIncoming(pixels, before);
            phi->addIncoming(updated, read_end);
            pixels = phi;
        }
        return pixels;
    }

    void _texture_write(
        const schedule::Instruction &instruction) {
        if (instruction.operands.size() != 3u) {
            _fail("texture write instruction is malformed");
            return;
        }
        auto *texture_value = _source.value(instruction.operands[0u]);
        auto *coordinate_value = _source.value(instruction.operands[1u]);
        auto *written_value = _source.value(instruction.operands[2u]);
        auto *texture = _load_value(instruction.operands[0u]);
        auto *coordinate = coordinate_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[1u]),
                *coordinate_value);
        auto *written = written_value == nullptr ? nullptr :
            _as_lane_vector(
                _load_value(instruction.operands[2u]), *written_value);
        if (texture_value == nullptr || coordinate_value == nullptr ||
            written_value == nullptr || texture == nullptr ||
            coordinate == nullptr || written == nullptr ||
            !texture_value->type->is_texture() ||
            !written_value->type->is_vector() ||
            written_value->type->dimension() != 4u ||
            !coordinate_value->type->is_vector()) {
            _fail("direct texture write has invalid operands");
            return;
        }
        auto op = static_cast<xir::ResourceWriteOp>(
            *instruction.source_op);
        auto expected_dimension =
            op == xir::ResourceWriteOp::TEXTURE2D_WRITE ? 2u :
            op == xir::ResourceWriteOp::TEXTURE3D_WRITE ? 3u : 0u;
        if (expected_dimension == 0u ||
            coordinate_value->type->dimension() != expected_dimension) {
            _fail("direct texture write dimension mismatch");
            return;
        }
        auto *element = written_value->type->element();
        auto floating = element->is_float32();
        auto integer = element->is_int32() || element->is_uint32();
        if (!floating && !integer) {
            _fail("direct texture writes currently support float32 and int32 elements");
            return;
        }
        auto *scalar_type = _data_type(element, false);
        auto *scratch_type = ::llvm::ArrayType::get(scalar_type, 4u);
        auto *scratch = _entry_scratch(
            scratch_type,
            "texture.write." + std::to_string(
                instruction.operands[2u].value));
        auto *write = _builder.CreateExtractValue(
            texture, {floating ? 3u : 4u});
        auto *object = _builder.CreateExtractValue(texture, {0u});
        auto *level = _builder.CreateExtractValue(texture, {6u});
        auto *write_type = ::llvm::FunctionType::get(
            _builder.getVoidTy(),
            {::llvm::PointerType::getUnqual(_module.getContext()),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             _builder.getInt32Ty(), _builder.getInt32Ty(),
             ::llvm::PointerType::getUnqual(_module.getContext())},
            false);
        std::array<::llvm::Value *, 3u> coordinates{
            nullptr, nullptr,
            _builder.CreateVectorSplat(
                _width, _builder.getInt32(0u))};
        for (auto axis = uint32_t{0u}; axis < expected_dimension; axis++) {
            coordinates[axis] = _extract_child(
                coordinate, coordinate_value->type, axis, true);
        }
        for (auto lane = uint32_t{0u}; lane < _width; lane++) {
            auto *active = _builder.CreateExtractElement(
                _active_mask, lane);
            auto *write_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "texture.write.lane." + std::to_string(lane), _entry);
            auto *continue_block = ::llvm::BasicBlock::Create(
                _module.getContext(),
                "texture.write.continue." + std::to_string(lane), _entry);
            _builder.CreateCondBr(active, write_block, continue_block);

            _builder.SetInsertPoint(write_block);
            for (auto component = uint32_t{0u}; component < 4u;
                 component++) {
                auto *lanes = _extract_child(
                    written, written_value->type, component, true);
                auto *scalar = _builder.CreateExtractElement(lanes, lane);
                auto *component_pointer = _builder.CreateGEP(
                    scratch_type, scratch,
                    {_builder.getInt32(0u),
                     _builder.getInt32(component)});
                _builder.CreateStore(scalar, component_pointer);
            }
            auto *x = _builder.CreateExtractElement(coordinates[0u], lane);
            auto *y = _builder.CreateExtractElement(coordinates[1u], lane);
            auto *z = _builder.CreateExtractElement(coordinates[2u], lane);
            _builder.CreateCall(
                write_type, write,
                {object, level, x, y, z, scratch});
            _builder.CreateBr(continue_block);
            _builder.SetInsertPoint(continue_block);
        }
    }

    [[nodiscard]] ::llvm::Value *_gather_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, size_t leaf_offset = 0u) {
        if (_is_scalar_data(type)) {
            auto *pointers = _leaf_pointers(base, offsets, leaf_offset);
            auto *element = type->is_bool() ?
                static_cast<::llvm::Type *>(_builder.getInt8Ty()) :
                _data_type(type, false);
            auto *lanes = ::llvm::FixedVectorType::get(element, _width);
            auto *gathered = _builder.CreateMaskedGather(
                lanes, pointers, ::llvm::Align{1u}, _active_mask,
                ::llvm::Constant::getNullValue(lanes));
            return type->is_bool() ?
                _builder.CreateICmpNE(
                    gathered, ::llvm::Constant::getNullValue(lanes)) :
                gathered;
        }
        return _assemble(type, true, [&](uint32_t i) {
            return _gather_data(
                base, offsets, _child_type(type, i),
                leaf_offset + _child_offset(type, i));
        });
    }

    void _scatter_data(
        ::llvm::Value *base, ::llvm::Value *offsets,
        const Type *type, ::llvm::Value *value,
        size_t leaf_offset = 0u) {
        if (_is_scalar_data(type)) {
            auto *pointers = _leaf_pointers(base, offsets, leaf_offset);
            if (type->is_bool()) {
                value = _builder.CreateZExt(
                    value,
                    ::llvm::FixedVectorType::get(
                        _builder.getInt8Ty(), _width));
            }
            _builder.CreateMaskedScatter(
                value, pointers, ::llvm::Align{1u}, _active_mask);
            return;
        }
        for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
            _scatter_data(
                base, offsets, _child_type(type, i),
                _extract_child(value, type, i, true),
                leaf_offset + _child_offset(type, i));
        }
    }

    [[nodiscard]] ::llvm::Value *_resource_read(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op ||
            instruction.operands.size() != 2u) {
            _fail("buffer read instruction is malformed");
            return nullptr;
        }
        auto op = static_cast<xir::ResourceReadOp>(
            *instruction.source_op);
        if (op == xir::ResourceReadOp::TEXTURE2D_READ ||
            op == xir::ResourceReadOp::TEXTURE3D_READ) {
            return _texture_read(instruction);
        }
        auto byte_address =
            op == xir::ResourceReadOp::BYTE_BUFFER_READ ||
            op == xir::ResourceReadOp::BYTE_BUFFER_VOLATILE_READ;
        if (!byte_address &&
            op != xir::ResourceReadOp::BUFFER_READ &&
            op != xir::ResourceReadOp::BUFFER_VOLATILE_READ) {
            _fail("LLVM packet codegen only supports direct buffer reads");
            return nullptr;
        }
        auto *buffer_value = _source.value(instruction.operands[0u]);
        auto *index_value = _source.value(instruction.operands[1u]);
        auto *result_value = _source.value(*instruction.result);
        auto *buffer = _load_value(instruction.operands[0u]);
        auto *index = _as_lane_vector(
            _load_value(instruction.operands[1u]), *index_value);
        if (buffer_value == nullptr || result_value == nullptr ||
            buffer == nullptr || index == nullptr ||
            !buffer_value->type->is_buffer()) {
            _fail("buffer read has invalid operands");
            return nullptr;
        }
        auto stride = byte_address ? 1u :
            static_cast<uint64_t>(buffer_value->type->element()->size());
        auto *base = _builder.CreateExtractValue(buffer, {0u});
        return _gather_data(
            base, _lane_offsets(index, stride), result_value->type);
    }

    void _resource_write(const schedule::Instruction &instruction) {
        if (!instruction.source_op || instruction.operands.size() != 3u) {
            _fail("buffer write instruction is malformed");
            return;
        }
        auto op = static_cast<xir::ResourceWriteOp>(
            *instruction.source_op);
        if (op == xir::ResourceWriteOp::TEXTURE2D_WRITE ||
            op == xir::ResourceWriteOp::TEXTURE3D_WRITE) {
            _texture_write(instruction);
            return;
        }
        auto byte_address =
            op == xir::ResourceWriteOp::BYTE_BUFFER_WRITE ||
            op == xir::ResourceWriteOp::BYTE_BUFFER_VOLATILE_WRITE;
        if (!byte_address &&
            op != xir::ResourceWriteOp::BUFFER_WRITE &&
            op != xir::ResourceWriteOp::BUFFER_VOLATILE_WRITE) {
            _fail("LLVM packet codegen only supports direct buffer writes");
            return;
        }
        auto *buffer_value = _source.value(instruction.operands[0u]);
        auto *index_value = _source.value(instruction.operands[1u]);
        auto *written_value = _source.value(instruction.operands[2u]);
        auto *buffer = _load_value(instruction.operands[0u]);
        auto *index = _as_lane_vector(
            _load_value(instruction.operands[1u]), *index_value);
        auto *value = _as_lane_vector(
            _load_value(instruction.operands[2u]), *written_value);
        if (buffer_value == nullptr || buffer == nullptr ||
            index == nullptr || value == nullptr ||
            !buffer_value->type->is_buffer()) {
            _fail("buffer write has invalid operands");
            return;
        }
        auto stride = byte_address ? 1u :
            static_cast<uint64_t>(buffer_value->type->element()->size());
        auto *base = _builder.CreateExtractValue(buffer, {0u});
        _scatter_data(
            base, _lane_offsets(index, stride), written_value->type, value);
    }

    [[nodiscard]] ::llvm::Value *_resource_query(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op ||
            instruction.operands.size() != 1u) {
            _fail("buffer query instruction is malformed");
            return nullptr;
        }
        auto *result = _source.value(*instruction.result);
        auto *buffer_value = _source.value(instruction.operands[0u]);
        auto *buffer = _load_value(instruction.operands[0u]);
        if (result == nullptr || buffer_value == nullptr ||
            buffer == nullptr) {
            _fail("resource query has invalid operands");
            return nullptr;
        }
        auto op = static_cast<xir::ResourceQueryOp>(
            *instruction.source_op);
        if (op == xir::ResourceQueryOp::TEXTURE2D_SIZE ||
            op == xir::ResourceQueryOp::TEXTURE3D_SIZE) {
            if (!buffer_value->type->is_texture() ||
                !result->type->is_vector()) {
                _fail("texture size query has invalid types");
                return nullptr;
            }
            auto dimension =
                op == xir::ResourceQueryOp::TEXTURE2D_SIZE ? 2u : 3u;
            if (result->type->dimension() != dimension) {
                _fail("texture size query result dimension mismatch");
                return nullptr;
            }
            auto *object = _builder.CreateExtractValue(buffer, {0u});
            auto *size = _builder.CreateExtractValue(buffer, {5u});
            auto *level = _builder.CreateExtractValue(buffer, {6u});
            auto *size_type = ::llvm::FunctionType::get(
                _builder.getInt32Ty(),
                {::llvm::PointerType::getUnqual(_module.getContext()),
                 _builder.getInt32Ty(), _builder.getInt32Ty()},
                false);
            auto *uniform = _assemble(
                result->type, false, [&](uint32_t axis) {
                    return _builder.CreateCall(
                        size_type, size,
                        {object, level, _builder.getInt32(axis)});
                });
            return result->value_class ==
                    schedule::ValueClass::varying ?
                _splat_data(uniform, result->type) : uniform;
        }
        if (!buffer_value->type->is_buffer()) {
            _fail("buffer query has invalid resource type");
            return nullptr;
        }
        auto *value = _builder.CreateExtractValue(buffer, {1u});
        switch (op) {
            case xir::ResourceQueryOp::BUFFER_SIZE:
                value = _builder.CreateUDiv(
                    value,
                    _builder.getInt64(
                        buffer_value->type->element()->size()));
                break;
            case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: break;
            case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: {
                auto *pointer = _builder.CreateExtractValue(buffer, {0u});
                value = _builder.CreatePtrToInt(
                    pointer, _builder.getInt64Ty());
                break;
            }
            default:
                _fail("LLVM packet codegen only supports direct buffer queries");
                return nullptr;
        }
        auto *destination = _data_type(result->type, false);
        value = _builder.CreateZExtOrTrunc(value, destination);
        return result->value_class == schedule::ValueClass::varying ?
            _splat_data(value, result->type) : value;
    }

    [[nodiscard]] ::llvm::Value *_collective(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op ||
            !instruction.participant_mask) {
            _fail("warp collective is missing result, operation, or participant mask");
            return nullptr;
        }
        auto *participants = _load_value(*instruction.participant_mask);
        auto *result_value = _source.value(*instruction.result);
        if (participants == nullptr) { return nullptr; }
        std::vector<::llvm::Value *> operands;
        std::vector<const schedule::Value *> operand_values;
        operands.reserve(instruction.operands.size());
        operand_values.reserve(instruction.operands.size());
        for (auto operand_id : instruction.operands) {
            auto *operand = _source.value(operand_id);
            if (operand == nullptr) {
                _fail("warp collective references an invalid operand");
                return nullptr;
            }
            auto *llvm_operand = _as_lane_vector(
                _load_value(operand_id), *operand);
            if (llvm_operand == nullptr) { return nullptr; }
            operands.emplace_back(llvm_operand);
            operand_values.emplace_back(operand);
        }
        auto require = [&](size_t count) {
            if (operands.size() != count) {
                _fail("warp collective has an invalid operand count");
                return false;
            }
            return true;
        };
        auto op = static_cast<xir::ThreadGroupOp>(*instruction.source_op);
        auto cohort_scalar = [&](::llvm::Value *lanes) {
            if (lanes == nullptr || result_value == nullptr ||
                result_value->value_class !=
                    schedule::ValueClass::cohort_uniform) {
                return lanes;
            }
            auto *first = _collectives.first_active_lane(
                _builder, participants);
            auto *safe = _builder.CreateSelect(
                _builder.CreateOrReduce(participants), first,
                _builder.getInt32(0u));
            return _extract_lane(lanes, result_value->type, safe);
        };
        auto reduce_components = [&](const UnaryLeaf &leaf) {
            if (!require(1u) || result_value == nullptr) {
                return static_cast<::llvm::Value *>(nullptr);
            }
            return _componentwise_varying_to_uniform(
                result_value->type, operands[0u],
                operand_values[0u]->type, leaf);
        };
        auto scan_components = [&](const UnaryLeaf &leaf) {
            if (!require(1u) || result_value == nullptr) {
                return static_cast<::llvm::Value *>(nullptr);
            }
            return _componentwise_unary(
                result_value->type, operands[0u],
                operand_values[0u]->type, true, leaf);
        };
        switch (op) {
            case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE:
                if (!require(0u)) { return nullptr; }
                return _collectives.is_first_active_lane(
                    _builder, participants);
            case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE:
                if (!require(0u)) { return nullptr; }
                return _collectives.first_active_lane(
                    _builder, participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_all_equal(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_bit_and(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_bit_or(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_bit_xor(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_count_bits(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *type) {
                        return _collectives.active_max(
                            _builder, value, participants,
                            type->is_int());
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *type) {
                        return _collectives.active_min(
                            _builder, value, participants,
                            type->is_int());
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_product(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
                return reduce_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.active_sum(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_ACTIVE_ALL:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_all(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_ANY:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_any(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_bit_mask(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS:
                if (!require(1u)) { return nullptr; }
                return _collectives.prefix_count_bits(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_PREFIX_SUM:
                return scan_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.prefix_sum(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
                return scan_components(
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.prefix_product(
                            _builder, value, participants);
                    });
            case xir::ThreadGroupOp::WARP_READ_LANE:
                if (!require(2u)) { return nullptr; }
                if (result_value == nullptr) { return nullptr; }
                return cohort_scalar(_componentwise_unary(
                    result_value->type, operands[0u],
                    operand_values[0u]->type, true,
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.read_lane(
                            _builder, value, operands[1u], participants)
                            .values;
                    }));
            case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
                if (!require(1u) || result_value == nullptr) {
                    return nullptr;
                }
                return cohort_scalar(_componentwise_unary(
                    result_value->type, operands[0u],
                    operand_values[0u]->type, true,
                    [&](::llvm::Value *value, const Type *) {
                        return _collectives.read_first_active_lane(
                            _builder, value, participants)
                            .values;
                    }));
            default:
                _fail("Phase-2 LLVM packet codegen encountered a non-warp thread-group operation");
                return nullptr;
        }
    }

    void _emit_instruction(const schedule::Instruction &instruction) {
        ::llvm::Value *value = nullptr;
        switch (instruction.opcode) {
            case schedule::Opcode::arithmetic:
                value = _arithmetic(instruction);
                break;
            case schedule::Opcode::cast:
                value = _cast(instruction);
                break;
            case schedule::Opcode::resource_query:
                value = _resource_query(instruction);
                break;
            case schedule::Opcode::resource_read:
                value = _resource_read(instruction);
                break;
            case schedule::Opcode::resource_write:
                _resource_write(instruction);
                break;
            case schedule::Opcode::atomic:
                value = _atomic(instruction);
                break;
            case schedule::Opcode::warp_collective:
                value = _collective(instruction);
                break;
            default:
                _fail("unsupported Schedule IR instruction reached LLVM emission");
                return;
        }
        if (instruction.result && value != nullptr) {
            _locals.insert_or_assign(instruction.result->value, value);
            auto id = instruction.result->value;
            if (id < _spilled_instruction_values.size() &&
                _spilled_instruction_values[id] != 0u) {
                auto *schedule_value = _source.value(*instruction.result);
                auto *slot = _state_slots[id];
                if (schedule_value->value_class ==
                    schedule::ValueClass::warp_uniform) {
                    _builder.CreateStore(value, slot);
                } else {
                    auto *lanes = schedule_value->value_class ==
                            schedule::ValueClass::token ?
                        _splat(value) :
                        _as_lane_vector(value, *schedule_value);
                    if (lanes == nullptr) { return; }
                    auto *old = _builder.CreateLoad(
                        slot->getAllocatedType(), slot);
                    auto *merged = schedule_value->type == nullptr ?
                        _builder.CreateSelect(_active_mask, lanes, old) :
                        _masked_merge(
                            lanes, old, schedule_value->type,
                            _active_mask);
                    if (merged != nullptr) {
                        _builder.CreateStore(merged, slot);
                    }
                }
            }
        }
        if (!_collectives.succeeded()) { _fail(_collectives.error()); }
    }

    void _assign(schedule::EdgeAssignment assignment,
                 ::llvm::Value *mask) {
        auto *destination = _source.value(assignment.destination);
        auto *source = _source.value(assignment.source);
        auto *value = _load_value(assignment.source);
        if (destination == nullptr || source == nullptr || value == nullptr) {
            return;
        }
        auto *slot = _state_slots[assignment.destination.value];
        if (destination->value_class == schedule::ValueClass::warp_uniform) {
            _builder.CreateStore(value, slot);
            return;
        }
        auto *lanes = _as_lane_vector(value, *source);
        if (lanes == nullptr) { return; }
        auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
        auto *merged = _masked_merge(lanes, old, destination->type, mask);
        if (merged != nullptr) { _builder.CreateStore(merged, slot); }
    }

    void _apply_assignments(
        const std::vector<schedule::EdgeAssignment> &assignments,
        ::llvm::Value *mask) {
        for (auto assignment : assignments) {
            _assign(assignment, mask);
            if (_failed()) { return; }
        }
    }

    [[nodiscard]] ::llvm::Value *_zero_mask() noexcept {
        return ::llvm::Constant::getNullValue(_layout.mask_type());
    }

    [[nodiscard]] ::llvm::Value *_splat(::llvm::Value *scalar) {
        return scalar->getType()->isVectorTy() ? scalar :
                                                _builder.CreateVectorSplat(
                                                    _width, scalar);
    }

    [[nodiscard]] ::llvm::Value *_safe_first_lane(::llvm::Value *mask) {
        auto *any = _builder.CreateOrReduce(mask);
        auto *first = _collectives.first_active_lane(_builder, mask);
        return _builder.CreateSelect(any, first, _builder.getInt32(0u));
    }

    void _masked_write(::llvm::AllocaInst *slot, ::llvm::Value *value,
                       ::llvm::Value *mask) {
        auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
        auto *lanes = _splat(value);
        _builder.CreateStore(_builder.CreateSelect(mask, lanes, old), slot);
    }

    [[nodiscard]] ::llvm::Value *_frame_mask_pointer(
        ::llvm::AllocaInst *frames, ::llvm::Value *index) {
        auto *array = ::llvm::cast<::llvm::ArrayType>(
            frames->getAllocatedType());
        return _builder.CreateInBoundsGEP(
            array, frames, {_builder.getInt32(0u), index});
    }

    void _trap_if(::llvm::Value *condition, std::string_view label) {
        auto *trap = ::llvm::BasicBlock::Create(
            _module.getContext(), std::string{label} + ".trap", _entry);
        auto *resume = ::llvm::BasicBlock::Create(
            _module.getContext(), std::string{label} + ".resume", _entry);
        _builder.CreateCondBr(condition, trap, resume);
        _builder.SetInsertPoint(trap);
#if LLVM_VERSION_MAJOR >= 22
        auto *intrinsic = ::llvm::Intrinsic::getOrInsertDeclaration(
#else
        auto *intrinsic = ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::trap);
        _builder.CreateCall(intrinsic);
        _builder.CreateUnreachable();
        _builder.SetInsertPoint(resume);
    }

    [[nodiscard]] ::llvm::Value *_current_token(::llvm::Value *mask) {
        auto *tokens = _builder.CreateLoad(
            _token_state->getAllocatedType(), _token_state);
        return _builder.CreateExtractElement(
            tokens, _safe_first_lane(mask));
    }

    // Allocate one of at most W simultaneously live divergence frames. Every
    // real allocation partitions a non-empty cohort into two non-empty sets,
    // so a W-lane warp can have at most W-1 such frames. Re-entering the same
    // convergence while traversing a loop reuses the current frame.
    void _declare_convergence(schedule::ConvergenceId convergence,
                              ::llvm::Value *true_mask,
                              ::llvm::Value *false_mask) {
        auto *has_true = _builder.CreateOrReduce(true_mask);
        auto *has_false = _builder.CreateOrReduce(false_mask);
        auto *divergent = _builder.CreateAnd(has_true, has_false);
        auto *current_token = _current_token(_active_mask);
        auto *has_current = _builder.CreateICmpNE(
            current_token, _builder.getInt32(0u));
        auto *current_index = _builder.CreateSelect(
            has_current,
            _builder.CreateSub(current_token, _builder.getInt32(1u)),
            _builder.getInt32(0u));
        auto *active_frames = _builder.CreateLoad(
            _frame_active->getAllocatedType(), _frame_active);
        auto *current_active = _builder.CreateExtractElement(
            active_frames, current_index);
        auto *static_ids = _builder.CreateLoad(
            _frame_static_id->getAllocatedType(), _frame_static_id);
        auto *current_static = _builder.CreateExtractElement(
            static_ids, current_index);
        auto *reuse = _builder.CreateAnd(
            has_current,
            _builder.CreateAnd(
                current_active,
                _builder.CreateICmpEQ(
                    current_static,
                    _builder.getInt32(convergence.value))));
        auto *allocate = _builder.CreateAnd(divergent,
                                            _builder.CreateNot(reuse));
        auto *free_frames = _builder.CreateNot(active_frames);
        auto *has_free = _builder.CreateOrReduce(free_frames);
        _trap_if(_builder.CreateAnd(allocate, _builder.CreateNot(has_free)),
                 "convergence.overflow");
        auto *free_index = _safe_first_lane(free_frames);

        auto *old_free_active = _builder.CreateExtractElement(
            active_frames, free_index);
        auto *new_free_active = _builder.CreateSelect(
            allocate, _builder.getTrue(), old_free_active);
        _builder.CreateStore(
            _builder.CreateInsertElement(
                active_frames, new_free_active, free_index),
            _frame_active);

        auto *old_static = _builder.CreateExtractElement(
            static_ids, free_index);
        auto *new_static = _builder.CreateSelect(
            allocate, _builder.getInt32(convergence.value), old_static);
        _builder.CreateStore(
            _builder.CreateInsertElement(
                static_ids, new_static, free_index),
            _frame_static_id);

        auto *parents = _builder.CreateLoad(
            _frame_parent_token->getAllocatedType(), _frame_parent_token);
        auto *old_parent = _builder.CreateExtractElement(parents, free_index);
        auto *new_parent = _builder.CreateSelect(
            allocate, current_token, old_parent);
        _builder.CreateStore(
            _builder.CreateInsertElement(parents, new_parent, free_index),
            _frame_parent_token);

        auto *expected_ptr = _frame_mask_pointer(
            _frame_expected, free_index);
        auto *arrived_ptr = _frame_mask_pointer(
            _frame_arrived, free_index);
        auto *old_expected = _builder.CreateLoad(
            _layout.mask_type(), expected_ptr);
        auto *old_arrived = _builder.CreateLoad(
            _layout.mask_type(), arrived_ptr);
        _builder.CreateStore(
            _builder.CreateSelect(allocate, _active_mask, old_expected),
            expected_ptr);
        _builder.CreateStore(
            _builder.CreateSelect(allocate, _zero_mask(), old_arrived),
            arrived_ptr);

        auto *allocated_token = _builder.CreateAdd(
            free_index, _builder.getInt32(1u));
        auto *gate_token = _builder.CreateSelect(
            reuse, current_token, allocated_token);
        auto *next_token = _builder.CreateSelect(
            divergent, gate_token, current_token);
        _masked_write(_token_state, next_token, _active_mask);
    }

    void _advance_loop_epoch(schedule::LoopId loop, ::llvm::Value *mask) {
        auto *slot = _loop_epochs[loop.value];
        auto *old = _builder.CreateLoad(slot->getAllocatedType(), slot);
        auto *one = _builder.CreateVectorSplat(
            _width, _builder.getInt32(1u));
        _builder.CreateStore(
            _builder.CreateSelect(mask, _builder.CreateAdd(old, one), old),
            slot);
    }

    [[nodiscard]] ::llvm::Value *_arrive_at_convergence(
        schedule::ConvergenceId convergence, ::llvm::Value *flow) {
        auto *any = _builder.CreateOrReduce(flow);
        auto *token = _current_token(flow);
        auto *has_token = _builder.CreateAnd(
            any, _builder.CreateICmpNE(token, _builder.getInt32(0u)));
        auto *index = _builder.CreateSelect(
            has_token,
            _builder.CreateSub(token, _builder.getInt32(1u)),
            _builder.getInt32(0u));
        auto *active_frames = _builder.CreateLoad(
            _frame_active->getAllocatedType(), _frame_active);
        auto *frame_active = _builder.CreateExtractElement(
            active_frames, index);
        auto *static_ids = _builder.CreateLoad(
            _frame_static_id->getAllocatedType(), _frame_static_id);
        auto *static_id = _builder.CreateExtractElement(static_ids, index);
        auto *matches = _builder.CreateAnd(
            has_token,
            _builder.CreateAnd(
                frame_active,
                _builder.CreateICmpEQ(
                    static_id,
                    _builder.getInt32(convergence.value))));

        auto *expected_ptr = _frame_mask_pointer(_frame_expected, index);
        auto *arrived_ptr = _frame_mask_pointer(_frame_arrived, index);
        auto *expected = _builder.CreateLoad(
            _layout.mask_type(), expected_ptr);
        auto *arrived = _builder.CreateLoad(
            _layout.mask_type(), arrived_ptr);
        auto *new_arrived = _builder.CreateOr(arrived, flow);
        auto *live = _builder.CreateLoad(
            _live_mask->getAllocatedType(), _live_mask);
        auto *expected_live = _builder.CreateAnd(expected, live);
        auto *complete = _builder.CreateAnd(
            matches,
            _builder.CreateAndReduce(
                _builder.CreateICmpEQ(new_arrived, expected_live)));
        auto *stored_arrived = _builder.CreateSelect(
            matches,
            _builder.CreateSelect(complete, _zero_mask(), new_arrived),
            arrived);
        _builder.CreateStore(stored_arrived, arrived_ptr);
        _builder.CreateStore(
            _builder.CreateSelect(complete, _zero_mask(), expected),
            expected_ptr);

        auto *old_frame_active = _builder.CreateExtractElement(
            active_frames, index);
        _builder.CreateStore(
            _builder.CreateInsertElement(
                active_frames,
                _builder.CreateSelect(
                    complete, _builder.getFalse(), old_frame_active),
                index),
            _frame_active);

        auto *matching_lanes = _builder.CreateAnd(flow, _splat(matches));
        auto *runnable = _builder.CreateLoad(
            _runnable_mask->getAllocatedType(), _runnable_mask);
        _builder.CreateStore(
            _builder.CreateAnd(runnable,
                               _builder.CreateNot(matching_lanes)),
            _runnable_mask);

        auto *released = _builder.CreateSelect(
            _splat(complete), new_arrived, _zero_mask());
        auto *parents = _builder.CreateLoad(
            _frame_parent_token->getAllocatedType(), _frame_parent_token);
        auto *parent_token = _builder.CreateExtractElement(parents, index);
        _masked_write(_token_state, parent_token, released);
        return _builder.CreateSelect(_splat(matches), released, flow);
    }

    void _resume(schedule::BlockId target, ::llvm::Value *mask) {
        _masked_write(_pc_state, _builder.getInt32(target.value), mask);
        auto *runnable = _builder.CreateLoad(
            _runnable_mask->getAllocatedType(), _runnable_mask);
        _builder.CreateStore(_builder.CreateOr(runnable, mask),
                             _runnable_mask);
    }

    void _route_edge(const schedule::ControlEdge &edge,
                     ::llvm::Value *mask) {
        _apply_assignments(edge.assignments, mask);
        if (_failed()) { return; }
        if (edge.loop_back) {
            _advance_loop_epoch(*edge.loop_back, mask);
        }
        auto *flow = mask;
        for (auto convergence : edge.joins) {
            flow = _arrive_at_convergence(convergence, flow);
        }
        _resume(edge.target, flow);
    }

    void _emit_arrival(const schedule::ControlEdge &edge,
                       ::llvm::Value *mask) {
        _route_edge(edge, mask);
        _builder.CreateBr(_scheduler_loop);
    }

    void _emit_terminator(const schedule::Terminator &terminator) {
        std::visit(
            [&](const auto &control) {
                using T = std::decay_t<decltype(control)>;
                if constexpr (std::is_same_v<T, schedule::BranchTerminator>) {
                    _emit_arrival(control.edge, _active_mask);
                } else if constexpr (
                    std::is_same_v<T, schedule::SplitTerminator>) {
                    auto *condition_value = _source.value(control.condition);
                    auto *condition = _load_value(control.condition);
                    if (condition == nullptr || condition_value == nullptr) {
                        return;
                    }
                    if (condition_value->value_class ==
                        schedule::ValueClass::varying) {
                        auto *true_mask = _builder.CreateAnd(
                            _active_mask, condition);
                        auto *false_mask = _builder.CreateAnd(
                            _active_mask, _builder.CreateNot(condition));
                        if (control.convergence) {
                            _declare_convergence(
                                *control.convergence,
                                true_mask, false_mask);
                        }
                        _route_edge(control.true_edge, true_mask);
                        _route_edge(control.false_edge, false_mask);
                        if (_failed()) { return; }
                        _builder.CreateBr(_scheduler_loop);
                    } else {
                        auto *true_path = ::llvm::BasicBlock::Create(
                            _module.getContext(), "uniform.true", _entry);
                        auto *false_path = ::llvm::BasicBlock::Create(
                            _module.getContext(), "uniform.false", _entry);
                        _builder.CreateCondBr(
                            condition, true_path, false_path);
                        _builder.SetInsertPoint(true_path);
                        _emit_arrival(control.true_edge, _active_mask);
                        _builder.SetInsertPoint(false_path);
                        _emit_arrival(control.false_edge, _active_mask);
                    }
                } else if constexpr (
                    std::is_same_v<T, schedule::JoinTerminator>) {
                    auto *point = _source.convergence(control.convergence);
                    schedule::ControlEdge edge{point->target};
                    edge.joins.emplace_back(control.convergence);
                    edge.assignments = control.assignments;
                    _emit_arrival(edge, _active_mask);
                } else if constexpr (
                    std::is_same_v<T, schedule::ReturnTerminator>) {
                    if (control.value) {
                        auto *schedule_value = _source.value(*control.value);
                        auto *value = _as_lane_vector(
                            _load_value(*control.value), *schedule_value);
                        if (value != nullptr) {
                            _builder.CreateMaskedStore(
                                value, _return_buffer,
                                ::llvm::Align{schedule_value->type->alignment()},
                                _active_mask);
                        }
                    }
                    auto *live = _builder.CreateLoad(
                        _live_mask->getAllocatedType(), _live_mask);
                    auto *new_live = _builder.CreateAnd(
                        live, _builder.CreateNot(_active_mask));
                    _builder.CreateStore(new_live, _live_mask);
                    auto *runnable = _builder.CreateLoad(
                        _runnable_mask->getAllocatedType(),
                        _runnable_mask);
                    _builder.CreateStore(
                        _builder.CreateAnd(
                            runnable, _builder.CreateNot(_active_mask)),
                        _runnable_mask);

                    // A lane that terminates is removed from every frame's
                    // expected mask. Frame storage is bounded by W rather
                    // than the number of static CFG convergence points.
                    std::vector<::llvm::Constant *> targets;
                    targets.reserve(_source.convergence_points().size());
                    for (auto &&point : _source.convergence_points()) {
                        targets.emplace_back(
                            _builder.getInt32(point.target.value));
                    }
                    auto *target_table = targets.empty() ? nullptr :
                        ::llvm::ConstantVector::get(targets);
                    for (auto frame = uint32_t{0u}; frame < _width; frame++) {
                        auto *index = _builder.getInt32(frame);
                        auto *active_frames = _builder.CreateLoad(
                            _frame_active->getAllocatedType(),
                            _frame_active);
                        auto *frame_active = _builder.CreateExtractElement(
                            active_frames, index);
                        auto *expected_ptr = _frame_mask_pointer(
                            _frame_expected, index);
                        auto *arrived_ptr = _frame_mask_pointer(
                            _frame_arrived, index);
                        auto *expected = _builder.CreateLoad(
                            _layout.mask_type(), expected_ptr);
                        auto *arrived = _builder.CreateLoad(
                            _layout.mask_type(), arrived_ptr);
                        auto *expected_live = _builder.CreateAnd(
                            expected, new_live);
                        auto *complete = _builder.CreateAnd(
                            frame_active,
                            _builder.CreateAndReduce(
                                _builder.CreateICmpEQ(
                                    arrived, expected_live)));
                        auto *released = _builder.CreateSelect(
                            _splat(complete), arrived, _zero_mask());
                        _builder.CreateStore(
                            _builder.CreateSelect(
                                complete, _zero_mask(), expected_live),
                            expected_ptr);
                        _builder.CreateStore(
                            _builder.CreateSelect(
                                complete, _zero_mask(), arrived),
                            arrived_ptr);
                        _builder.CreateStore(
                            _builder.CreateInsertElement(
                                active_frames,
                                _builder.CreateSelect(
                                    complete, _builder.getFalse(),
                                    frame_active),
                                index),
                            _frame_active);

                        auto *parents = _builder.CreateLoad(
                            _frame_parent_token->getAllocatedType(),
                            _frame_parent_token);
                        auto *parent = _builder.CreateExtractElement(
                            parents, index);
                        _masked_write(_token_state, parent, released);
                        auto *static_ids = _builder.CreateLoad(
                            _frame_static_id->getAllocatedType(),
                            _frame_static_id);
                        auto *static_id = _builder.CreateExtractElement(
                            static_ids, index);
                        if (target_table != nullptr) {
                            auto *target = _builder.CreateExtractElement(
                                target_table, static_id);
                            _masked_write(_pc_state, target, released);
                        }
                        auto *current_runnable = _builder.CreateLoad(
                            _runnable_mask->getAllocatedType(),
                            _runnable_mask);
                        _builder.CreateStore(
                            _builder.CreateOr(
                                current_runnable, released),
                            _runnable_mask);
                    }
                    _builder.CreateBr(_scheduler_loop);
                } else if constexpr (
                    std::is_same_v<T, schedule::UnreachableTerminator>) {
                    _builder.CreateUnreachable();
                } else {
                    _fail("unsupported Schedule IR terminator reached LLVM emission");
                }
            },
            terminator);
    }

    void _find_instruction_spills() {
        _spilled_instruction_values.assign(
            _source.values().size(), uint8_t{0u});
        auto mark = [&](schedule::ValueId id,
                        schedule::BlockId use_block) noexcept {
            auto *value = _source.value(id);
            if (value != nullptr &&
                value->origin == schedule::ValueOrigin::instruction &&
                value->defining_block &&
                *value->defining_block != use_block) {
                _spilled_instruction_values[id.value] = 1u;
            }
        };
        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                for (auto operand : instruction.operands) {
                    mark(operand, block.id);
                }
                if (instruction.participant_mask) {
                    mark(*instruction.participant_mask, block.id);
                }
            }
            auto mark_assignments = [&](const auto &assignments) noexcept {
                for (auto assignment : assignments) {
                    mark(assignment.source, block.id);
                }
            };
            auto mark_edge = [&](const schedule::ControlEdge &edge) noexcept {
                mark_assignments(edge.assignments);
            };
            std::visit(
                [&](const auto &terminator) noexcept {
                    using T = std::decay_t<decltype(terminator)>;
                    if constexpr (std::is_same_v<
                                      T, schedule::BranchTerminator>) {
                        mark_edge(terminator.edge);
                    } else if constexpr (std::is_same_v<
                                             T, schedule::SplitTerminator>) {
                        mark(terminator.condition, block.id);
                        mark_edge(terminator.true_edge);
                        mark_edge(terminator.false_edge);
                    } else if constexpr (std::is_same_v<
                                             T, schedule::SwitchTerminator>) {
                        mark(terminator.selector, block.id);
                        for (auto &&item : terminator.cases) {
                            mark_edge(item.edge);
                        }
                        mark_edge(terminator.default_edge);
                    } else if constexpr (std::is_same_v<
                                             T, schedule::JoinTerminator> ||
                                         std::is_same_v<
                                             T, schedule::LoopBackTerminator>) {
                        mark_assignments(terminator.assignments);
                    } else if constexpr (std::is_same_v<
                                             T,
                                             schedule::BlockBarrierTerminator>) {
                        mark_edge(terminator.resume_edge);
                    } else if constexpr (std::is_same_v<
                                             T, schedule::ReturnTerminator>) {
                        if (terminator.value) {
                            mark(*terminator.value, block.id);
                        }
                    }
                },
                block.terminator);
        }
    }

    void _allocate_state() {
        auto *mask_type = _layout.mask_type();
        auto *zero_mask = ::llvm::Constant::getNullValue(mask_type);
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *zero_i32_lanes = ::llvm::Constant::getNullValue(i32_lanes);
        _live_mask = _builder.CreateAlloca(
            mask_type, nullptr, "live.mask");
        _runnable_mask = _builder.CreateAlloca(
            mask_type, nullptr, "runnable.mask");
        _pc_state = _builder.CreateAlloca(
            i32_lanes, nullptr, "lane.pc");
        _token_state = _builder.CreateAlloca(
            i32_lanes, nullptr, "lane.convergence.token");
        _frame_active = _builder.CreateAlloca(
            mask_type, nullptr, "frame.active");
        _frame_static_id = _builder.CreateAlloca(
            i32_lanes, nullptr, "frame.static.id");
        _frame_parent_token = _builder.CreateAlloca(
            i32_lanes, nullptr, "frame.parent.token");
        auto *frame_masks = ::llvm::ArrayType::get(mask_type, _width);
        _frame_expected = _builder.CreateAlloca(
            frame_masks, nullptr, "frame.expected");
        _frame_arrived = _builder.CreateAlloca(
            frame_masks, nullptr, "frame.arrived");
        _builder.CreateStore(zero_mask, _live_mask);
        _builder.CreateStore(zero_mask, _runnable_mask);
        _builder.CreateStore(zero_i32_lanes, _pc_state);
        _builder.CreateStore(zero_i32_lanes, _token_state);
        _builder.CreateStore(zero_mask, _frame_active);
        _builder.CreateStore(zero_i32_lanes, _frame_static_id);
        _builder.CreateStore(zero_i32_lanes, _frame_parent_token);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(frame_masks), _frame_expected);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(frame_masks), _frame_arrived);

        _loop_epochs.resize(_source.loops().size());
        _block_loops.resize(_source.blocks().size());
        for (auto &&loop : _source.loops()) {
            auto *epoch = _builder.CreateAlloca(
                i32_lanes, nullptr,
                "loop.epoch." + std::to_string(loop.id.value));
            _builder.CreateStore(zero_i32_lanes, epoch);
            _loop_epochs[loop.id.value] = epoch;
            for (auto block : loop.blocks) {
                _block_loops[block.value].emplace_back(loop.id);
            }
        }
        _find_instruction_spills();
        _state_slots.resize(_source.values().size(), nullptr);
        for (auto &&value : _source.values()) {
            auto spill = value.id.value <
                             _spilled_instruction_values.size() &&
                         _spilled_instruction_values[value.id.value] != 0u;
            if (value.origin != schedule::ValueOrigin::state_slot &&
                !spill) {
                continue;
            }
            auto *type = _layout.state_type(value);
            if (type == nullptr) {
                _fail(_layout.error());
                return;
            }
            auto *slot = _builder.CreateAlloca(
                type, nullptr,
                value.name + (spill ? ".spill" : ".slot"));
            _builder.CreateStore(::llvm::Constant::getNullValue(type), slot);
            _state_slots[value.id.value] = slot;
        }
    }

    void _build() {
        auto &context = _module.getContext();
        auto *function_type = ::llvm::FunctionType::get(
            ::llvm::Type::getVoidTy(context),
            {::llvm::PointerType::getUnqual(context),
             ::llvm::PointerType::getUnqual(context),
             ::llvm::PointerType::getUnqual(context),
             ::llvm::Type::getInt32Ty(context)},
            false);
        if (_entry_name.empty()) {
            _entry_name = _source.name().empty() ? "simd_kernel" :
                                                  _source.name();
            _entry_name += ".simd_w" + std::to_string(_width);
        }
        if (_module.getFunction(_entry_name) != nullptr) {
            _fail("LLVM module already contains the requested SIMD entry name");
            return;
        }
        _entry = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::ExternalLinkage,
            _entry_name, _module);
        auto argument = _entry->arg_begin();
        _argument_buffer = &*argument++;
        _argument_buffer->setName("argument_buffer");
        _return_buffer = &*argument++;
        _return_buffer->setName("return_lanes");
        _launch_config = &*argument++;
        _launch_config->setName("launch_config");
        _active_lane_count = &*argument;
        _active_lane_count->setName("active_lane_count");

        auto *prologue = ::llvm::BasicBlock::Create(
            context, "prologue", _entry);
        _builder.SetInsertPoint(prologue);
        _allocate_state();
        if (_failed()) { return; }
        _create_external_values();
        if (_failed()) { return; }

        auto *lane_ids = _lane_ids();
        auto *count = _builder.CreateVectorSplat(
            _width, _active_lane_count);
        auto *initial_mask = _builder.CreateICmpULT(lane_ids, count);
        _ensure_launch_vectors();
        for (auto i = uint32_t{0u}; i < 3u; i++) {
            initial_mask = _builder.CreateAnd(
                initial_mask,
                _builder.CreateICmpULT(
                    _dispatch_id[i],
                    _builder.CreateVectorSplat(
                        _width, _dispatch_size[i])));
        }
        _builder.CreateStore(initial_mask, _live_mask);
        _builder.CreateStore(initial_mask, _runnable_mask);
        _builder.CreateStore(
            _builder.CreateVectorSplat(
                _width, _builder.getInt32(_source.entry().value)),
            _pc_state);

        _scheduler_loop = ::llvm::BasicBlock::Create(
            context, "scheduler.loop", _entry);
        auto *dispatch = ::llvm::BasicBlock::Create(
            context, "scheduler.dispatch", _entry);
        auto *done = ::llvm::BasicBlock::Create(
            context, "scheduler.done", _entry);
        auto *exit = ::llvm::BasicBlock::Create(
            context, "scheduler.exit", _entry);
        auto *stalled = ::llvm::BasicBlock::Create(
            context, "scheduler.stalled", _entry);
        auto *invalid = ::llvm::BasicBlock::Create(
            context, "scheduler.invalid", _entry);
        std::vector<::llvm::BasicBlock *> cases;
        cases.reserve(_source.blocks().size());
        for (auto &&block : _source.blocks()) {
            cases.emplace_back(::llvm::BasicBlock::Create(
                context,
                "schedule." + std::to_string(block.id.value), _entry));
        }
        _builder.CreateBr(_scheduler_loop);

        _builder.SetInsertPoint(_scheduler_loop);
        auto *runnable = _builder.CreateLoad(
            _runnable_mask->getAllocatedType(), _runnable_mask);
        _builder.CreateCondBr(
            _builder.CreateOrReduce(runnable),
            dispatch, done);

        _builder.SetInsertPoint(dispatch);
        _seed_lane = _safe_first_lane(runnable);
        auto *pcs = _builder.CreateLoad(
            _pc_state->getAllocatedType(), _pc_state);
        auto *pc = _builder.CreateExtractElement(pcs, _seed_lane);
        auto *dispatch_switch = _builder.CreateSwitch(
            pc,
            invalid, static_cast<unsigned>(cases.size()));
        for (auto &&block : _source.blocks()) {
            dispatch_switch->addCase(
                _builder.getInt32(block.id.value), cases[block.id.value]);
        }

        _builder.SetInsertPoint(invalid);
        auto *invalid_trap =
#if LLVM_VERSION_MAJOR >= 22
            ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, ::llvm::Intrinsic::trap);
        _builder.CreateCall(invalid_trap);
        _builder.CreateUnreachable();

        _builder.SetInsertPoint(done);
        auto *live = _builder.CreateLoad(
            _live_mask->getAllocatedType(), _live_mask);
        _builder.CreateCondBr(
            _builder.CreateOrReduce(live), stalled, exit);

        _builder.SetInsertPoint(stalled);
        auto *stalled_trap =
#if LLVM_VERSION_MAJOR >= 22
            ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            ::llvm::Intrinsic::getDeclaration(
#endif
                &_module, ::llvm::Intrinsic::trap);
        _builder.CreateCall(stalled_trap);
        _builder.CreateUnreachable();

        _builder.SetInsertPoint(exit);
        _builder.CreateRetVoid();

        for (auto &&block : _source.blocks()) {
            _builder.SetInsertPoint(cases[block.id.value]);
            _locals.clear();
            auto *current_runnable = _builder.CreateLoad(
                _runnable_mask->getAllocatedType(), _runnable_mask);
            auto *current_pcs = _builder.CreateLoad(
                _pc_state->getAllocatedType(), _pc_state);
            auto *current_tokens = _builder.CreateLoad(
                _token_state->getAllocatedType(), _token_state);
            auto *seed_token = _builder.CreateExtractElement(
                current_tokens, _seed_lane);
            _active_mask = _builder.CreateAnd(
                current_runnable,
                _builder.CreateAnd(
                    _builder.CreateICmpEQ(
                        current_pcs,
                        _builder.CreateVectorSplat(
                            _width,
                            _builder.getInt32(block.id.value))),
                    _builder.CreateICmpEQ(
                        current_tokens,
                        _builder.CreateVectorSplat(
                            _width, seed_token))));
            for (auto loop : _block_loops[block.id.value]) {
                auto *epochs = _builder.CreateLoad(
                    _loop_epochs[loop.value]->getAllocatedType(),
                    _loop_epochs[loop.value]);
                auto *seed_epoch = _builder.CreateExtractElement(
                    epochs, _seed_lane);
                _active_mask = _builder.CreateAnd(
                    _active_mask,
                    _builder.CreateICmpEQ(
                        epochs,
                        _builder.CreateVectorSplat(
                            _width, seed_epoch)));
            }
            for (auto &&instruction : block.instructions) {
                _emit_instruction(instruction);
                if (_failed()) { return; }
            }
            _emit_terminator(block.terminator);
            if (_failed()) { return; }
        }
    }

public:
    ScheduleEmitter(::llvm::Module &module,
                    const schedule::Function &source, uint32_t width,
                    std::string_view entry_name)
        : _module{module},
          _source{source},
          _width{width},
          _entry_name{entry_name},
          _layout{module.getContext(), width},
          _collectives{width},
          _builder{module.getContext()} {}

    [[nodiscard]] LLVMScheduleCodegenResult run() {
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
};

}// namespace

LLVMScheduleCodegenResult lower_schedule_to_llvm(
    ::llvm::Module &module, const schedule::Function &function,
    uint32_t specialization_width, std::string_view entry_name) {
    return ScheduleEmitter{
        module, function, specialization_width, entry_name}
        .run();
}

}// namespace luisa::compute::simd
