#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <array>
#include <bit>
#include <unordered_map>

#include <llvm/IR/GlobalVariable.h>

#include "../../common/env_flag.h"
#include "../../common/indirect_dispatch_layout.h"
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::simd::detail {

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_lane_offsets(
    ::llvm::Value *index, uint64_t stride) {
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *extended = _builder.CreateZExtOrTrunc(index, i64_lanes);
    return stride == 1u ? extended : _builder.CreateMul(extended, _builder.CreateVectorSplat(_width, _builder.getInt64(stride)));
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_lane_offsets_i32(
    ::llvm::Value *index, uint32_t stride) {
    auto *i32_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt32Ty(), _width);
    auto *extended = _builder.CreateZExtOrTrunc(index, i32_lanes);
    return stride == 1u ? extended : _builder.CreateMul(extended, _builder.CreateVectorSplat(_width, _builder.getInt32(stride)));
}

[[nodiscard]] luisa::optional<uint64_t> ScheduleEmitter::_constant_aggregate_index(
    schedule::ValueId id) const noexcept {
    auto *value = _source.value(id);
    if (value == nullptr ||
        value->origin != schedule::ValueOrigin::constant ||
        value->type == nullptr || !value->type->is_scalar() ||
        value->type->is_float() || value->type->size() == 0u ||
        value->type->size() > sizeof(uint64_t)) {
        return luisa::nullopt;
    }
    auto *metadata = std::get_if<schedule::ConstantValueMetadata>(
        &value->metadata);
    if (metadata == nullptr ||
        metadata->bytes.size() < value->type->size()) {
        return luisa::nullopt;
    }
    auto result = uint64_t{0u};
    std::memcpy(
        &result, metadata->bytes.data(), value->type->size());
    auto bits = static_cast<uint32_t>(value->type->size() * 8u);
    if (bits < 64u) {
        result &= (uint64_t{1u} << bits) - 1u;
    }
    if (value->type->is_int() &&
        (result & (uint64_t{1u} << (bits - 1u))) != 0u) {
        return luisa::nullopt;
    }
    return result;
}

[[nodiscard]] bool ScheduleEmitter::_advance_aggregate_offset(
    ::llvm::Value *&offsets, const Type *&current_type,
    schedule::ValueId index_id, uint32_t scale) {
    auto *index_value = _source.value(index_id);
    if (index_value == nullptr || index_value->type == nullptr ||
        !index_value->type->is_scalar() ||
        index_value->type->is_float() || current_type == nullptr ||
        _child_count(current_type) == 0u) {
        _fail("aggregate address has an invalid index or type path");
        return false;
    }
    if (current_type->is_structure()) {
        auto index = _constant_aggregate_index(index_id);
        if (!index || *index >= _child_count(current_type)) {
            _fail("structure address requires a valid constant member index");
            return false;
        }
        auto child_offset = _child_offset(
            current_type, static_cast<uint32_t>(*index));
        if (child_offset != 0u) {
            offsets = _builder.CreateAdd(
                offsets,
                _builder.CreateVectorSplat(
                    _width, _builder.getInt64(child_offset * scale)));
        }
        current_type = _child_type(
            current_type, static_cast<uint32_t>(*index));
        return true;
    }

    auto *index = _as_lane_vector(
        _load_value(index_id), *index_value);
    if (index == nullptr) { return false; }
    auto *i64_lanes = ::llvm::FixedVectorType::get(
        _builder.getInt64Ty(), _width);
    auto *extended = index_value->type->is_int() ?
                         _builder.CreateSExtOrTrunc(index, i64_lanes) :
                         _builder.CreateZExtOrTrunc(index, i64_lanes);
    auto stride = current_type->is_vector() ?
                      current_type->element()->size() :
                      current_type->size() / current_type->dimension();
    stride *= scale;
    if (stride != 1u) {
        extended = _builder.CreateMul(
            extended,
            _builder.CreateVectorSplat(
                _width, _builder.getInt64(stride)));
    }
    offsets = _builder.CreateAdd(offsets, extended);
    current_type = _child_type(current_type, 0u);
    return true;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_alloca(
    const schedule::Instruction &instruction) {
    if (!instruction.result ||
        !_is_local_lvalue(*instruction.result) ||
        instruction.result->value >= _local_allocations.size()) {
        _fail("thread-local allocation is malformed");
        return nullptr;
    }
    auto *handle = _local_allocations[instruction.result->value];
    if (handle == nullptr) {
        _fail("thread-local allocation has no packet storage");
    }
    return handle;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_gep(
    const schedule::Instruction &instruction) {
    if (!instruction.result || instruction.operands.empty()) {
        _fail("thread-local GEP is malformed");
        return nullptr;
    }
    auto base_id = instruction.operands.front();
    auto res_id = *instruction.result;
    auto *base_value = _source.value(base_id);
    auto *result_value = _source.value(res_id);
    auto *handle = _load_value(base_id);
    if (base_value == nullptr || result_value == nullptr ||
        handle == nullptr || base_value->type == nullptr) {
        _fail("thread-local GEP has invalid values");
        return nullptr;
    }
    auto base_kind = _local_soa_kind_of(base_id);
    auto res_kind = _local_soa_kind_of(res_id);
    auto cell = _local_soa_cell_of(res_id);
    auto *current_type = base_value->type;
    auto soa_stride = _soa_stride();
    if (res_kind == LocalSoAKind::uniform) {
        auto *ptr = handle;
        for (auto i = size_t{1u}; i < instruction.operands.size(); i++) {
            auto idx_id = instruction.operands[i];
            if (current_type->is_structure()) {
                auto idx = _constant_aggregate_index(idx_id);
                if (!idx || *idx >= _child_count(current_type)) {
                    _fail("structure address requires a valid constant member index");
                    return nullptr;
                }
                auto child_offset = _child_offset(
                    current_type, static_cast<uint32_t>(*idx));
                if (child_offset != 0u) {
                    ptr = _builder.CreateGEP(
                        _builder.getInt8Ty(), ptr,
                        _builder.getInt64(child_offset * soa_stride));
                }
                current_type = _child_type(
                    current_type, static_cast<uint32_t>(*idx));
                continue;
            }
            auto stride = static_cast<uint64_t>(
                              current_type->is_vector() ?
                                  current_type->element()->size() :
                                  current_type->size() / current_type->dimension()) *
                          soa_stride;
            if (auto c = _constant_aggregate_index(idx_id)) {
                if (*c != 0u) {
                    ptr = _builder.CreateGEP(
                        _builder.getInt8Ty(), ptr,
                        _builder.getInt64(*c * stride));
                }
            } else {
                auto *idx_val_meta = _source.value(idx_id);
                auto *idx_val = _load_value(idx_id);
                if (idx_val_meta == nullptr || idx_val == nullptr) {
                    _fail("uniform SoA GEP has an invalid index");
                    return nullptr;
                }
                if (idx_val->getType()->isVectorTy()) {
                    auto *seed = _seed_lane ?
                                     _seed_lane :
                                     _safe_first_lane(_active_mask);
                    idx_val = _builder.CreateExtractElement(idx_val, seed);
                }
                auto *ext = idx_val_meta->type->is_int() ?
                                _builder.CreateSExtOrTrunc(
                                    idx_val, _builder.getInt64Ty()) :
                                _builder.CreateZExtOrTrunc(
                                    idx_val, _builder.getInt64Ty());
                auto *byte_off = stride == 1u ?
                                     ext :
                                     _builder.CreateMul(
                                         ext, _builder.getInt64(stride));
                byte_off = _builder.CreateSelect(
                    _builder.CreateOrReduce(_active_mask), byte_off,
                    _builder.getInt64(0u));
                ptr = _builder.CreateGEP(_builder.getInt8Ty(), ptr, byte_off);
            }
            current_type = _child_type(current_type, 0u);
        }
        if (current_type != result_value->type) {
            _fail("thread-local GEP result type does not match its index path");
            return nullptr;
        }
        return ptr;
    }
    if (res_kind == LocalSoAKind::varying) {
        auto *i32_lanes = ::llvm::FixedVectorType::get(
            _builder.getInt32Ty(), _width);
        auto *base_ptr = base_kind == LocalSoAKind::uniform ?
                             handle :
                             _local_base(_builder, handle);
        auto *offsets = base_kind == LocalSoAKind::uniform ?
                            static_cast<::llvm::Value *>(
                                ::llvm::Constant::getNullValue(i32_lanes)) :
                            _local_offsets(_builder, handle);
        for (auto i = size_t{1u}; i < instruction.operands.size(); i++) {
            auto idx_id = instruction.operands[i];
            if (current_type->is_structure()) {
                auto idx = _constant_aggregate_index(idx_id);
                if (!idx || *idx >= _child_count(current_type)) {
                    _fail("structure address requires a valid constant member index");
                    return nullptr;
                }
                auto child_offset = _child_offset(
                    current_type, static_cast<uint32_t>(*idx));
                if (child_offset != 0u) {
                    offsets = _builder.CreateAdd(
                        offsets,
                        _builder.CreateVectorSplat(
                            _width,
                            _builder.getInt32(static_cast<uint32_t>(
                                child_offset * soa_stride))));
                }
                current_type = _child_type(
                    current_type, static_cast<uint32_t>(*idx));
                continue;
            }
            auto *idx_val_meta = _source.value(idx_id);
            auto *idx_vec = idx_val_meta == nullptr ?
                                nullptr :
                                _as_lane_vector(
                                    _load_value(idx_id), *idx_val_meta);
            if (idx_vec == nullptr) { return nullptr; }
            auto *ext = idx_val_meta->type->is_int() ?
                            _builder.CreateSExtOrTrunc(idx_vec, i32_lanes) :
                            _builder.CreateZExtOrTrunc(idx_vec, i32_lanes);
            auto stride = static_cast<uint32_t>(
                              current_type->is_vector() ?
                                  current_type->element()->size() :
                                  current_type->size() / current_type->dimension()) *
                          soa_stride;
            if (stride != 1u) {
                ext = _builder.CreateMul(
                    ext, _builder.CreateVectorSplat(
                             _width, _builder.getInt32(stride)));
            }
            offsets = _builder.CreateAdd(offsets, ext);
            current_type = _child_type(current_type, 0u);
        }
        if (current_type != result_value->type) {
            _fail("thread-local GEP result type does not match its index path");
            return nullptr;
        }
        return _soa_varying_handle(base_ptr, offsets);
    }
    auto *base = _local_base(_builder, handle);
    auto *offsets = _local_offsets(_builder, handle);
    for (auto i = size_t{1u}; i < instruction.operands.size(); i++) {
        if (!_advance_aggregate_offset(
                offsets, current_type, instruction.operands[i], 1u)) {
            return nullptr;
        }
    }
    if (current_type != result_value->type) {
        _fail("thread-local GEP result type does not match its index path");
        return nullptr;
    }
    return _local_handle(base, offsets);
}

void ScheduleEmitter::_analyze_local_soa() {
    std::vector<std::vector<uint8_t>> callee_cells;
    callee_cells.reserve(_outlined_callables.size());
    for (auto &&callee : _outlined_callables) {
        callee_cells.emplace_back(callee.parameter_soa_cell_bytes);
    }
    auto analysis = analyze_local_soa(
        _source, _width,
        _is_outlined_callable_entry() ?
            _outlined_parameter_soa_cell_bytes :
            luisa::span<const uint8_t>{},
        callee_cells, _enable_interleaved_private_arrays, _soa_stride());
    _local_soa_kind = std::move(analysis.kind);
    _local_soa_cell_bytes = std::move(analysis.cell_bytes);
    _result.interleaved_private_arrays =
        analysis.interleaved_allocation_count;
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_load_uniform_soa_data(
    ::llvm::Value *ptr, const Type *type,
    uint8_t cell_bytes, size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto *element = type->is_bool() ?
                            static_cast<::llvm::Type *>(_builder.getInt8Ty()) :
                            _data_type(type, false);
        auto stride = _soa_stride();
        auto leaf_sz = type->size();
        auto *addr = leaf_offset == 0u ?
                         ptr :
                         _builder.CreateGEP(
                             _builder.getInt8Ty(), ptr,
                             _builder.getInt64(leaf_offset * stride));
        if (_width == 1u && _outlined_soa_lane != nullptr) {
            auto *lane64 = _builder.CreateZExt(
                _outlined_soa_lane, _builder.getInt64Ty());
            auto *lane_off = leaf_sz == 1u ?
                                 lane64 :
                                 _builder.CreateMul(
                                     lane64, _builder.getInt64(leaf_sz));
            auto *lane_addr = _builder.CreateGEP(
                _builder.getInt8Ty(), addr, lane_off);
            ::llvm::Value *scalar = _builder.CreateAlignedLoad(
                element, lane_addr, ::llvm::Align{type->alignment()},
                "private.soa.w1.load");
            if (type->is_bool()) {
                scalar = _builder.CreateICmpNE(
                    scalar, ::llvm::Constant::getNullValue(element));
            }
            return _builder.CreateInsertElement(
                ::llvm::PoisonValue::get(
                    ::llvm::FixedVectorType::get(scalar->getType(), 1u)),
                scalar, _builder.getInt32(0u));
        }
        auto *lanes = ::llvm::FixedVectorType::get(element, _width);
        auto vector_alignment = std::max<size_t>(
            type->alignment(),
            std::min<size_t>(
                static_cast<size_t>(_width) * leaf_sz, size_t{64u}));
        ::llvm::Value *loaded = _builder.CreateAlignedLoad(
            lanes, addr, ::llvm::Align{vector_alignment},
            "private.contiguous.load");
        if (type->is_bool()) {
            loaded = _builder.CreateICmpNE(
                loaded, ::llvm::Constant::getNullValue(lanes));
        }
        return _builder.CreateSelect(
            _active_mask, loaded,
            ::llvm::Constant::getNullValue(loaded->getType()));
    }
    return _assemble(type, true, [&](uint32_t i) {
        return _load_uniform_soa_data(
            ptr, _child_type(type, i), cell_bytes,
            leaf_offset + _child_offset(type, i));
    });
}

void ScheduleEmitter::_store_uniform_soa_data(
    ::llvm::Value *ptr, const Type *type, ::llvm::Value *value,
    uint8_t cell_bytes, size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto stride = _soa_stride();
        auto leaf_sz = type->size();
        auto *addr = leaf_offset == 0u ?
                         ptr :
                         _builder.CreateGEP(
                             _builder.getInt8Ty(), ptr,
                             _builder.getInt64(leaf_offset * stride));
        if (_width == 1u && _outlined_soa_lane != nullptr) {
            auto *lane64 = _builder.CreateZExt(
                _outlined_soa_lane, _builder.getInt64Ty());
            auto *lane_off = leaf_sz == 1u ?
                                 lane64 :
                                 _builder.CreateMul(
                                     lane64, _builder.getInt64(leaf_sz));
            auto *lane_addr = _builder.CreateGEP(
                _builder.getInt8Ty(), addr, lane_off);
            auto *scalar = _builder.CreateExtractElement(
                value, _builder.getInt32(0u));
            if (type->is_bool()) {
                scalar = _builder.CreateZExt(scalar, _builder.getInt8Ty());
            }
            _builder.CreateAlignedStore(
                scalar, lane_addr, ::llvm::Align{type->alignment()});
            return;
        }
        auto vector_alignment = std::max<size_t>(
            type->alignment(),
            std::min<size_t>(
                static_cast<size_t>(_width) * leaf_sz, size_t{64u}));
        auto alignment = ::llvm::Align{vector_alignment};
        auto *stored = type->is_bool() ?
                           _builder.CreateZExt(
                               value,
                               ::llvm::FixedVectorType::get(
                                   _builder.getInt8Ty(), _width)) :
                           value;
        auto *previous = _builder.CreateAlignedLoad(
            stored->getType(), addr, alignment,
            "private.soa.uniform.preserve");
        _builder.CreateAlignedStore(
            _builder.CreateSelect(_active_mask, stored, previous),
            addr, alignment);
        return;
    }
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        _store_uniform_soa_data(
            ptr, _child_type(type, i),
            _extract_child(value, type, i, true), cell_bytes,
            leaf_offset + _child_offset(type, i));
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_gather_varying_soa_data(
    ::llvm::Value *base, ::llvm::Value *offsets, const Type *type,
    uint8_t cell_bytes, size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto leaf_sz = static_cast<uint32_t>(type->size());
        std::vector<::llvm::Constant *> lane_const;
        lane_const.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; ++lane) {
            lane_const.emplace_back(_builder.getInt32(
                static_cast<uint32_t>(leaf_offset * _width) + lane * leaf_sz));
        }
        auto *off = _builder.CreateAdd(
            offsets, ::llvm::ConstantVector::get(lane_const));
        auto *pointers = _builder.CreateGEP(_builder.getInt8Ty(), base, off);
        auto *element = type->is_bool() ?
                            static_cast<::llvm::Type *>(_builder.getInt8Ty()) :
                            _data_type(type, false);
        auto *lanes = ::llvm::FixedVectorType::get(element, _width);
        auto *gathered = _builder.CreateMaskedGather(
            lanes, pointers, ::llvm::Align{leaf_sz}, _active_mask,
            ::llvm::Constant::getNullValue(lanes),
            "private.soa.varying.gather");
        return type->is_bool() ?
                   _builder.CreateICmpNE(
                       gathered, ::llvm::Constant::getNullValue(lanes)) :
                   gathered;
    }
    return _assemble(type, true, [&](uint32_t i) {
        return _gather_varying_soa_data(
            base, offsets, _child_type(type, i), cell_bytes,
            leaf_offset + _child_offset(type, i));
    });
}

void ScheduleEmitter::_scatter_varying_soa_data(
    ::llvm::Value *base, ::llvm::Value *offsets, const Type *type,
    ::llvm::Value *value, uint8_t cell_bytes, size_t leaf_offset) {
    if (_is_scalar_data(type)) {
        auto leaf_sz = static_cast<uint32_t>(type->size());
        std::vector<::llvm::Constant *> lane_const;
        lane_const.reserve(_width);
        for (auto lane = uint32_t{0u}; lane < _width; ++lane) {
            lane_const.emplace_back(_builder.getInt32(
                static_cast<uint32_t>(leaf_offset * _width) + lane * leaf_sz));
        }
        auto *off = _builder.CreateAdd(
            offsets, ::llvm::ConstantVector::get(lane_const));
        auto *pointers = _builder.CreateGEP(_builder.getInt8Ty(), base, off);
        auto *stored = type->is_bool() ?
                           _builder.CreateZExt(
                               value,
                               ::llvm::FixedVectorType::get(
                                   _builder.getInt8Ty(), _width)) :
                           value;
        _builder.CreateMaskedScatter(
            stored, pointers, ::llvm::Align{leaf_sz}, _active_mask);
        return;
    }
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        _scatter_varying_soa_data(
            base, offsets, _child_type(type, i),
            _extract_child(value, type, i, true), cell_bytes,
            leaf_offset + _child_offset(type, i));
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_local_load(
    const schedule::Instruction &instruction) {
    if (!instruction.result || instruction.operands.size() != 1u) {
        _fail("thread-local load is malformed");
        return nullptr;
    }
    auto ptr_id = instruction.operands.front();
    auto *variable = _source.value(ptr_id);
    auto *result = _source.value(*instruction.result);
    auto *handle = _load_value(ptr_id);
    if (variable == nullptr || result == nullptr || handle == nullptr ||
        variable->type != result->type) {
        _fail("thread-local load has mismatched value types");
        return nullptr;
    }
    auto kind = _local_soa_kind_of(ptr_id);
    auto cell = _local_soa_cell_of(ptr_id);
    if (kind == LocalSoAKind::uniform) {
        if (_enable_contiguous_private_access ||
            (_width == 1u && _outlined_soa_lane != nullptr)) {
            _result.contiguous_private_read_count++;
            return _load_uniform_soa_data(handle, result->type, cell);
        }
        auto *zero_off = ::llvm::Constant::getNullValue(
            ::llvm::FixedVectorType::get(_builder.getInt32Ty(), _width));
        return _gather_varying_soa_data(handle, zero_off, result->type, cell);
    }
    if (kind == LocalSoAKind::varying) {
        return _gather_varying_soa_data(
            _local_base(_builder, handle), _local_offsets(_builder, handle),
            result->type, cell);
    }
    return _gather_data(
        _local_base(_builder, handle),
        _local_offsets(_builder, handle), result->type);
}

void ScheduleEmitter::_local_store(const schedule::Instruction &instruction) {
    if (instruction.operands.size() != 2u) {
        _fail("thread-local store is malformed");
        return;
    }
    auto ptr_id = instruction.operands[0u];
    auto *variable = _source.value(ptr_id);
    auto *written_value = _source.value(instruction.operands[1u]);
    auto *handle = _load_value(ptr_id);
    auto *written = written_value == nullptr ? nullptr :
                                               _as_lane_vector(
                                                   _load_value(instruction.operands[1u]), *written_value);
    if (variable == nullptr || written_value == nullptr ||
        handle == nullptr || written == nullptr ||
        variable->type != written_value->type) {
        _fail("thread-local store has mismatched value types");
        return;
    }
    auto kind = _local_soa_kind_of(ptr_id);
    auto cell = _local_soa_cell_of(ptr_id);
    if (kind == LocalSoAKind::uniform) {
        if (_enable_contiguous_private_access ||
            (_width == 1u && _outlined_soa_lane != nullptr)) {
            _result.contiguous_private_write_count++;
            _store_uniform_soa_data(handle, written_value->type, written, cell);
        } else {
            auto *zero_off = ::llvm::Constant::getNullValue(
                ::llvm::FixedVectorType::get(_builder.getInt32Ty(), _width));
            _scatter_varying_soa_data(
                handle, zero_off, written_value->type, written, cell);
        }
    } else if (kind == LocalSoAKind::varying) {
        _scatter_varying_soa_data(
            _local_base(_builder, handle), _local_offsets(_builder, handle),
            written_value->type, written, cell);
    } else {
        _scatter_data(
            _local_base(_builder, handle), _local_offsets(_builder, handle),
            written_value->type, written);
    }
    // A proven ray-query sidecar becomes valid only when the pointer value is
    // actually installed in its thread-local owner. Construction may precede
    // this masked store, so publishing validity in _ray_query_create would let
    // an intervening read observe the new status with the old object pointer.
    if (_ray_query_status_slot(instruction.operands[0u]) != nullptr) {
        auto *state_handle_slot = _ray_query_state_handle_slot(
            instruction.operands[0u]);
        if (state_handle_slot != nullptr) {
            // Publish only the lanes whose authoritative local pointer was
            // just written. Other divergent cohorts sharing this color keep
            // their handles until their own masked construction store.
            auto *old_states = _builder.CreateAlignedLoad(
                written->getType(), state_handle_slot,
                ::llvm::Align{alignof(void *)});
            _builder.CreateAlignedStore(
                _builder.CreateSelect(
                    _active_mask, written, old_states),
                state_handle_slot, ::llvm::Align{alignof(void *)});
        }
        _ray_query_update_status(
            instruction.operands[0u], _builder.getInt64(0u));
    }
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_atomic_lanes(
    xir::AtomicOp op, const schedule::Value &result,
    ::llvm::Value *base, ::llvm::Value *offsets,
    const std::vector<::llvm::Value *> &values) {
    auto *result_type = _layout.expression_type(result);
    if (result_type == nullptr || !result_type->isVectorTy()) {
        _fail("packet atomic result must be lane-varying");
        return nullptr;
    }
    ::llvm::Value *old_values =
        ::llvm::Constant::getNullValue(result_type);
    auto alignment = ::llvm::MaybeAlign{result.type->alignment()};
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
        auto *lane_base = base->getType()->isVectorTy() ?
                              _builder.CreateExtractElement(base, lane) :
                              base;
        auto *pointer = _builder.CreateGEP(
            _builder.getInt8Ty(), lane_base, offset);
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
            auto floating = result.type->is_float16() ||
                            result.type->is_float32() ||
                            result.type->is_float64();
            auto signed_integer = result.type->is_int();
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
                _fail("unsupported packet atomic operation");
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

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_shared_atomic(
    const schedule::Instruction &instruction) {
    auto op = static_cast<xir::AtomicOp>(*instruction.source_op);
    auto value_count = xir::atomic_op_value_count(op);
    auto index_count =
        instruction.operands.size() - 1u - value_count;
    auto *result = _source.value(*instruction.result);
    auto base_id = instruction.operands.front();
    auto *base_value = _source.value(base_id);
    auto *handle = _load_value(base_id);
    if (result == nullptr || base_value == nullptr ||
        base_value->type == nullptr || handle == nullptr ||
        !_is_shared_lvalue(base_id) ||
        !_is_scalar_data(result->type)) {
        _fail("shared atomic has an invalid target");
        return nullptr;
    }
    auto *base = _local_base(_builder, handle);
    auto *offsets = _local_offsets(_builder, handle);
    auto *current_type = base_value->type;
    for (auto i = size_t{0u}; i < index_count; i++) {
        if (!_advance_aggregate_offset(
                offsets, current_type,
                instruction.operands[1u + i])) {
            return nullptr;
        }
    }
    if (current_type != result->type) {
        _fail("shared atomic result type does not match its aggregate index path");
        return nullptr;
    }
    std::vector<::llvm::Value *> values;
    values.reserve(value_count);
    for (auto i = size_t{0u}; i < value_count; i++) {
        auto operand_id = instruction.operands[instruction.operands.size() - value_count + i];
        auto *operand = _source.value(operand_id);
        auto *value = operand == nullptr ?
                          nullptr :
                          _as_lane_vector(
                              _load_value(operand_id), *operand);
        if (value == nullptr) { return nullptr; }
        values.emplace_back(value);
    }
    return _atomic_lanes(op, *result, base, offsets, values);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_atomic(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.source_op) {
        _fail("atomic instruction is missing a result or operation");
        return nullptr;
    }
    auto op = static_cast<xir::AtomicOp>(*instruction.source_op);
    auto value_count = xir::atomic_op_value_count(op);
    if (instruction.operands.size() < 1u + value_count) {
        _fail("atomic instruction has an invalid operand count");
        return nullptr;
    }
    if (_is_shared_lvalue(instruction.operands.front())) {
        return _shared_atomic(instruction);
    }
    auto index_count =
        instruction.operands.size() - 1u - value_count;
    auto *result = _source.value(*instruction.result);
    auto *buffer_value = _source.value(instruction.operands[0u]);
    auto *buffer = _load_value(instruction.operands[0u]);
    if (result == nullptr || buffer_value == nullptr ||
        buffer_value->type == nullptr || buffer == nullptr ||
        !buffer_value->type->is_buffer() ||
        buffer_value->type->element() == nullptr ||
        index_count == 0u || !_is_scalar_data(result->type)) {
        _fail("LLVM packet codegen requires a scalar typed-buffer atomic target");
        return nullptr;
    }

    auto *current_type = buffer_value->type->element();
    ::llvm::Value *offsets = nullptr;
    for (auto i = size_t{0u}; i < index_count; i++) {
        auto index_id = instruction.operands[1u + i];
        auto *index_value = _source.value(index_id);
        if (index_value == nullptr || index_value->type == nullptr ||
            !index_value->type->is_scalar() ||
            index_value->type->is_float()) {
            _fail("buffer atomic has an invalid aggregate index");
            return nullptr;
        }
        if (i == 0u) {
            auto *index = _as_lane_vector(
                _load_value(index_id), *index_value);
            if (index == nullptr) { return nullptr; }
            offsets = _lane_offsets(
                index,
                static_cast<uint64_t>(current_type->size()));
            continue;
        }
        if (!_advance_aggregate_offset(
                offsets, current_type, index_id)) {
            return nullptr;
        }
    }
    if (current_type != result->type) {
        _fail("buffer atomic result type does not match its aggregate index path");
        return nullptr;
    }
    std::vector<::llvm::Value *> values;
    values.reserve(value_count);
    for (auto i = size_t{0u}; i < value_count; i++) {
        auto operand_id = instruction.operands[instruction.operands.size() - value_count + i];
        auto *operand = _source.value(operand_id);
        auto *value = operand == nullptr ?
                          nullptr :
                          _as_lane_vector(
                              _load_value(operand_id), *operand);
        if (value == nullptr) { return nullptr; }
        values.emplace_back(value);
    }
    auto *base = _builder.CreateExtractValue(buffer, {0u});
    return _atomic_lanes(op, *result, base, offsets, values);
}

}// namespace luisa::compute::simd::detail
