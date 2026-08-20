#include "llvm_schedule_emitter.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include <llvm/IR/GlobalVariable.h>

#include <luisa/ast/type_registry.h>

namespace luisa::compute::simd::detail {

namespace {

[[nodiscard]] bool is_printable_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    if (type->is_scalar()) { return !type->is_float8(); }
    if (type->is_vector() || type->is_matrix() || type->is_array()) {
        return is_printable_type(type->element());
    }
    if (type->is_structure()) {
        return std::all_of(
            type->members().begin(), type->members().end(),
            [](const Type *member) noexcept {
                return is_printable_type(member);
            });
    }
    return false;
}

}// namespace

void ScheduleEmitter::_store_debug_data(
    ::llvm::Value *base, const Type *type,
    ::llvm::Value *value, size_t offset) {
    if (base == nullptr || type == nullptr || value == nullptr) {
        _fail("debug argument store received a null value or type");
        return;
    }
    if (_is_scalar_data(type)) {
        auto *pointer = _byte_pointer(base, offset);
        auto *stored = value;
        auto alignment = type->alignment();
        if (type->is_bool()) {
            stored = _builder.CreateZExt(
                value, _builder.getInt8Ty(), "debug.bool.byte");
            alignment = 1u;
        }
        auto *store = _builder.CreateStore(stored, pointer);
        store->setAlignment(::llvm::Align{alignment});
        return;
    }
    if (!is_printable_type(type)) {
        _fail("debug argument has an unsupported data type");
        return;
    }
    for (auto i = uint32_t{0u}; i < _child_count(type); i++) {
        auto *child_type = _child_type(type, i);
        _store_debug_data(
            base, child_type,
            _extract_child(value, type, i, false),
            offset + _child_offset(type, i));
        if (_failed()) { return; }
    }
}

void ScheduleEmitter::_print(
    const schedule::Instruction &instruction) {
    if (instruction.result || !instruction.message) {
        _fail("print instruction is missing its format or has a result");
        return;
    }

    auto [format_iter, inserted] = _print_format_ids.try_emplace(
        &instruction,
        _print_format_id_base + _result.print_formats.size());
    auto format_id = format_iter->second;
    if (inserted) {
        SIMDLLVMPrintFormat format{.format = *instruction.message};
        format.argument_types.reserve(instruction.operands.size());
        for (auto operand : instruction.operands) {
            auto *value = _source.value(operand);
            if (value == nullptr || !is_printable_type(value->type)) {
                _fail("print instruction has an unsupported operand type");
                return;
            }
            format.argument_types.emplace_back(value->type);
        }
        _result.print_formats.emplace_back(std::move(format));
    }

    std::vector<size_t> offsets;
    offsets.reserve(instruction.operands.size());
    auto argument_size = size_t{0u};
    for (auto operand : instruction.operands) {
        auto *value = _source.value(operand);
        if (value == nullptr || value->type == nullptr) {
            _fail("print instruction references an invalid operand");
            return;
        }
        argument_size = _align_up(
            argument_size, value->type->alignment());
        offsets.emplace_back(argument_size);
        argument_size += value->type->size();
    }
    argument_size = _align_up(argument_size, 16u);
    auto storage_iter = _print_argument_storage.find(&instruction);
    ::llvm::AllocaInst *storage = nullptr;
    if (storage_iter == _print_argument_storage.end()) {
        auto *storage_type = ::llvm::ArrayType::get(
            _builder.getInt8Ty(), std::max(argument_size, size_t{1u}));
        storage = _entry_scratch(
            storage_type,
            "print.arguments." + std::to_string(format_id));
        storage->setAlignment(::llvm::Align{16u});
        _print_argument_storage.emplace(&instruction, storage);
    } else {
        storage = storage_iter->second;
    }

    struct Operand {
        const schedule::Value *metadata{nullptr};
        ::llvm::Value *value{nullptr};
    };
    std::vector<Operand> operands;
    operands.reserve(instruction.operands.size());
    for (auto operand : instruction.operands) {
        auto *metadata = _source.value(operand);
        auto *value = _load_value(operand);
        if (metadata == nullptr || value == nullptr) { return; }
        if (!schedule::is_uniform(metadata->value_class)) {
            value = _as_lane_vector(value, *metadata);
            if (value == nullptr) { return; }
        }
        operands.emplace_back(Operand{metadata, value});
    }

    auto *pointer_type = ::llvm::PointerType::getUnqual(
        _module.getContext());
    auto load_launch_pointer = [&](size_t offset,
                                   const char *name) {
        auto *address = _builder.CreateConstInBoundsGEP1_64(
            _builder.getInt8Ty(), _launch_config, offset,
            std::string{name} + ".address");
        auto *value = _builder.CreateLoad(
            pointer_type, address, name);
        value->setAlignment(::llvm::Align{alignof(void *)});
        return value;
    };
    auto *context = load_launch_pointer(
        offsetof(SIMDPacketLaunchConfig, debug_context),
        "debug.context");
    auto *callback = load_launch_pointer(
        offsetof(SIMDPacketLaunchConfig, print_callback),
        "debug.print.callback");
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(),
        {pointer_type, _builder.getInt64Ty(), pointer_type}, false);

    for (auto lane = uint32_t{0u}; lane < _width; lane++) {
        auto *call_block = ::llvm::BasicBlock::Create(
            _module.getContext(), "print.active", _entry);
        auto *continue_block = ::llvm::BasicBlock::Create(
            _module.getContext(), "print.continue", _entry);
        auto *active = _builder.CreateExtractElement(
            _active_mask, lane, "print.lane.active");
        _builder.CreateCondBr(active, call_block, continue_block);

        _builder.SetInsertPoint(call_block);
        for (auto i = size_t{0u}; i < operands.size(); i++) {
            auto &&operand = operands[i];
            auto *value = operand.value;
            if (!schedule::is_uniform(
                    operand.metadata->value_class)) {
                value = _extract_lane(
                    value, operand.metadata->type,
                    _builder.getInt32(lane));
            }
            _store_debug_data(
                storage, operand.metadata->type,
                value, offsets[i]);
            if (_failed()) { return; }
        }
        _builder.CreateCall(
            callback_type, callback,
            {context, _builder.getInt64(format_id), storage});
        _builder.CreateBr(continue_block);
        _builder.SetInsertPoint(continue_block);
    }
}

void ScheduleEmitter::_assert(
    const schedule::Instruction &instruction) {
    if (instruction.result || instruction.operands.size() != 1u ||
        !instruction.message) {
        _fail("assert instruction is malformed");
        return;
    }
    auto *condition_metadata = _source.value(
        instruction.operands.front());
    auto *condition = _load_value(instruction.operands.front());
    if (condition_metadata == nullptr || condition == nullptr ||
        condition_metadata->type != Type::of<bool>()) {
        _fail("assert instruction requires one Boolean operand");
        return;
    }

    ::llvm::Value *passes = nullptr;
    if (schedule::is_uniform(condition_metadata->value_class)) {
        passes = condition;
    } else {
        auto *lanes = _as_lane_vector(
            condition, *condition_metadata);
        if (lanes == nullptr) { return; }
        auto *safe = _builder.CreateSelect(
            _active_mask, lanes,
            ::llvm::Constant::getAllOnesValue(
                _layout.mask_type()),
            "assert.active.conditions");
        passes = _builder.CreateAndReduce(safe);
    }

    auto *failure = ::llvm::BasicBlock::Create(
        _module.getContext(), "assert.failure", _entry);
    auto *continuation = ::llvm::BasicBlock::Create(
        _module.getContext(), "assert.continue", _entry);
    _builder.CreateCondBr(passes, continuation, failure);

    _builder.SetInsertPoint(failure);
    auto *message_data = ::llvm::ConstantDataArray::getString(
        _module.getContext(), *instruction.message, true);
    auto *message = new ::llvm::GlobalVariable(
        _module, message_data->getType(), true,
        ::llvm::GlobalValue::PrivateLinkage,
        message_data, "simd.assert.message");
    message->setUnnamedAddr(
        ::llvm::GlobalValue::UnnamedAddr::Global);
    message->setAlignment(::llvm::Align{1u});
    auto *message_pointer = _builder.CreateConstInBoundsGEP2_32(
        message_data->getType(), message, 0u, 0u,
        "assert.message");

    auto *pointer_type = ::llvm::PointerType::getUnqual(
        _module.getContext());
    auto *callback_address = _builder.CreateConstInBoundsGEP1_64(
        _builder.getInt8Ty(), _launch_config,
        offsetof(SIMDPacketLaunchConfig, assert_fail_callback),
        "assert.callback.address");
    auto *callback = _builder.CreateLoad(
        pointer_type, callback_address, "assert.callback");
    callback->setAlignment(::llvm::Align{alignof(void *)});
    auto *call_callback = ::llvm::BasicBlock::Create(
        _module.getContext(), "assert.callback.present", _entry);
    auto *trap = ::llvm::BasicBlock::Create(
        _module.getContext(), "assert.trap", _entry);
    _builder.CreateCondBr(
        _builder.CreateIsNotNull(callback), call_callback, trap);

    _builder.SetInsertPoint(call_callback);
    auto *callback_type = ::llvm::FunctionType::get(
        _builder.getVoidTy(), {pointer_type}, false);
    _builder.CreateCall(
        callback_type, callback, {message_pointer});
    _builder.CreateBr(trap);

    _builder.SetInsertPoint(trap);
    _builder.CreateIntrinsic(
        _builder.getVoidTy(), ::llvm::Intrinsic::trap, {});
    _builder.CreateUnreachable();
    _builder.SetInsertPoint(continuation);
}

[[nodiscard]] ::llvm::Value *ScheduleEmitter::_clock(
    const schedule::Instruction &instruction) {
    if (!instruction.result || !instruction.operands.empty() ||
        instruction.message) {
        _fail("clock instruction is malformed");
        return nullptr;
    }
    auto *result = _source.value(*instruction.result);
    if (result == nullptr || result->type != Type::of<uint64_t>()) {
        _fail("clock instruction requires a uint64 result");
        return nullptr;
    }
    auto *counter = _builder.CreateIntrinsic(
        _builder.getInt64Ty(), ::llvm::Intrinsic::readcyclecounter, {});
    return schedule::is_uniform(result->value_class) ?
               counter :
               _builder.CreateVectorSplat(_width, counter);
}

}// namespace luisa::compute::simd::detail
