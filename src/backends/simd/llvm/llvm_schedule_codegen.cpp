#include "llvm_schedule_codegen.h"

#include <algorithm>
#include <cstring>
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
    std::vector<::llvm::Value *> _external_values{};
    std::vector<size_t> _parameter_offsets{};
    std::unordered_map<uint32_t, ::llvm::Value *> _locals{};
    ::llvm::Value *_active_mask{nullptr};
    ::llvm::Value *_seed_lane{nullptr};

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
                    metadata->argument_tag != static_cast<uint32_t>(
                                                  xir::DerivedArgumentTag::VALUE) ||
                    value.value_class != schedule::ValueClass::warp_uniform ||
                    !_is_scalar_data(value.type)) {
                    _fail("Phase-2 packet ABI supports scalar warp-uniform value arguments only");
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
                       value.type != nullptr &&
                       !_is_scalar_data(value.type)) {
                // The one currently-supported aggregate result is Luisa's
                // uint4 ballot, represented as a uniform LLVM <4 x i32>.
                auto is_ballot = value.type->is_vector() &&
                                 value.type->dimension() == 4u &&
                                 value.type->element()->is_uint32();
                if (!is_ballot) {
                    _fail("Phase-2 packet codegen currently supports scalar Schedule IR values only");
                    return;
                }
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
            offset = _align_up(offset, type->alignment());
            _parameter_offsets[index] = offset;
            offset += type->size();
        }
        _result.argument_buffer_size = offset;

        for (auto &&block : _source.blocks()) {
            for (auto &&instruction : block.instructions) {
                if (instruction.opcode != schedule::Opcode::arithmetic &&
                    instruction.opcode != schedule::Opcode::warp_collective) {
                    _fail("Phase-2 LLVM packet codegen encountered an unsupported Schedule IR instruction");
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
                    auto *type = _layout.expression_type(value);
                    auto *load = _builder.CreateLoad(type, pointer, value.name);
                    load->setAlignment(::llvm::Align{value.type->alignment()});
                    llvm_value = load;
                    break;
                }
                case schedule::ValueOrigin::constant: {
                    auto *metadata = std::get_if<
                        schedule::ConstantValueMetadata>(&value.metadata);
                    llvm_value = _scalar_constant(
                        value.type, metadata->bytes.data());
                    break;
                }
                case schedule::ValueOrigin::special_register: {
                    auto *metadata = std::get_if<
                        schedule::SpecialRegisterValueMetadata>(&value.metadata);
                    auto tag = static_cast<xir::DerivedSpecialRegisterTag>(
                        metadata->tag);
                    switch (tag) {
                        case xir::DerivedSpecialRegisterTag::WARP_LANE_ID:
                            llvm_value = _lane_ids();
                            break;
                        case xir::DerivedSpecialRegisterTag::WARP_SIZE:
                            llvm_value = _builder.getInt32(_width);
                            break;
                        default:
                            _fail("Phase-2 packet ABI does not provide this special register");
                            return;
                    }
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
                return _builder.CreateExtractElement(state, safe);
            }
            return state;
        }
        if (auto iter = _locals.find(id.value); iter != _locals.end()) {
            return iter->second;
        }
        _fail("instruction value is not available in the current Schedule IR block");
        return nullptr;
    }

    [[nodiscard]] ::llvm::Value *_as_lane_vector(
        ::llvm::Value *value, const schedule::Value &schedule_value) {
        if (value == nullptr) { return nullptr; }
        if (schedule_value.value_class == schedule::ValueClass::varying ||
            schedule_value.value_class == schedule::ValueClass::mask) {
            return value;
        }
        if (!_is_scalar_data(schedule_value.type)) {
            _fail("cannot splat an aggregate Schedule IR value yet");
            return nullptr;
        }
        return _builder.CreateVectorSplat(_width, value);
    }

    [[nodiscard]] ::llvm::Value *_arithmetic(
        const schedule::Instruction &instruction) {
        if (!instruction.result || !instruction.source_op) {
            _fail("arithmetic instruction is missing result or source operation");
            return nullptr;
        }
        auto *result = _source.value(*instruction.result);
        if (result == nullptr || !_is_scalar_data(result->type)) {
            _fail("Phase-2 arithmetic requires a scalar Luisa result type");
            return nullptr;
        }
        std::vector<::llvm::Value *> operands;
        operands.reserve(instruction.operands.size());
        for (auto operand_id : instruction.operands) {
            auto *operand = _source.value(operand_id);
            auto *llvm_operand = _load_value(operand_id);
            if (result->value_class == schedule::ValueClass::varying) {
                llvm_operand = _as_lane_vector(llvm_operand, *operand);
            }
            if (llvm_operand == nullptr) { return nullptr; }
            operands.emplace_back(llvm_operand);
        }
        auto require = [&](size_t count) {
            if (operands.size() != count) {
                _fail("arithmetic operation has an invalid operand count");
                return false;
            }
            return true;
        };
        auto op = static_cast<xir::ArithmeticOp>(*instruction.source_op);
        auto *operand_type = instruction.operands.empty() ?
                                 result->type :
                                 _source.value(instruction.operands.front())->type;
        auto is_float = operand_type->is_float16() ||
                        operand_type->is_float32() ||
                        operand_type->is_float64();
        auto is_signed = operand_type->is_int();
        switch (op) {
            case xir::ArithmeticOp::UNARY_MINUS:
                if (!require(1u)) { return nullptr; }
                return is_float ? _builder.CreateFNeg(operands[0u]) :
                                  _builder.CreateNeg(operands[0u]);
            case xir::ArithmeticOp::UNARY_BIT_NOT:
                if (!require(1u)) { return nullptr; }
                return _builder.CreateNot(operands[0u]);
            case xir::ArithmeticOp::BINARY_ADD:
                if (!require(2u)) { return nullptr; }
                return is_float ? _builder.CreateFAdd(operands[0u], operands[1u]) :
                                  _builder.CreateAdd(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_SUB:
                if (!require(2u)) { return nullptr; }
                return is_float ? _builder.CreateFSub(operands[0u], operands[1u]) :
                                  _builder.CreateSub(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_MUL:
                if (!require(2u)) { return nullptr; }
                return is_float ? _builder.CreateFMul(operands[0u], operands[1u]) :
                                  _builder.CreateMul(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_DIV:
                if (!require(2u)) { return nullptr; }
                return is_float ? _builder.CreateFDiv(operands[0u], operands[1u]) :
                       is_signed ? _builder.CreateSDiv(operands[0u], operands[1u]) :
                                   _builder.CreateUDiv(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_MOD:
                if (!require(2u)) { return nullptr; }
                return is_float ? _builder.CreateFRem(operands[0u], operands[1u]) :
                       is_signed ? _builder.CreateSRem(operands[0u], operands[1u]) :
                                   _builder.CreateURem(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_BIT_AND:
                if (!require(2u)) { return nullptr; }
                return _builder.CreateAnd(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_BIT_OR:
                if (!require(2u)) { return nullptr; }
                return _builder.CreateOr(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_BIT_XOR:
                if (!require(2u)) { return nullptr; }
                return _builder.CreateXor(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_SHIFT_LEFT:
                if (!require(2u)) { return nullptr; }
                return _builder.CreateShl(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_SHIFT_RIGHT:
                if (!require(2u)) { return nullptr; }
                return is_signed ? _builder.CreateAShr(operands[0u], operands[1u]) :
                                   _builder.CreateLShr(operands[0u], operands[1u]);
            case xir::ArithmeticOp::BINARY_LESS:
            case xir::ArithmeticOp::BINARY_GREATER:
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
            case xir::ArithmeticOp::BINARY_EQUAL:
            case xir::ArithmeticOp::BINARY_NOT_EQUAL: {
                if (!require(2u)) { return nullptr; }
                if (is_float) {
                    auto predicate = ::llvm::CmpInst::FCMP_FALSE;
                    switch (op) {
                        case xir::ArithmeticOp::BINARY_LESS:
                            predicate = ::llvm::CmpInst::FCMP_OLT;
                            break;
                        case xir::ArithmeticOp::BINARY_GREATER:
                            predicate = ::llvm::CmpInst::FCMP_OGT;
                            break;
                        case xir::ArithmeticOp::BINARY_LESS_EQUAL:
                            predicate = ::llvm::CmpInst::FCMP_OLE;
                            break;
                        case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
                            predicate = ::llvm::CmpInst::FCMP_OGE;
                            break;
                        case xir::ArithmeticOp::BINARY_EQUAL:
                            predicate = ::llvm::CmpInst::FCMP_OEQ;
                            break;
                        case xir::ArithmeticOp::BINARY_NOT_EQUAL:
                            predicate = ::llvm::CmpInst::FCMP_UNE;
                            break;
                        default: break;
                    }
                    return _builder.CreateFCmp(
                        predicate, operands[0u], operands[1u]);
                }
                auto predicate = ::llvm::CmpInst::BAD_ICMP_PREDICATE;
                switch (op) {
                    case xir::ArithmeticOp::BINARY_LESS:
                        predicate = is_signed ? ::llvm::CmpInst::ICMP_SLT :
                                                ::llvm::CmpInst::ICMP_ULT;
                        break;
                    case xir::ArithmeticOp::BINARY_GREATER:
                        predicate = is_signed ? ::llvm::CmpInst::ICMP_SGT :
                                                ::llvm::CmpInst::ICMP_UGT;
                        break;
                    case xir::ArithmeticOp::BINARY_LESS_EQUAL:
                        predicate = is_signed ? ::llvm::CmpInst::ICMP_SLE :
                                                ::llvm::CmpInst::ICMP_ULE;
                        break;
                    case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
                        predicate = is_signed ? ::llvm::CmpInst::ICMP_SGE :
                                                ::llvm::CmpInst::ICMP_UGE;
                        break;
                    case xir::ArithmeticOp::BINARY_EQUAL:
                        predicate = ::llvm::CmpInst::ICMP_EQ;
                        break;
                    case xir::ArithmeticOp::BINARY_NOT_EQUAL:
                        predicate = ::llvm::CmpInst::ICMP_NE;
                        break;
                    default: break;
                }
                return _builder.CreateICmp(
                    predicate, operands[0u], operands[1u]);
            }
            case xir::ArithmeticOp::SELECT:
                if (!require(3u)) { return nullptr; }
                return _builder.CreateSelect(
                    operands[2u], operands[0u], operands[1u]);
            default:
                _fail("Phase-2 LLVM packet codegen does not implement this arithmetic operation yet");
                return nullptr;
        }
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
        operands.reserve(instruction.operands.size());
        for (auto operand_id : instruction.operands) {
            auto *operand = _source.value(operand_id);
            auto *llvm_operand = _as_lane_vector(
                _load_value(operand_id), *operand);
            if (llvm_operand == nullptr) { return nullptr; }
            operands.emplace_back(llvm_operand);
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
            return _builder.CreateExtractElement(lanes, safe);
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
                if (!require(1u)) { return nullptr; }
                return _collectives.active_all_equal(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_bit_and(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_bit_or(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_bit_xor(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_count_bits(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_max(
                    _builder, operands[0u], participants,
                    _source.value(instruction.operands[0u])->type->is_int());
            case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_min(
                    _builder, operands[0u], participants,
                    _source.value(instruction.operands[0u])->type->is_int());
            case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_product(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
                if (!require(1u)) { return nullptr; }
                return _collectives.active_sum(
                    _builder, operands[0u], participants);
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
                if (!require(1u)) { return nullptr; }
                return _collectives.prefix_sum(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT:
                if (!require(1u)) { return nullptr; }
                return _collectives.prefix_product(
                    _builder, operands[0u], participants);
            case xir::ThreadGroupOp::WARP_READ_LANE:
                if (!require(2u)) { return nullptr; }
                return cohort_scalar(_collectives.read_lane(
                                         _builder, operands[0u],
                                         operands[1u], participants)
                                         .values);
            case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE:
                if (!require(1u)) { return nullptr; }
                return cohort_scalar(
                    _collectives.read_first_active_lane(
                        _builder, operands[0u], participants)
                        .values);
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
            case schedule::Opcode::warp_collective:
                value = _collective(instruction);
                break;
            default:
                _fail("unsupported Schedule IR instruction reached LLVM emission");
                return;
        }
        if (instruction.result && value != nullptr) {
            _locals.insert_or_assign(instruction.result->value, value);
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
        _builder.CreateStore(
            _builder.CreateSelect(mask, lanes, old), slot);
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
        _state_slots.resize(_source.values().size(), nullptr);
        for (auto &&value : _source.values()) {
            if (value.origin != schedule::ValueOrigin::state_slot) {
                continue;
            }
            auto *type = _layout.state_type(value);
            if (type == nullptr) {
                _fail(_layout.error());
                return;
            }
            auto *slot = _builder.CreateAlloca(
                type, nullptr, value.name + ".slot");
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
