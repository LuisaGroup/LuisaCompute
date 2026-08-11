#include "llvm_schedule_codegen.h"

#include <algorithm>
#include <cstring>
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
    ::llvm::AllocaInst *_ready_bits{nullptr};
    ::llvm::AllocaInst *_live_mask{nullptr};
    std::vector<::llvm::AllocaInst *> _pending_masks{};
    std::vector<::llvm::AllocaInst *> _convergence_expected{};
    std::vector<::llvm::AllocaInst *> _convergence_arrived{};
    std::vector<::llvm::AllocaInst *> _state_slots{};
    std::vector<::llvm::Value *> _external_values{};
    std::vector<size_t> _parameter_offsets{};
    std::unordered_map<uint32_t, ::llvm::Value *> _locals{};
    ::llvm::Value *_active_mask{nullptr};

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
        if (edge.loop_back) {
            _fail("Phase-2 LLVM packet codegen does not support loop epochs yet");
            return;
        }
        if (edge.joins.size() > 1u) {
            _fail("Phase-2 LLVM packet codegen does not support cascading nested convergence yet");
            return;
        }
        if (split_edge && !edge.joins.empty()) {
            _fail("a divergent split edge must not close a convergence gate directly");
            return;
        }
        if (!edge.joins.empty()) {
            auto *point = _source.convergence(edge.joins.front());
            if (point == nullptr || point->target != edge.target) {
                _fail("convergence arrival edge does not target its gate block");
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
        if (_source.blocks().size() > 64u) {
            _fail("Phase-2 LLVM packet dispatcher supports at most 64 blocks");
            return;
        }
        if (!_source.loops().empty()) {
            _fail("Phase-2 LLVM packet codegen supports acyclic Schedule IR only");
            return;
        }
        for (auto &&point : _source.convergence_points()) {
            if (point.parent) {
                _fail("Phase-2 LLVM packet codegen does not support nested convergence yet");
                return;
            }
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

    void _enqueue(schedule::BlockId target, ::llvm::Value *mask) {
        auto *slot = _pending_masks[target.value];
        auto *pending = _builder.CreateLoad(slot->getAllocatedType(), slot);
        _builder.CreateStore(_builder.CreateOr(pending, mask), slot);
        auto *any = _builder.CreateOrReduce(mask);
        auto bit = uint64_t{1u} << target.value;
        auto *ready = _builder.CreateLoad(
            _ready_bits->getAllocatedType(), _ready_bits);
        auto *bit_value = _builder.CreateSelect(
            any, _builder.getInt64(bit), _builder.getInt64(0u));
        _builder.CreateStore(_builder.CreateOr(ready, bit_value), _ready_bits);
    }

    void _emit_arrival(const schedule::ControlEdge &edge,
                       ::llvm::Value *mask) {
        _apply_assignments(edge.assignments, mask);
        if (_failed()) { return; }
        if (edge.joins.empty()) {
            _enqueue(edge.target, mask);
            _builder.CreateBr(_scheduler_loop);
            return;
        }
        auto convergence = edge.joins.front();
        auto *arrived_slot = _convergence_arrived[convergence.value];
        ::llvm::Value *arrived = _builder.CreateLoad(
            arrived_slot->getAllocatedType(), arrived_slot);
        arrived = _builder.CreateOr(arrived, mask);
        _builder.CreateStore(arrived, arrived_slot);
        auto *expected_slot = _convergence_expected[convergence.value];
        auto *expected = _builder.CreateLoad(
            expected_slot->getAllocatedType(), expected_slot);
        auto *complete = _builder.CreateAndReduce(
            _builder.CreateICmpEQ(arrived, expected));
        auto *release = ::llvm::BasicBlock::Create(
            _module.getContext(), "convergence.release", _entry);
        auto *wait = ::llvm::BasicBlock::Create(
            _module.getContext(), "convergence.wait", _entry);
        _builder.CreateCondBr(complete, release, wait);

        _builder.SetInsertPoint(release);
        auto *zero = ::llvm::Constant::getNullValue(
            arrived_slot->getAllocatedType());
        _builder.CreateStore(zero, arrived_slot);
        _builder.CreateStore(zero, expected_slot);
        _enqueue(edge.target, arrived);
        _builder.CreateBr(_scheduler_loop);

        _builder.SetInsertPoint(wait);
        _builder.CreateBr(_scheduler_loop);
    }

    void _declare_convergence(schedule::ConvergenceId convergence) {
        auto *expected = _convergence_expected[convergence.value];
        auto *arrived = _convergence_arrived[convergence.value];
        _builder.CreateStore(_active_mask, expected);
        _builder.CreateStore(
            ::llvm::Constant::getNullValue(arrived->getAllocatedType()),
            arrived);
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
                    if (control.convergence) {
                        _declare_convergence(*control.convergence);
                    }
                    if (condition_value->value_class ==
                        schedule::ValueClass::varying) {
                        auto *true_mask = _builder.CreateAnd(
                            _active_mask, condition);
                        auto *false_mask = _builder.CreateAnd(
                            _active_mask, _builder.CreateNot(condition));
                        _apply_assignments(
                            control.true_edge.assignments, true_mask);
                        _apply_assignments(
                            control.false_edge.assignments, false_mask);
                        if (_failed()) { return; }
                        _enqueue(control.true_edge.target, true_mask);
                        _enqueue(control.false_edge.target, false_mask);
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
                    auto *zero_mask = ::llvm::Constant::getNullValue(
                        _layout.mask_type());
                    // Terminated lanes no longer count toward any open gate.
                    // Release a gate immediately if its remaining expected
                    // live mask has now fully arrived.
                    for (auto &&point : _source.convergence_points()) {
                        auto *expected_slot =
                            _convergence_expected[point.id.value];
                        auto *arrived_slot =
                            _convergence_arrived[point.id.value];
                        auto *expected = _builder.CreateAnd(
                            _builder.CreateLoad(
                                expected_slot->getAllocatedType(),
                                expected_slot),
                            new_live);
                        auto *arrived = _builder.CreateLoad(
                            arrived_slot->getAllocatedType(), arrived_slot);
                        auto *complete = _builder.CreateAndReduce(
                            _builder.CreateICmpEQ(arrived, expected));
                        auto *released = _builder.CreateSelect(
                            complete, arrived, zero_mask);
                        _builder.CreateStore(
                            _builder.CreateSelect(
                                complete, zero_mask, expected),
                            expected_slot);
                        _builder.CreateStore(
                            _builder.CreateSelect(
                                complete, zero_mask, arrived),
                            arrived_slot);
                        _enqueue(point.target, released);
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
        _ready_bits = _builder.CreateAlloca(
            _builder.getInt64Ty(), nullptr, "ready.bits");
        _live_mask = _builder.CreateAlloca(
            mask_type, nullptr, "live.mask");
        _pending_masks.resize(_source.blocks().size());
        for (auto &&block : _source.blocks()) {
            auto *slot = _builder.CreateAlloca(
                mask_type, nullptr,
                "pending." + std::to_string(block.id.value));
            _builder.CreateStore(zero_mask, slot);
            _pending_masks[block.id.value] = slot;
        }
        _convergence_expected.resize(
            _source.convergence_points().size());
        _convergence_arrived.resize(
            _source.convergence_points().size());
        for (auto &&point : _source.convergence_points()) {
            auto *expected = _builder.CreateAlloca(
                mask_type, nullptr,
                "convergence.expected." + std::to_string(point.id.value));
            auto *arrived = _builder.CreateAlloca(
                mask_type, nullptr,
                "convergence.arrived." + std::to_string(point.id.value));
            _builder.CreateStore(zero_mask, expected);
            _builder.CreateStore(zero_mask, arrived);
            _convergence_expected[point.id.value] = expected;
            _convergence_arrived[point.id.value] = arrived;
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
        _builder.CreateStore(
            initial_mask, _pending_masks[_source.entry().value]);
        auto entry_bit = uint64_t{1u} << _source.entry().value;
        _builder.CreateStore(
            _builder.CreateSelect(
                _builder.CreateOrReduce(initial_mask),
                _builder.getInt64(entry_bit), _builder.getInt64(0u)),
            _ready_bits);

        _scheduler_loop = ::llvm::BasicBlock::Create(
            context, "scheduler.loop", _entry);
        auto *dispatch = ::llvm::BasicBlock::Create(
            context, "scheduler.dispatch", _entry);
        auto *done = ::llvm::BasicBlock::Create(
            context, "scheduler.done", _entry);
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
        auto *ready = _builder.CreateLoad(
            _ready_bits->getAllocatedType(), _ready_bits);
        _builder.CreateCondBr(
            _builder.CreateICmpNE(ready, _builder.getInt64(0u)),
            dispatch, done);

        _builder.SetInsertPoint(dispatch);
        auto *cttz =
#if LLVM_VERSION_MAJOR >= 22
            ::llvm::Intrinsic::getOrInsertDeclaration(
#else
            ::llvm::Intrinsic::getDeclaration(
#endif
            &_module, ::llvm::Intrinsic::cttz,
            {_builder.getInt64Ty()});
        auto *pc = _builder.CreateCall(
            cttz, {ready, _builder.getFalse()});
        auto *dispatch_switch = _builder.CreateSwitch(
            _builder.CreateTrunc(pc, _builder.getInt32Ty()),
            invalid, static_cast<unsigned>(cases.size()));
        for (auto &&block : _source.blocks()) {
            dispatch_switch->addCase(
                _builder.getInt32(block.id.value), cases[block.id.value]);
        }

        _builder.SetInsertPoint(invalid);
        _builder.CreateUnreachable();

        _builder.SetInsertPoint(done);
        _builder.CreateRetVoid();

        auto *zero_mask = ::llvm::Constant::getNullValue(
            _layout.mask_type());
        for (auto &&block : _source.blocks()) {
            _builder.SetInsertPoint(cases[block.id.value]);
            _locals.clear();
            auto *pending = _pending_masks[block.id.value];
            _active_mask = _builder.CreateLoad(
                pending->getAllocatedType(), pending, "active.mask");
            _builder.CreateStore(zero_mask, pending);
            auto *current_ready = _builder.CreateLoad(
                _ready_bits->getAllocatedType(), _ready_bits);
            auto bit = uint64_t{1u} << block.id.value;
            _builder.CreateStore(
                _builder.CreateAnd(
                    current_ready, _builder.getInt64(~bit)),
                _ready_bits);
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
