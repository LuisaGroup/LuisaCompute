//
// Created by mike on 11/1/25.
//

#include <span>
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept {
    std::string printf_format = "[cycle %llu] [cuda] ";
    llvm::SmallVector<llvm::Value *, 8> llvm_args;
    // use clock to present the time
    auto llvm_clock = b.CreateIntrinsic(llvm::Intrinsic::nvvm_read_ptx_sreg_clock64, {});
    llvm_args.emplace_back(llvm_clock);
    // decode real print arguments
    auto op_uses = inst->operand_uses();
    auto decode_value = [&](auto &&self, const Type *type, llvm::Value *llvm_value) -> void {
        switch (type->tag()) {
            case Type::Tag::BOOL: {// %s, select(llvm_value, "true", "false")
                printf_format.append("%s");
                auto llvm_true_str = b.CreateGlobalString("true", "luisa.string.true");
                auto llvm_false_str = b.CreateGlobalString("false", "luisa.string.false");
                llvm_args.emplace_back(b.CreateSelect(llvm_value, llvm_true_str, llvm_false_str));
                break;
            }
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: {// signed integers that are at most 32 bits: %d
                printf_format.append("%d");
                llvm_args.emplace_back(b.CreateZExt(llvm_value, b.getInt32Ty()));
                break;
            }
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: {// unsigned integers that are at most 32 bits: %u
                printf_format.append("%u");
                llvm_args.emplace_back(b.CreateZExt(llvm_value, b.getInt32Ty()));
                break;
            }
            case Type::Tag::INT64: {// %lld
                printf_format.append("%lld");
                llvm_args.emplace_back(llvm_value);
                break;
            }
            case Type::Tag::UINT64: {// %llu
                printf_format.append("%llu");
                llvm_args.emplace_back(llvm_value);
                break;
            }
            case Type::Tag::FLOAT16: [[fallthrough]];
            case Type::Tag::FLOAT32: [[fallthrough]];
            case Type::Tag::FLOAT64: {// floating points: %f
                printf_format.append("%f");
                llvm_args.emplace_back(_safe_fp_cast(b, llvm_value, b.getDoubleTy()));
                break;
            }
            case Type::Tag::VECTOR: {
                auto dim = type->dimension();
                auto elem_type = type->element();
                printf_format.push_back('(');
                for (auto i = 0u; i < dim; i++) {
                    auto llvm_elem = b.CreateExtractElement(llvm_value, i);
                    self(self, elem_type, llvm_elem);
                    if (i + 1u != dim) {
                        printf_format.append(", ");
                    }
                }
                printf_format.push_back(')');
                break;
            }
            case Type::Tag::MATRIX: {
                auto dim = type->dimension();
                auto col_type = Type::vector(type->element(), dim);
                printf_format.push_back('<');
                for (auto i = 0u; i < dim; i++) {
                    auto llvm_col = b.CreateExtractValue(llvm_value, i);
                    self(self, col_type, llvm_col);
                    if (i + 1u != dim) {
                        printf_format.append(", ");
                    }
                }
                printf_format.push_back('>');
                break;
            }
            case Type::Tag::ARRAY: {
                auto dim = type->dimension();
                auto elem_type = type->element();
                printf_format.push_back('[');
                for (auto i = 0u; i < dim; i++) {
                    auto llvm_elem = b.CreateExtractValue(llvm_value, i);
                    self(self, elem_type, llvm_elem);
                    if (i + 1u != dim) {
                        printf_format.append(", ");
                    }
                }
                printf_format.push_back(']');
                break;
            }
            case Type::Tag::STRUCTURE: {
                auto member_types = type->members();
                printf_format.push_back('{');
                for (auto [i, member_type] : llvm::enumerate(std::span{member_types.data(), member_types.size()})) {
                    auto llvm_member = b.CreateExtractValue(llvm_value, i);
                    self(self, member_type, llvm_member);
                    if (i + 1u != member_types.size()) {
                        printf_format.append(", ");
                    }
                }
                printf_format.push_back('}');
                break;
            }
            default: LUISA_ERROR("Unsupported type in PrintInst: {}", type->description());
        }
    };
    auto decode_one_operand = [&] {
        LUISA_ASSERT(!op_uses.empty(), "Not enough operands for PrintInst.");
        auto op = op_uses.front()->value();
        op_uses = op_uses.subspan(1);
        // we need to recursively convert the operand to primitive types
        auto llvm_op = _get_llvm_value(b, func_ctx, op);
        decode_value(decode_value, op->type(), llvm_op);
    };
    auto fmt_format = std::string_view{inst->format()};
    // convert from C++20 std::format to C-style printf format
    while (!fmt_format.empty()) {
        if (fmt_format.front() == '%') {// we need to escape % for printf
            printf_format.append("%%");
            fmt_format.remove_prefix(1);
        } else if (fmt_format.front() == '{') {// might be the start of a format specifier
            fmt_format.remove_prefix(1);
            LUISA_ASSERT(!fmt_format.empty(), "Invalid format string in PrintInst: missing '}'");
            if (fmt_format.front() == '{') {// escaped '{'
                printf_format.push_back('{');
                fmt_format.remove_prefix(1);
            } else {// format specifier, currently we simply ignore all format options but just search for the closing '}'
                while (!fmt_format.empty() && fmt_format.front() != '}') {
                    fmt_format.remove_prefix(1);
                }
                LUISA_ASSERT(!fmt_format.empty(), "Invalid format string in PrintInst: missing '}'");
                LUISA_DEBUG_ASSERT(fmt_format.front() == '}');
                fmt_format.remove_prefix(1);
                // decode one operand
                decode_one_operand();
            }
        } else {// just some normal character
            printf_format.push_back(fmt_format.front());
            fmt_format.remove_prefix(1);
        }
    }
    if (!op_uses.empty()) {
        LUISA_WARNING_WITH_LOCATION("Too many operands for PrintInst: {} unused.", op_uses.size());
    }
    printf_format.push_back('\n');// append newline
    // get the vprintf struct and insert elements
    llvm::SmallVector<llvm::Type *, 8> printf_arg_types;
    printf_arg_types.reserve(llvm_args.size());
    for (auto arg : llvm_args) { printf_arg_types.emplace_back(arg->getType()); }
    auto llvm_vprintf_struct_type = llvm::StructType::get(_llvm_context, printf_arg_types, false);
    auto llvm_vprintf_struct = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_vprintf_struct_type));
    for (auto i = 0u; i < llvm_args.size(); i++) {
        llvm_vprintf_struct = b.CreateInsertValue(llvm_vprintf_struct, llvm_args[i], i);
    }
    // create a temporary alloca for vprintf struct
    auto llvm_temp = _create_temp_in_alloca_block(func_ctx, llvm_vprintf_struct_type);
    b.CreateStore(llvm_vprintf_struct, llvm_temp);
    // get vprintf function
    auto llvm_vprintf = _get_vprintf_function();
    // create global string for format
    auto llvm_format = b.CreateGlobalString(printf_format, "luisa.string.print.format");
    // all set, call vprintf
    auto llvm_call = b.CreateCall(llvm_vprintf, {llvm_format, llvm_temp});
    llvm_call->addFnAttr(llvm::Attribute::NoUnwind);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept {
    auto llvm_i64_type = b.getInt64Ty();
    auto llvm_clock = b.CreateIntrinsic(llvm_i64_type, llvm::Intrinsic::nvvm_read_ptx_sreg_clock, {});
    auto llvm_clock_type = _get_llvm_type(inst->type())->reg_type;
    return b.CreateZExtOrTrunc(llvm_clock, llvm_clock_type);
}

void CUDACodegenLLVMImpl::_translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept {
    b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::debugtrap, {});
}

void CUDACodegenLLVMImpl::_translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept {
    auto llvm_cond = _get_llvm_value(b, func_ctx, inst->condition());
    _create_assertion_with_message(b, llvm_cond, fmt::format("Assertion failed: {}\n", inst->message()));
}

void CUDACodegenLLVMImpl::_translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept {
    // auto cond = _get_llvm_value(b, func_ctx, inst->condition());
    // b.CreateAssumption(cond);
}

void CUDACodegenLLVMImpl::_create_assertion_with_message(IB &b, llvm::Value *cond, luisa::string_view message) noexcept {
    if (_config.enable_debug_info) {// we only create assertions when debug info is enabled
        auto llvm_msg = llvm::ConstantDataArray::getString(_llvm_context, message);
        // ReSharper disable once CppDFAMemoryLeak
        auto llvm_msg_gv = new llvm::GlobalVariable(
            *_llvm_module, llvm_msg->getType(), true,
            llvm::GlobalValue::PrivateLinkage, llvm_msg, "luisa.assert.message",
            nullptr, llvm::GlobalValue::NotThreadLocal, nvptx_address_space_constant);
        auto llvm_assert_f = _get_assert_function();
        b.CreateCall(llvm_assert_f, {cond, llvm_msg_gv});
    }
}

}// namespace luisa::compute::cuda
