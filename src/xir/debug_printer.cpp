#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/ast/type.h>
#include <luisa/xir/value.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/debug_printer.h>

namespace luisa::compute::xir {

using namespace std::string_view_literals;

struct XIRDebugPrinter::Impl {

    luisa::unordered_map<const Value *, size_t> value_indices;

    void reset() noexcept {
        value_indices.clear();
    }

    [[nodiscard]] auto index_of(const Value *value) noexcept {
        auto it = value_indices.try_emplace(value, value_indices.size());
        return it.first->second;
    }
};

XIRDebugPrinter::XIRDebugPrinter() noexcept
    : _impl{luisa::make_unique<Impl>()} {}

XIRDebugPrinter::~XIRDebugPrinter() noexcept = default;

void XIRDebugPrinter::reset() noexcept {
    _impl->reset();
}

void XIRDebugPrinter::emit_type(luisa::string &s, const Type *type) noexcept {
    if (type == nullptr) {
        s.append("void"sv);
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: s.append("bool"sv); break;
        case Type::Tag::INT8: s.append("i8"sv); break;
        case Type::Tag::UINT8: s.append("u8"sv); break;
        case Type::Tag::INT16: s.append("i16"sv); break;
        case Type::Tag::UINT16: s.append("u16"sv); break;
        case Type::Tag::INT32: s.append("i32"sv); break;
        case Type::Tag::UINT32: s.append("u32"sv); break;
        case Type::Tag::INT64: s.append("i64"sv); break;
        case Type::Tag::UINT64: s.append("u64"sv); break;
        case Type::Tag::FLOAT16: s.append("f16"sv); break;
        case Type::Tag::FLOAT32: s.append("f32"sv); break;
        case Type::Tag::FLOAT64: s.append("f64"sv); break;
        case Type::Tag::VECTOR: {
            s.append("<"sv);
            emit_type(s, type->element());
            luisa::format_to(std::back_inserter(s), " x {}>", type->dimension());
            break;
        }
        case Type::Tag::MATRIX: {
            s.append("("sv);
            emit_type(s, type->element());
            auto n = type->dimension();
            luisa::format_to(std::back_inserter(s), " x {} x {})", n, n);
            break;
        }
        case Type::Tag::ARRAY: {
            s.append("["sv);
            emit_type(s, type->element());
            luisa::format_to(std::back_inserter(s), " x {}]", type->dimension());
            break;
        }
        case Type::Tag::STRUCTURE: {
            luisa::format_to(std::back_inserter(s), "{{align {}", type->alignment());
            for (auto m : type->members()) {
                s.append(", "sv);
                emit_type(s, m);
            }
            s.append("}"sv);
            break;
        }
        case Type::Tag::BUFFER: {
            s.append("Buffer<"sv);
            emit_type(s, type->element());
            s.append(">"sv);
            break;
        }
        case Type::Tag::TEXTURE: {
            luisa::format_to(std::back_inserter(s), "Texture{}D<", type->element()->dimension());
            emit_type(s, type->element());
            s.append(">"sv);
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: s.append("Bindless"sv); break;
        case Type::Tag::ACCEL: s.append("Accel"sv); break;
        case Type::Tag::CUSTOM: luisa::format_to(std::back_inserter(s), "T{:?}", type->description()); break;
        default: LUISA_ERROR_WITH_LOCATION("Unknown type");
    }
}

namespace {

void print_literal(luisa::string &s, const Type *type, const void *data) noexcept {
    LUISA_DEBUG_ASSERT(type != nullptr, "Type should not be null.");
    auto print_scalar = [&]<typename T>(T x) noexcept {
        std::memcpy(&x, data, sizeof(T));
        if constexpr (std::is_same_v<T, bool>) {
            s.append(x ? "true"sv : "false"sv);
        } else if constexpr (luisa::is_floating_point_v<T>) {
            luisa::format_to(std::back_inserter(s), "{}", static_cast<double>(x));
        } else if constexpr (luisa::is_signed_integral_v<T>) {
            luisa::format_to(std::back_inserter(s), "{}", static_cast<int64_t>(x));
        } else if constexpr (luisa::is_unsigned_integral_v<T>) {
            luisa::format_to(std::back_inserter(s), "{}", static_cast<uint64_t>(x));
        }
    };
    switch (type->tag()) {
        case Type::Tag::BOOL: print_scalar(bool{}); break;
        case Type::Tag::INT8: print_scalar(int8_t{}); break;
        case Type::Tag::UINT8: print_scalar(uint8_t{}); break;
        case Type::Tag::INT16: print_scalar(int16_t{}); break;
        case Type::Tag::UINT16: print_scalar(uint16_t{}); break;
        case Type::Tag::INT32: print_scalar(int32_t{}); break;
        case Type::Tag::UINT32: print_scalar(uint32_t{}); break;
        case Type::Tag::INT64: print_scalar(int64_t{}); break;
        case Type::Tag::UINT64: print_scalar(uint64_t{}); break;
        case Type::Tag::FLOAT16: print_scalar(half{}); break;
        case Type::Tag::FLOAT32: print_scalar(float{}); break;
        case Type::Tag::FLOAT64: print_scalar(double{}); break;
        case Type::Tag::VECTOR: {
            auto elem_stride = type->element()->size();
            auto elem_count = type->dimension();
            s.append("<"sv);
            auto p = static_cast<const std::byte *>(data);
            for (auto i = 0u; i < elem_count; i++) {
                if (i != 0u) { s.append(", "sv); }
                print_literal(s, type->element(), p);
                p += elem_stride;
            }
            s.append(">"sv);
            break;
        }
        case Type::Tag::MATRIX: {
            auto dim = type->dimension();
            auto col_type = Type::vector(type->element(), dim);
            auto col_stride = col_type->size();
            auto p = static_cast<const std::byte *>(data);
            s.append("("sv);
            for (auto i = 0u; i < dim; i++) {
                if (i != 0u) { s.append(", "sv); }
                print_literal(s, col_type, p);
                p += col_stride;
            }
            s.append(")"sv);
            break;
        }
        case Type::Tag::ARRAY: {
            auto elem_stride = type->element()->size();
            auto elem_count = type->dimension();
            s.append("["sv);
            auto p = static_cast<const std::byte *>(data);
            for (auto i = 0u; i < elem_count; i++) {
                if (i != 0u) { s.append(", "sv); }
                print_literal(s, type->element(), p);
                p += elem_stride;
            }
            s.append("]"sv);
            break;
        }
        case Type::Tag::STRUCTURE: {
            auto members = type->members();
            auto p = static_cast<const std::byte *>(data);
            s.append("{");
            for (auto i = 0u; i < members.size(); i++) {
                if (i != 0u) { s.append(", "sv); }
                print_literal(s, members[i], p);
                p += members[i]->size();
            }
            s.append("}"sv);
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION(
            "Invalid constant type: {}",
            type == nullptr ? "void" : type->description());
    }
}

}// namespace

void XIRDebugPrinter::emit_value_name(luisa::string &s, const Value *value) noexcept {
    if (value == nullptr) {
        s.append("null"sv);
        return;
    }
    if (value->isa<SpecialRegister>()) {
        auto sreg = static_cast<const SpecialRegister *>(value);
        s.append(xir::to_string(sreg->derived_special_register_tag()));
        return;
    }
    if (value->isa<Undefined>()) {
        s.append(xir::to_string(Undefined::static_derived_value_tag()));
        return;
    }
    if (value->isa<Constant>() && value->type()->is_basic()) {
        return print_literal(s, value->type(), static_cast<const Constant *>(value)->data());
    }
    auto uid = _impl->index_of(value);
    if (value->is_lvalue()) { s.append("*"sv); }
    luisa::format_to(std::back_inserter(s), "%{}", uid);
}

void XIRDebugPrinter::emit_value_debug_info(luisa::string &s, const Value *value) noexcept {
    if (value != nullptr) {
        auto any_info = false;
        constexpr auto prefix = " // "sv;
        s.append(prefix);
        if (value->isa<Instruction>()) {
            auto inst = static_cast<const Instruction *>(value);
            if (auto merge = inst->control_flow_merge()) {
                any_info = true;
                s.append("merge = "sv);
                emit_value_name(s, merge->merge_block());
                s.append(", "sv);
            }
        }
        if (!value->use_list().empty()) {
            any_info = true;
            s.append("users = {"sv);
            for (auto use : value->use_list()) {
                emit_value_name(s, use->user());
                s.append(", "sv);
            }
            s.pop_back();
            s.pop_back();
            s.append("}, "sv);
        }
        if (value->isa<BasicBlock>()) {
            auto bb = static_cast<const BasicBlock *>(value);
            auto any_pred = false;
            bb->traverse_predecessors(false, [&](const BasicBlock *pred) noexcept {
                if (any_pred == false) {
                    any_pred = true;
                    s.append("preds = {"sv);
                }
                emit_value_name(s, pred);
                s.append(", "sv);
            });
            if (any_pred) {
                any_info = true;
                s.pop_back();
                s.pop_back();
                s.append("}, "sv);
            }
        }
        if (any_info) {
            s.pop_back();
            s.pop_back();
        } else {
            for (auto _ : prefix) {
                s.pop_back();
            }
        }
    }
}

void XIRDebugPrinter::emit_operand(luisa::string &s, const Value *value) noexcept {
    if (value == nullptr) {
        return emit_value_name(s, value);
    }
    if (!value->isa<BasicBlock>() && !value->isa<Function>()) {
        emit_type(s, value->type());
        s.append(" "sv);
    }
    if (value->isa<SpecialRegister>()) {
        return emit_value_name(s, value);
    }
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(value->derived_value_tag()));
    emit_value_name(s, value);
}

void XIRDebugPrinter::emit_instruction(luisa::string &s, const Instruction *instruction) noexcept {
    LUISA_DEBUG_ASSERT(instruction != nullptr);
    if (auto &metadata = instruction->metadata_list(); !metadata.empty()) {
        emit_metadata_list(s, metadata);
        s.append("\n    "sv);
    }
    emit_value_name(s, instruction);
    s.append(": "sv);
    if (auto t = instruction->type()) {
        emit_type(s, t);
        s.append(" "sv);
    }
    luisa::format_to(std::back_inserter(s), "{}",
                     instruction->intrinsic_identifier());
    for (auto op_use : instruction->operand_uses()) {
        if (op_use == instruction->operand_uses().front()) {
            s.append(" "sv);
        } else {
            s.append(", "sv);
        }
        emit_operand(s, op_use->value());
    }
    emit_value_debug_info(s, instruction);
}

void XIRDebugPrinter::emit_basic_block(luisa::string &s, const BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(block != nullptr);
    if (auto &metadata = block->metadata_list(); !metadata.empty()) {
        s.append("\n  "sv);
        emit_metadata_list(s, metadata);
    }
    s.append("\n  "sv);
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(block->derived_value_tag()));
    emit_value_name(s, block);
    s.append(": {"sv);
    emit_value_debug_info(s, block);
    s.append("\n"sv);
    for (auto inst : block->instructions()) {
        s.append("    "sv);
        emit_instruction(s, inst);
        s.append("\n"sv);
    }
    luisa::format_to(std::back_inserter(s), "  }} // end of {} ",
                     xir::to_string(block->derived_value_tag()));
    emit_value_name(s, block);
}

void XIRDebugPrinter::emit_constant(luisa::string &s, const Constant *value) noexcept {
    LUISA_DEBUG_ASSERT(value != nullptr);
    if (auto &metadata = value->metadata_list(); !metadata.empty()) {
        emit_metadata_list(s, metadata);
        s.append("\n"sv);
    }
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(Constant::static_derived_value_tag()));
    emit_value_name(s, value);
    s.append(": "sv);
    emit_type(s, value->type());
    s.append(" "sv);
    print_literal(s, value->type(), value->data());
    emit_value_debug_info(s, value);
}

void XIRDebugPrinter::emit_function_decl(luisa::string &s, const Function *function) noexcept {
    LUISA_DEBUG_ASSERT(function != nullptr);
    if (auto &metadata = function->metadata_list(); !metadata.empty()) {
        emit_metadata_list(s, metadata);
        s.append("\n"sv);
    }
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(Function::static_derived_value_tag()));
    emit_value_name(s, function);
    s.append(": "sv);
    emit_type(s, function->type());
    luisa::format_to(std::back_inserter(s), " {}"sv,
                     xir::to_string(function->derived_function_tag()));
    emit_value_debug_info(s, function);
    s.append("\n"sv);
    s.append("("sv);
    for (auto arg : function->arguments()) {
        s.append("\n  "sv);
        emit_value_name(s, arg);
        s.append(": "sv);
        emit_type(s, arg->type());
        emit_value_debug_info(s, arg);
    }
    if (!function->arguments().empty()) {
        s.append("\n"sv);
    }
    s.append(")"sv);
}

void XIRDebugPrinter::emit_function(luisa::string &s, const Function *function) noexcept {
    emit_function_decl(s, function);
    if (auto def = function->definition()) {
        s.append(" {"sv);
        def->traverse_basic_blocks(
            BasicBlockTraversalOrder::REVERSE_POST_ORDER,
            [this, &s](const BasicBlock *block) {
                this->emit_basic_block(s, block);
                s.append("\n"sv);
            });
        s.append("}"sv);
    }
}

void XIRDebugPrinter::emit_module(luisa::string &s, const Module *module) noexcept {
    if (auto &metadata = module->metadata_list(); !metadata.empty()) {
        emit_metadata_list(s, metadata);
        s.append("\n"sv);
    }
    s.append("module"sv);
    if (auto name = module->name()) {
        luisa::format_to(std::back_inserter(s), "({})", name.value());
    }
    s.append("\n");
    auto any_const = false;
    for (auto c : module->constant_list()) {
        if (!c->type()->is_basic()) {
            any_const = true;
            s.append("\n"sv);
            emit_constant(s, c);
        }
    }
    if (any_const) {
        s.append("\n"sv);
    }
    s.append("\n"sv);
    for (auto f : module->function_list()) {
        emit_function(s, f);
        s.append("\n\n"sv);
    }
}

void XIRDebugPrinter::emit_metadata_list(luisa::string &s, const MetadataList &metadata) noexcept {
    if (!metadata.empty()) {
        s.append("[["sv);
        for (auto md : metadata) {
            switch (md->derived_metadata_tag()) {
                case DerivedMetadataTag::NAME: {
                    auto name_md = static_cast<const NameMD *>(md);
                    luisa::format_to(std::back_inserter(s), "name = {:?}, ",
                                     name_md->name());
                    break;
                }
                case DerivedMetadataTag::LOCATION: {
                    auto loc_md = static_cast<const LocationMD *>(md);
                    luisa::format_to(std::back_inserter(s), "location = ({:?}, {}), ",
                                     loc_md->file().string(), loc_md->line());
                    break;
                }
                case DerivedMetadataTag::COMMENT: {
                    auto comment_md = static_cast<const CommentMD *>(md);
                    luisa::format_to(std::back_inserter(s), "comment = {:?}, ",
                                     comment_md->comment());
                    break;
                }
                case DerivedMetadataTag::SIGNATURE_CONSTRAINT: {
                    s.append("signature_constraint, "sv);
                    break;
                }
                case DerivedMetadataTag::CURVE_BASIS: {
                    auto curve_md = static_cast<const CurveBasisMD *>(md);
                    s.append("curve_basis = ("sv);
                    auto any_basis = false;
                    if (curve_md->curve_basis_set().test(CurveBasis::PIECEWISE_LINEAR)) {
                        any_basis = true;
                        s.append("piecewise_linear, "sv);
                    }
                    if (curve_md->curve_basis_set().test(CurveBasis::CUBIC_BSPLINE)) {
                        any_basis = true;
                        s.append("cubic_bspline, "sv);
                    }
                    if (curve_md->curve_basis_set().test(CurveBasis::CATMULL_ROM)) {
                        any_basis = true;
                        s.append("catmull_rom, "sv);
                    }
                    if (curve_md->curve_basis_set().test(CurveBasis::BEZIER)) {
                        any_basis = true;
                        s.append("bezier, "sv);
                    }
                    if (any_basis) {
                        s.pop_back();
                        s.pop_back();
                    }
                    s.append("), "sv);
                    break;
                }
            }
        }
        s.pop_back();
        s.pop_back();
        s.append("]]"sv);
    }
}

XIRDebugPrinter &XIRDebugPrinter::global() noexcept {
    static XIRDebugPrinter printer;
    return printer;
}

}// namespace luisa::compute::xir
