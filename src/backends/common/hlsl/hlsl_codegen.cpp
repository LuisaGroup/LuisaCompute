#include "hlsl_codegen.h"
#include "struct_generator.h"
#include "codegen_stack_data.h"
namespace lc::hlsl {

namespace {

// Software BF16 (builtin/bfloat16_fallback.bytes): `bfloat16_t` is a raw 16-bit
// pattern, so HLSL operators and intrinsics do not apply to it.  Every operator
// is emitted as a `_lc_bf16_*` helper call instead, and these tables name that
// helper.  An empty name means the plain HLSL spelling is already correct for
// the emulated type (the bitwise operators, for example, work directly on the
// pattern).
[[nodiscard]] vstd::string_view bf16_binary_helper_name(BinaryOp op) noexcept {
    switch (op) {
        case BinaryOp::ADD: return "add"sv;
        case BinaryOp::SUB: return "sub"sv;
        case BinaryOp::MUL: return "mul"sv;
        case BinaryOp::DIV: return "div"sv;
        case BinaryOp::MOD: return "mod"sv;
        case BinaryOp::LESS: return "lt"sv;
        case BinaryOp::GREATER: return "gt"sv;
        case BinaryOp::LESS_EQUAL: return "le"sv;
        case BinaryOp::GREATER_EQUAL: return "ge"sv;
        case BinaryOp::EQUAL: return "eq"sv;
        case BinaryOp::NOT_EQUAL: return "ne"sv;
        default: return {};
    }
}

// Whether an expression evaluates a call.  A packed struct member is written
// through a read-modify-write helper, which renders the member's owner twice; that
// is only sound for an owner without side effects.
[[nodiscard]] bool expression_has_call(Expression const *expr) noexcept {
    if (expr == nullptr) { return false; }
    switch (expr->tag()) {
        case Expression::Tag::CALL:
            return true;
        case Expression::Tag::UNARY:
            return expression_has_call(static_cast<UnaryExpr const *>(expr)->operand());
        case Expression::Tag::BINARY: {
            auto e = static_cast<BinaryExpr const *>(expr);
            return expression_has_call(e->lhs()) || expression_has_call(e->rhs());
        }
        case Expression::Tag::MEMBER:
            return expression_has_call(static_cast<MemberExpr const *>(expr)->self());
        case Expression::Tag::ACCESS: {
            auto e = static_cast<AccessExpr const *>(expr);
            return expression_has_call(e->range()) || expression_has_call(e->index());
        }
        case Expression::Tag::CAST:
            return expression_has_call(static_cast<CastExpr const *>(expr)->expression());
        default:
            return false;
    }
}

[[nodiscard]] vstd::string_view bf16_unary_helper_name(UnaryOp op) noexcept {
    switch (op) {
        case UnaryOp::PLUS: return "pos"sv;
        case UnaryOp::MINUS: return "neg"sv;
        case UnaryOp::NOT: return "not"sv;
        default: return {};
    }
}

}// namespace

void StringStateVisitor::visit(const UnaryExpr *expr) {
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
    if (util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(expr->operand()->type())) {
        if (auto helper = bf16_unary_helper_name(expr->op());
            !helper.empty()) {
            str << "_lc_bf16_"sv << helper << '(';
            expr->operand()->accept(*this);
            str << ')';
            return;
        }
    }
    str << "(";
    switch (expr->op()) {
        case UnaryOp::PLUS://+x
            str << '+';
            break;
        case UnaryOp::MINUS://-x
            str << '-';
            break;
        case UnaryOp::NOT://!x
            str << '!';
            break;
        case UnaryOp::BIT_NOT://~x
            str << '~';
            break;
    }
    expr->operand()->accept(*this);
    str << ")";
}
void StringStateVisitor::visit(const BinaryExpr *expr) {
    literalBrace = true;
    auto fallBackLiteral = vstd::scope_exit([this]() { literalBrace = false; });
    auto op = expr->op();
    if (util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(expr->lhs()->type())) {
        if (auto helper = bf16_binary_helper_name(op);
            !helper.empty()) {
            str << "_lc_bf16_"sv << helper << '(';
            expr->lhs()->accept(*this);
            str << ',';
            expr->rhs()->accept(*this);
            str << ')';
            return;
        }
    }
    auto IsMulFuncCall = [&]() -> bool {
        if (op == BinaryOp::MUL) [[unlikely]] {
            if ((expr->lhs()->type()->is_matrix() &&
                 (!expr->rhs()->type()->is_scalar())) ||
                (expr->rhs()->type()->is_matrix() &&
                 (!expr->lhs()->type()->is_scalar()))) {
                return true;
            }
        }
        return false;
    };
    if (IsMulFuncCall()) {
        str << "_Mul("sv;
        expr->lhs()->accept(*this);
        str << ',';
        expr->rhs()->accept(*this);//Reverse matrix
        str << ')';

    } else if (op == BinaryOp::MOD &&
               (expr->type()->is_float() ||
                (expr->type()->is_vector() &&
                 expr->type()->element()->is_float()))) {
        auto element = expr->type()->is_vector() ?
                           expr->type()->element() :
                           expr->type();
        str << (element->is_float32() ?
                    "_lc_fmod("sv :
                    "fmod("sv);
        expr->lhs()->accept(*this);
        str << ',';
        expr->rhs()->accept(*this);
        str << ')';
    } else if (op == BinaryOp::AND) {
        str << "and(";
        expr->lhs()->accept(*this);
        str << ",";
        expr->rhs()->accept(*this);
        str << ")";
    } else if (op == BinaryOp::OR) {
        str << "or(";
        expr->lhs()->accept(*this);
        str << ",";
        expr->rhs()->accept(*this);
        str << ")";
    } else {
        str << "(";
        expr->lhs()->accept(*this);
        switch (op) {
            case BinaryOp::ADD:
                str << '+';
                break;
            case BinaryOp::SUB:
                str << '-';
                break;
            case BinaryOp::MUL: {
                str << '*';
            } break;
            case BinaryOp::DIV:
                str << '/';
                break;
            case BinaryOp::MOD:
                str << '%';
                break;
            case BinaryOp::BIT_AND:
                str << '&';
                break;
            case BinaryOp::BIT_OR:
                str << '|';
                break;
            case BinaryOp::BIT_XOR:
                str << '^';
                break;
            case BinaryOp::SHL:
                str << "<<"sv;
                break;
            case BinaryOp::SHR:
                str << ">>"sv;
                break;
            case BinaryOp::LESS:
                str << '<';
                break;
            case BinaryOp::GREATER:
                str << '>';
                break;
            case BinaryOp::LESS_EQUAL:
                str << "<="sv;
                break;
            case BinaryOp::GREATER_EQUAL:
                str << ">="sv;
                break;
            case BinaryOp::EQUAL:
                str << "=="sv;
                break;
            case BinaryOp::NOT_EQUAL:
                str << "!="sv;
                break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Not implemented.");
        }
        expr->rhs()->accept(*this);
        str << ")";
    }
}
void StringStateVisitor::visit(const MemberExpr *expr) {
    char const *xyzw = "xyzw";
    if (expr->is_swizzle()) {
        accessCount++;
        expr->self()->accept(*this);
        accessCount--;
        str << '.';
        for (auto i : vstd::range(expr->swizzle_size())) {
            str << xyzw[expr->swizzle_index(i)];
        }

    } else {
        vstd::StringBuilder curStr;
        StringStateVisitor vis(f, curStr, util);
        expr->self()->accept(vis);
        // A member of a word that mixes 16-bit scalars with one-byte bools lives
        // in the packed `uint` of that word (see `PackedStructMember`).
        if (auto packed = util->GetPackedStructMember(
                expr->self()->type(), expr->member_index())) {
            str << "_lc_pack_get_"sv << packed_kind_suffix(packed->kind) << '('
                << curStr << ".v"sv << vstd::to_string(packed->word_index) << ','
                << vstd::to_string(packed->byte_offset) << ')';
            return;
        }
        str << curStr << ".v"sv << vstd::to_string(expr->member_index());
        auto t = expr->type();
        // Not a special vec3

        if (t->is_vector() && t->dimension() == 3 && !t->element()->is_bool()) {
            auto self_type = expr->self()->type();
            // Internal raster structs (_mesh, v2p) emit raw vectors — no .v wrapper
            bool is_internal = (util->opt->internalStruct.find(self_type) != util->opt->internalStruct.end());
            if (!is_internal && !(self_type->is_structure() && !self_type->member_attributes().empty())) [[likely]] {
                str << ".v"sv;
            }
        }
    }
}
void StringStateVisitor::visit(const AccessExpr *expr) {
    auto t = expr->range()->type();
    auto basicAccess = [&]() {
        accessCount++;
        expr->range()->accept(*this);
        accessCount--;
        str << '[';
        expr->index()->accept(*this);
        str << ']';
    };
    if (expr->range()->tag() == Expression::Tag::REF) {
        auto variable = static_cast<RefExpr const *>(expr->range())->variable();
        if (variable.tag() == Variable::Tag::SHARED) {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->is_matrix() && t->dimension() == 3u) {
                str << ".xyz"sv;
            } else if (t->is_array() && t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
            return;
        }
    }
    //    else if (expr->range()->tag() == Expression::Tag::CONSTANT) {
    //        basicAccess();
    //        return;
    //    }
    switch (t->tag()) {
        case Type::Tag::BUFFER:
        case Type::Tag::VECTOR: {
            basicAccess();
        } break;
        case Type::Tag::COOPERATIVE_VECTOR: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << ".Get(";
            expr->index()->accept(*this);
            str << ')';
        } break;
        case Type::Tag::MATRIX: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << '[';
            expr->index()->accept(*this);
            str << ']';
            if (accessCount == 0 && t->dimension() == 3u) {
                str << ".xyz"sv;
            }
        } break;
        default: {
            accessCount++;
            expr->range()->accept(*this);
            accessCount--;
            str << ".v[";
            expr->index()->accept(*this);
            str << ']';
            if (t->element()->is_vector() && t->element()->dimension() == 3) {
                str << ".v"sv;
            }
        } break;
    }
}
void StringStateVisitor::visit(const RefExpr *expr) {
    Variable v = expr->variable();
    vstd::StringBuilder tempStr;
    util->GetVariableName(f, v, tempStr);
    util->RegistStructType(v.type());
    str << tempStr;
}

void StringStateVisitor::visit(const LiteralExpr *expr) {
    if (literalBrace) {
        str << '(';
    }
    luisa::visit(
        [&]<typename T>(T const &value) -> void {
            PrintValue<T> prt;
            if constexpr (std::is_same_v<T, bfloat16> ||
                          std::is_same_v<T, bfloat162> ||
                          std::is_same_v<T, bfloat163> ||
                          std::is_same_v<T, bfloat164>) {
                prt(value, str, util->opt->bf16_fallback);
            } else {
                prt(value, str);
            }
        },
        expr->value());
    if (literalBrace) {
        str << ')';
    }
}
void StringStateVisitor::visit(const CallExpr *expr) {
    util->GetFunctionName(expr, str, *this);
}

void StringStateVisitor::visit(const StringIDExpr *expr) {
    str << "((";
    util->GetTypeName(*expr->type(), str, Usage::READ);
    str << ")" << luisa::hash_value(expr->data()) << "ull)";
}

void StringStateVisitor::visit(const CastExpr *expr) {
    if (expr->type() == expr->expression()->type()) [[unlikely]] {
        expr->expression()->accept(*this);
        return;
    }
    // Software BF16 (builtin/bfloat16_fallback.bytes): the value lives in a raw
    // 16-bit pattern, so a value conversion is a `_lc_bf16_*` helper call and a
    // bit cast is a plain cast between two 16-bit patterns.
    if (util->opt->bf16_fallback &&
        (CodegenUtility::IsBf16(expr->type()) ||
         CodegenUtility::IsBf16(expr->expression()->type()))) {
        auto dst_bf16 = CodegenUtility::IsBf16(expr->type());
        auto src_bf16 = CodegenUtility::IsBf16(expr->expression()->type());
        auto emit_dst_cast = [&]() noexcept {
            // BF16 scalars and vectors share one storage type, so the outer
            // cast performs the broadcast/truncation a scalar-to-vector
            // conversion needs.
            str << '(';
            util->GetTypeName(*expr->type(), str, Usage::READ);
            str << ')';
        };
        if (expr->op() == CastOp::BITWISE || (dst_bf16 && src_bf16)) {
            emit_dst_cast();
            expr->expression()->accept(*this);
            return;
        }
        auto element_tag = [](Type const *t) noexcept {
            return t->is_vector() ? t->element()->tag() : t->tag();
        };
        if (dst_bf16) {
            vstd::string_view helper;
            switch (element_tag(expr->expression()->type())) {
                case Type::Tag::BOOL: helper = "from_bool"sv; break;
                case Type::Tag::FLOAT16: helper = "from_f16"sv; break;
                case Type::Tag::FLOAT32:
                case Type::Tag::FLOAT64: helper = "from_f32"sv; break;
                case Type::Tag::INT8:
                case Type::Tag::INT16:
                case Type::Tag::INT32:
                case Type::Tag::INT64: helper = "from_i32"sv; break;
                case Type::Tag::UINT8:
                case Type::Tag::UINT16:
                case Type::Tag::UINT32:
                case Type::Tag::UINT64: helper = "from_u32"sv; break;
                default:
                    LUISA_ERROR_WITH_LOCATION(
                        "Cannot convert '{}' to BF16: unsupported source type '{}'.",
                        expr->type()->description(),
                        expr->expression()->type()->description());
            }
            emit_dst_cast();
            str << "_lc_bf16_"sv << helper << '(';
            expr->expression()->accept(*this);
            str << ')';
            return;
        }
        // BF16 source, ordinary destination.
        vstd::string_view helper;
        switch (element_tag(expr->type())) {
            case Type::Tag::BOOL: helper = "to_bool"sv; break;
            case Type::Tag::FLOAT16: helper = "to_f16"sv; break;
            case Type::Tag::FLOAT32:
            case Type::Tag::FLOAT64: helper = "to_f32"sv; break;
            case Type::Tag::INT8:
            case Type::Tag::INT16:
            case Type::Tag::INT32:
            case Type::Tag::INT64: helper = "to_i32"sv; break;
            case Type::Tag::UINT8:
            case Type::Tag::UINT16:
            case Type::Tag::UINT32:
            case Type::Tag::UINT64: helper = "to_u32"sv; break;
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot convert BF16 to '{}': unsupported destination type.",
                    expr->type()->description());
        }
        emit_dst_cast();
        str << "_lc_bf16_"sv << helper << '(';
        expr->expression()->accept(*this);
        str << ')';
        return;
    }
    str << "(";
    switch (expr->op()) {
        case CastOp::STATIC:
            str << '(';
            util->GetTypeName(*expr->type(), str, Usage::READ);
            str << ')';
            expr->expression()->accept(*this);
            break;
        case CastOp::BITWISE: {
            auto type = expr->type();
            while (type->is_vector()) {
                type = type->element();
            }
            auto arg_type = expr->expression()->type();
            auto bit_cast_vec_to_int = [&]() {
                if (arg_type->dimension() == 2 && arg_type->element()->size() == 2) {
                    str << "Vec2AsInt<"sv;
                    util->GetTypeName(*arg_type, str, Usage::READ, true);
                    str << ',';
                    util->GetTypeName(*expr->type(), str, Usage::READ, true);
                    str << '>';
                } else if (arg_type->dimension() == 4 && arg_type->element()->is_bool()) {
                    str << "Bool4AsInt<"sv;
                    util->GetTypeName(*expr->type(), str, Usage::READ, true);
                    str << '>';
                } else {
                    LUISA_ERROR("Vector to int unsupported.");
                }
            };
            auto bit_cast_int_to_vec = [&]() {
                if (expr->type()->dimension() == 2 && expr->type()->element()->size() == 2) {
                    str << "IntAsVec2<"sv;
                    util->GetTypeName(*arg_type, str, Usage::READ, true);
                    str << ',';
                    util->GetTypeName(*expr->type(), str, Usage::READ, true);
                    str << '>';
                } else if (expr->type()->dimension() == 4 && expr->type()->element()->is_bool()) {
                    str << "IntAsBool4"sv;
                } else {
                    LUISA_ERROR("Int to vector unsupported.");
                }
            };
            // Vector to int
            if (expr->type()->dimension() < arg_type->dimension()) {
                LUISA_ASSERT(expr->type()->is_int() || expr->type()->is_uint(), "Only support vector to scalar");
                bit_cast_vec_to_int();
            }
            // Int to vector
            else if (expr->type()->dimension() > arg_type->dimension()) {
                LUISA_ASSERT(arg_type->is_int() || arg_type->is_uint(), "Only support vector to scalar");
                bit_cast_int_to_vec();
            } else {
                switch (type->tag()) {
                    case Type::Tag::FLOAT16:
                    case Type::Tag::FLOAT32:
                        str << "asfloat"sv;
                        break;
                    case Type::Tag::INT8:
                    case Type::Tag::INT16:
                    case Type::Tag::INT32:
                    case Type::Tag::INT64:
                        str << "asint"sv;
                        break;
                    case Type::Tag::UINT8:
                    case Type::Tag::UINT16:
                    case Type::Tag::UINT32:
                    case Type::Tag::UINT64:
                        str << "asuint"sv;
                        break;
                    default:
                        LUISA_ERROR_WITH_LOCATION(
                            "Bitwise cast not implemented for type '{}'.",
                            type->description());
                }
            }
            str << '(';
            expr->expression()->accept(*this);
            str << ')';
        } break;
    }
    str << ")";
}

void StringStateVisitor::visit(const ConstantExpr *expr) {
    util->GetConstName(expr->data().hash(), expr->data(), str);
}

void StringStateVisitor::visit(const BreakStmt *state) {
    str << "break;\n";
}
void StringStateVisitor::visit(const ContinueStmt *state) {

    str << "continue;\n";
}
void StringStateVisitor::visit(const ReturnStmt *state) {
    if (state->expression()) {
        str << "return ";
        state->expression()->accept(*this);
        str << ";\n";
    } else {
        str << "return;\n";
    }
}
void StringStateVisitor::visit(const ScopeStmt *state) {
    for (auto &&i : state->statements()) {
        i->accept(*this);
        switch (i->tag()) {
            case Statement::Tag::BREAK:
            case Statement::Tag::CONTINUE:
            case Statement::Tag::RETURN:
                return;
            default: break;
        }
    }
}
void StringStateVisitor::visit(const AutoDiffStmt *stmt) {
    visit(stmt->body());
}
void StringStateVisitor::visit(const PrintStmt *stmt) {
    vstd::fixed_vector<Type const *, 16> types;
    vstd::push_back_func(
        types,
        stmt->arguments().size(),
        [&](size_t i) {
            return stmt->arguments()[i]->type();
        });
    auto structType = Type::structure(types);
    vstd::string counterName = "_p";
    vstd::to_string(printCount, counterName);
    str << "uint "sv << counterName << ";\nInterlockedAdd(_printCounter[0],"sv;
    size_t align = std::max<size_t>(structType->alignment(), 4);
    size_t ele_size = structType->size() + align;
    ele_size = ((ele_size + 15ull) & (~15ull));
    vstd::to_string(ele_size, str);
    str << ',' << counterName << ");\n"sv
        << "if("sv << counterName << "<1048576){\n"sv;
    {
        vstd::StringBuilder typeName;
        util->GetTypeName(*structType, typeName, Usage::READ_WRITE);
        vstd::string dataName = "_pv";
        vstd::to_string(printCount, dataName);
        str << typeName << ' ' << dataName << "=("sv << typeName << ")0;\n"sv;
        for (auto i : vstd::range(static_cast<int64>(types.size()))) {
            str << dataName << ".v"sv;
            vstd::to_string(i, str);
            auto t = types[i];
            if (t->is_vector() && t->dimension() == 3) {
                str << ".v"sv;
            }
            str << '=';
            stmt->arguments()[i]->accept(*this);
            str << ";\n"sv;
        }
        str << "_printBuffer.Store("sv
            << counterName << ',';
        auto printerIdx = util->AddPrinter(stmt->format(), structType);
        vstd::to_string(printerIdx, str);
        str << ");\n"sv
            << "_printBuffer.Store("sv
            << counterName << '+' + vstd::to_string(align) + ',' << dataName << ");\n"sv;
        printCount++;
    }
    str << "}\n"sv;
}
void StringStateVisitor::visit(const CommentStmt *state) {
#ifndef NDEBUG
    str << "/* " << state->comment() << " */\n";
#endif
}
void StringStateVisitor::visit(const IfStmt *state) {
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Branch)) != 0) {
        str << "[[branch]] "sv;
    } else if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Flatten)) != 0) {
        str << "[[flatten]] "sv;
    }
    util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
    str << "if(";
    state->condition()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->true_branch()->accept(*this);
    }
    if (!state->false_branch()->statements().empty()) {
        str << "else";
        {
            Scope scope{this};
            state->false_branch()->accept(*this);
        }
    }
}
void StringStateVisitor::visit(const LoopStmt *state) {
    str << "while(true)";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
void StringStateVisitor::visit(const ExprStmt *state) {
    state->expression()->accept(*this);
    str << ";\n";
}
void StringStateVisitor::visit(const SwitchStmt *state) {
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Branch)) != 0) {
        str << "[[branch]] "sv;
    } else if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::Flatten)) != 0) {
        str << "[[flatten]] "sv;
    }
    if ((luisa::to_underlying(util->opt->cond_opt_value) & luisa::to_underlying(CodegenStackData::CondOptValue::ForceCase)) != 0) {
        str << "[[forcecase]] "sv;
    }
    switchCount.emplace_back(0);
    auto dsp = vstd::scope_exit([&]() {
        switchCount.pop_back();
    });
    util->opt->cond_opt_value = CodegenStackData::CondOptValue::None;
    str << "switch(";
    state->expression()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
    if (switchCount.back() == 0) {
        str << "{ default: break; }\n";
    }
}
void StringStateVisitor::visit(const SwitchCaseStmt *state) {
    if (state->body()->statements().empty()) {
        return;
    }
    switchCount.back()++;
    for (auto expression : state->expressions()) {
        str << "case ";
        expression->accept(*this);
        str << ":";
    }
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
    if (std::none_of(state->body()->statements().begin(),
                     state->body()->statements().end(),
                     [](const auto &stmt) {
                         return stmt->tag() == Statement::Tag::BREAK;
                     })) {
        str << "break;\n";
    }
}
void StringStateVisitor::visit(const SwitchDefaultStmt *state) {
    if (state->body()->statements().empty()) {
        return;
    }
    switchCount.back()++;
    str << "default:";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
    if (std::none_of(state->body()->statements().begin(),
                     state->body()->statements().end(),
                     [](const auto &stmt) {
                         return stmt->tag() == Statement::Tag::BREAK;
                     })) {
        str << "break;\n";
    }
}

void StringStateVisitor::visit(const AssignStmt *state) {
    if (is_local_undefined_lifetime_seed(state)) { return; }
    // A member that lives in a packed 32-bit word (a 1-byte bool mixed with 16-bit
    // scalars, see `PackedStructMember`) is part of the word's single `uint`
    // member, so writing it is a read-modify-write through the matching
    // `_lc_pack_set_*` helper.
    if (state->lhs()->tag() == Expression::Tag::MEMBER) {
        auto member = static_cast<MemberExpr const *>(state->lhs());
        if (auto packed = util->GetPackedStructMember(
                member->self()->type(), member->member_index())) {
            if (expression_has_call(member->self())) [[unlikely]] {
                LUISA_ERROR_WITH_LOCATION(
                    "Cannot assign to the packed struct member '{}': its owner "
                    "calls a function, and writing the member needs the owner "
                    "twice.",
                    member->type()->description());
            }
            vstd::StringBuilder owner;
            StringStateVisitor vis(f, owner, util);
            member->self()->accept(vis);
            str << owner << ".v"sv << vstd::to_string(packed->word_index)
                << "=_lc_pack_set_"sv << packed_kind_suffix(packed->kind) << '('
                << owner << ".v"sv << vstd::to_string(packed->word_index)
                << ',' << vstd::to_string(packed->byte_offset) << ',';
            state->rhs()->accept(*this);
            str << ");\n";
            return;
        }
    }
    auto is_shared = [&](const Expression *x) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            auto v = static_cast<RefExpr const *>(x)->variable();
            if (v.tag() == Variable::Tag::SHARED) {
                return true;
            }
        }
        return false;
    };
    auto is_custom = [&](const Expression *x, Variable &v) noexcept {
        if (x->tag() == Expression::Tag::REF) {
            v = static_cast<RefExpr const *>(x)->variable();
            if (v.type()->is_custom()) {
                return true;
            }
        }
        return false;
    };
    // shared variables are not wrapped in array
    // structs, so some hack is necessary
    bool isLazyDecl = false;
    Variable rqVar;
    bool lhs_is_shared{false};
    auto is_rayquery = [&]() {
        if (state->rhs()->tag() != Expression::Tag::CALL) return false;
        auto s = static_cast<CallExpr const *>(state->rhs());
        return (s->op() == CallOp::RAY_TRACING_QUERY_ANY || s->op() == CallOp::RAY_TRACING_QUERY_ALL);
    };
    if (is_custom(state->lhs(), rqVar)) {
        auto iter = lazyDeclVars.find(rqVar);
        vstd::StringBuilder var_name;
        util->GetVariableName(f, rqVar, var_name);
        if (iter != lazyDeclVars.end()) {
            util->GetTypeName(*rqVar.type(), str, Usage::READ);
            str << ' '
                << var_name;
            lazyDeclVars.erase(iter);
            isLazyDecl = true;
        }
        if (is_rayquery()) {
            str << ";\n";
            state->rhs()->accept(*this);
            str << ',';
            state->lhs()->accept(*this);
            str << ");\n";
            return;
        }
    }
    // Detect assignment to cooperative vector element: v[i] = x → v.Set(x, i)
    auto lhs_access = state->lhs()->tag() == Expression::Tag::ACCESS ?
                          static_cast<AccessExpr const *>(state->lhs()) :
                          nullptr;
    bool lhs_is_coopvec_element = lhs_access &&
                                  lhs_access->range()->type()->is_cooperative_vector();
    if (lhs_is_coopvec_element) {
        // Generate: range.Set(rhs, index)
        lhs_access->range()->accept(*this);
        str << ".Set(";
        state->rhs()->accept(*this);
        str << ',';
        lhs_access->index()->accept(*this);
        str << ");\n";
        return;
    }
    // DXC can miscompile `m[col] = floatN(...)` (and later mat-vec products) when
    // columns share components. Store each element instead of a vector write.
    bool lhs_is_matrix_column = lhs_access &&
                                lhs_access->range()->type()->is_matrix() &&
                                state->rhs()->type() &&
                                state->rhs()->type()->is_vector();
    if (lhs_is_matrix_column) {
        auto dim = lhs_access->range()->type()->dimension();
        auto tmp = util->GetNewTempVarName();
        util->GetTypeName(*state->rhs()->type(), str, Usage::READ);
        str << ' ' << tmp << ";\n"sv;
        str << tmp << '=';
        state->rhs()->accept(*this);
        str << ";\n"sv;
        static constexpr char xyzw[] = "xyzw";
        for (auto i = 0u; i < dim; i++) {
            accessCount++;
            lhs_access->range()->accept(*this);
            accessCount--;
            str << '[';
            lhs_access->index()->accept(*this);
            str << "]["sv;
            vstd::to_string(i, str);
            str << "]="sv;
            str << tmp << '.' << xyzw[i] << ";\n"sv;
        }
        return;
    }
    auto rhs_is_shared = is_shared(state->rhs());
    if (!isLazyDecl) {
        lhs_is_shared = is_shared(state->lhs());
        if (!lhs_is_shared && rhs_is_shared) {
            str << "(";
            state->lhs()->accept(*this);
            str << ").v";
        } else {
            state->lhs()->accept(*this);
        }
    }
    str << '=';
    // The AST keeps a plain literal on the right-hand side of a BF16
    // declaration ("BFloat16 zero = 0.0f;") and relies on the target type's
    // implicit conversion.  Software BF16 has no such conversion (its storage is
    // a raw 16-bit pattern), so the narrowing is explicit here.
    auto rhs_needs_bf16_narrow =
        util->opt->bf16_fallback &&
        CodegenUtility::IsBf16(state->lhs()->type()) &&
        !CodegenUtility::IsBf16(state->rhs()->type());
    if (rhs_needs_bf16_narrow) {
        str << "_lc_bf16_from_f32("sv;
    }
    if (lhs_is_shared && !rhs_is_shared) {
        str << "(";
        state->rhs()->accept(*this);
        str << ").v";
    } else {
        state->rhs()->accept(*this);
    }
    if (rhs_needs_bf16_narrow) {
        str << ')';
    }
    str << ";\n";
}
void StringStateVisitor::visit(const ForStmt *state) {
    str << "for(";
    //    state->variable()->accept(*this);
    str << ';';
    state->condition()->accept(*this);
    str << ';';
    state->variable()->accept(*this);
    str << "+=";
    state->step()->accept(*this);
    str << ")";
    {
        Scope scope{this};
        state->body()->accept(*this);
    }
}
void StringStateVisitor::visit(const RayQueryStmt *stmt) {
    // Ray queries (inline or the RayQuery object form) are not implemented by
    // the software ray-tracing fallback: it has no candidate/committed-hit
    // machinery to drive this loop.  Report it instead of emitting a loop over
    // a resource the fallback never declares.
    if (util->opt->fallback_rtx) {
        LUISA_ERROR("RayQuery (RAY_TRACING_QUERY_*) is not available on the "
                    "software ray-tracing fallback "
                    "(DeviceConfigExt::use_fallback_rtx()).  The fallback answers "
                    "closest-hit and any-hit tracing only; run on a device with "
                    "hardware ray tracing and leave use_fallback_rtx() false, or "
                    "rewrite the shader with Accel::intersect()/intersect_any().");
    }
    str << "{\n"sv;
    str << "while("sv;
    stmt->query()->accept(*this);
    str << ".Proceed()){\n"sv
        << "if("sv;
    stmt->query()->accept(*this);
    str << ".CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE){\n"sv;
    stmt->on_triangle_candidate()->accept(*this);
    str << "}else{\n"sv;
    stmt->on_procedural_candidate()->accept(*this);
    str << "}}}\n"sv;
}
void StringStateVisitor::visit(const SuspendStmt *) {
}
StringStateVisitor::StringStateVisitor(
    Function f,
    vstd::StringBuilder &str,
    CodegenUtility *util)
    : f(f), util(util), str(str), switchCount{}, lazyDeclVars{} {
}
void StringStateVisitor::VisitFunction(Function func) {
    lazyDeclVars.clear();
    for (auto &&v : func.local_variables()) {
        Usage usage = func.variable_usage(v.uid());
        if (usage == Usage::NONE) [[unlikely]] {
            continue;
        }
        if (v.type()->tag() == Type::Tag::CUSTOM) {
            auto desc = v.type()->description();
            // rayquery need specialization to workaround DXC's bug
            if (desc == "LC_RayQueryAll"sv || desc == "LC_RayQueryAny"sv) {
                lazyDeclVars.emplace(v);
                continue;
            }
        }
        if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(Usage::WRITE)) == 0) {
            str << "const "sv;
        }
        vstd::StringBuilder typeName;
        util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
        vstd::StringBuilder varName;
        util->GetVariableName(f, v, varName);

        str << typeName << ' ' << varName;
        if (!(v.type()->is_resource() || v.type()->is_custom() || v.type()->is_cooperative_vector())) [[likely]] {
            str << "=("sv << typeName << ")0"sv;
        }
        str << ";\n";
    }
    if (sharedVariables) {
        size_t shared_size{};
        for (auto &&v : func.shared_variables()) {
            // FIXME: redundant creation of string
            vstd::StringBuilder typeName;
            util->GetTypeName(*v.type(), typeName, f.variable_usage(v.uid()));
            sharedVariables->emplace(f, v);
            shared_size += v.type()->size();
        }

        // SM 6.10 will increase to 64kb in the future
        LUISA_ASSERT(shared_size <= 32768, "Shared memory size must be less than 32kb.");
    }
    func.body()->accept(*this);
}

StringStateVisitor::Scope::Scope(StringStateVisitor *self)
    : self(self) {
    self->str << "{\n"sv;
}
}// namespace lc::hlsl
