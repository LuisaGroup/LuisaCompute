#include <luisa/ast/statement.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>

namespace luisa::compute {

uint64_t Statement::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        static auto statement_seed = hash_value("__hash_statement"sv);
        _hash = hash_combine({static_cast<uint64_t>(_tag), _compute_hash()}, statement_seed);
        _hash_computed = true;
    }
    return _hash;
}

uint64_t BreakStmt::_compute_hash() const noexcept {
    return hash64_default_seed;
}

uint64_t ContinueStmt::_compute_hash() const noexcept {
    return hash64_default_seed;
}

uint64_t ReturnStmt::_compute_hash() const noexcept {
    return _expr == nullptr ? hash64_default_seed : _expr->hash();
}

uint64_t ScopeStmt::_compute_hash() const noexcept {
    auto h = hash64_default_seed;
    for (auto &&s : _statements) {
        auto hh = s->hash();
        h = hash64(&hh, sizeof(hh), h);
    }
    return h;
}

const Statement *ScopeStmt::pop() noexcept {
    auto stmt = _statements.back();
    _statements.pop_back();
    return stmt;
}

uint64_t AssignStmt::_compute_hash() const noexcept {
    auto hl = _lhs->hash();
    auto hr = _rhs->hash();
    return hash_combine({hl, hr});
}

uint64_t IfStmt::_compute_hash() const noexcept {
    return hash_combine({_condition->hash(),
                         _true_branch.hash(),
                         _false_branch.hash()});
}

uint64_t LoopStmt::_compute_hash() const noexcept {
    return _body.hash();
}

void SuspendStmt::_validate_extension_bindings() const noexcept {
    auto is_writable = [](auto &&self,
                          const Expression *expression) noexcept -> bool {
        if (expression == nullptr) { return false; }
        switch (expression->tag()) {
            case Expression::Tag::REF: return true;
            case Expression::Tag::ACCESS:
                return self(self,
                            static_cast<const AccessExpr *>(expression)
                                ->range());
            case Expression::Tag::MEMBER: {
                auto *member =
                    static_cast<const MemberExpr *>(expression);
                return (!member->is_swizzle() ||
                        member->swizzle_size() == 1u) &&
                       self(self, member->self());
            }
            default: return false;
        }
    };
    luisa::vector<bool> bound(_extension_binding_values.size(), false);
    for (auto &&extension : _extensions) {
        LUISA_ASSERT(extension != nullptr,
                     "Coroutine suspend extension must be non-null.");
        for (auto &&binding : extension->bindings()) {
            LUISA_ASSERT(binding.index <
                             _extension_binding_values.size(),
                         "Coroutine suspend extension '{}' binding '{}' "
                         "references owner index {} outside [0, {}).",
                         extension->schema(), binding.name,
                         binding.index,
                         _extension_binding_values.size());
            LUISA_ASSERT(
                !bound[binding.index],
                "Coroutine suspend extension '{}' binding '{}' reuses owner "
                "index {}; each binding must have its own owner slot.",
                extension->schema(), binding.name, binding.index);
            bound[binding.index] = true;
            auto *value = _extension_binding_values[binding.index];
            LUISA_ASSERT(value != nullptr && value->type() != nullptr,
                         "Coroutine suspend extension '{}' binding '{}' "
                         "must reference a typed AST value.",
                         extension->schema(), binding.name);
            switch (binding.access) {
                case CoroSuspendBindingAccess::read: break;
                case CoroSuspendBindingAccess::write:
                    LUISA_ASSERT(
                        is_writable(is_writable, value),
                        "Coroutine suspend extension '{}' write binding '{}' "
                        "must reference a writable AST lvalue.",
                        extension->schema(), binding.name);
                    break;
                case CoroSuspendBindingAccess::read_write:
                    LUISA_ASSERT(
                        is_writable(is_writable, value),
                        "Coroutine suspend extension '{}' read-write binding "
                        "'{}' must reference a writable AST lvalue.",
                        extension->schema(), binding.name);
                    break;
            }
        }
    }
    for (size_t i = 0u; i < bound.size(); ++i) {
        LUISA_ASSERT(bound[i],
                     "Coroutine suspend extension owner binding index {} "
                     "is not referenced by an extension.",
                     i);
    }
}

void SuspendStmt::_mark_extension_bindings() const noexcept {
    _validate_extension_bindings();
    for (auto &&extension : _extensions) {
        for (auto &&binding : extension->bindings()) {
            auto *value = _extension_binding_values[binding.index];
            switch (binding.access) {
                case CoroSuspendBindingAccess::read:
                    value->mark(Usage::READ);
                    break;
                case CoroSuspendBindingAccess::write:
                    value->mark(Usage::WRITE);
                    break;
                case CoroSuspendBindingAccess::read_write:
                    value->mark(Usage::READ_WRITE);
                    break;
            }
        }
    }
}

uint64_t SuspendStmt::_compute_hash() const noexcept {
    auto h = hash_combine(
        {static_cast<uint64_t>(_token), hash_value(_name)});
    // Export order is observable only as an ABI declaration order. Delimit
    // each (name, value) pair so no concatenation ambiguity can alias another
    // boundary contract.
    h = hash_value(_frame_exports.size(), h);
    for (auto &&frame_export : _frame_exports) {
        h = hash_value(frame_export.name, h);
        h = hash_value(frame_export.value->hash(), h);
    }
    h = hash_value(_extensions.size(), h);
    for (auto &&extension : _extensions) {
        h = hash_value(extension->schema(), h);
        h = hash_value(extension->version(), h);
        h = hash_value(extension->is_annotation(), h);
        h = hash_value(static_cast<uint8_t>(extension->fallback()), h);
        h = hash_value(extension->bindings().size(), h);
        for (auto &&binding : extension->bindings()) {
            h = hash_value(binding.name, h);
            h = hash_value(static_cast<uint8_t>(binding.access), h);
            h = hash_value(static_cast<uint8_t>(binding.lifetime), h);
            h = hash_value(binding.index, h);
            h = hash_value(
                _extension_binding_values[binding.index]->hash(), h);
        }
        h = hash_value(extension->attributes().size(), h);
        for (auto &&attribute : extension->attributes()) {
            h = hash_value(attribute.name, h);
            h = hash_value(attribute.value.index(), h);
            luisa::visit(
                [&](auto &&value) noexcept {
                    h = hash_value(value, h);
                },
                attribute.value);
        }
    }
    return h;
}

uint64_t ExprStmt::_compute_hash() const noexcept {
    return _expr->hash();
}

uint64_t SwitchStmt::_compute_hash() const noexcept {
    return hash_combine({_body.hash(), _expr->hash()});
}

uint64_t SwitchCaseStmt::_compute_hash() const noexcept {
    return hash_combine({_body.hash(), _expr->hash()});
}

uint64_t SwitchDefaultStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t ForStmt::_compute_hash() const noexcept {
    return hash_combine({_body.hash(),
                         _var->hash(),
                         _cond->hash(),
                         _step->hash()});
}

uint64_t CommentStmt::_compute_hash() const noexcept {
    return hash_value(_comment);
}

uint64_t RayQueryStmt::_compute_hash() const noexcept {
    return hash_combine({_query->hash(),
                         _on_triangle_candidate.hash(),
                         _on_procedural_candidate.hash()});
}

uint64_t AutoDiffStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t PrintStmt::_compute_hash() const noexcept {
    auto h = luisa::hash_value(_format);
    for (auto &&e : _args) {
        h = luisa::hash_value(e->hash(), h);
    }
    return h;
}

PrintStmt::PrintStmt(luisa::string fmt, luisa::vector<const Expression *> args) noexcept
    : Statement{Tag::PRINT}, _format{std::move(fmt)}, _args{std::move(args)} {
    for (auto arg : _args) { arg->mark(Usage::READ); }
}

uint64_t DebugBreakStmt::_compute_hash() const noexcept {
    auto h = luisa::hash_value(_wrapper);
    for (auto &&w : _watches) {
        h = luisa::hash_value(w->hash(), h);
    }
    return h;
}

DebugBreakStmt::DebugBreakStmt(Wrapper *wrapper, luisa::vector<const Expression *> watches) noexcept
    : Statement{Tag::DEBUG_BREAK}, _wrapper{wrapper}, _watches{std::move(watches)} {}

void StmtVisitor::visit(const AutoDiffStmt *stmt) {
    // reports error by default since it should be
    // handled by the IR when reaching the backend
    LUISA_ERROR_WITH_LOCATION("AutoDiffStmt is not supported.");
}

void StmtVisitor::visit(const PrintStmt *stmt) {
    // Not supporting the print statement is not
    // critical, so we just log a warning.
    LUISA_WARNING_WITH_LOCATION("PrintStmt is not supported.");
}

void StmtVisitor::visit(const DebugBreakStmt *stmt) {
    // Not supporting the debug break statement is not
    // critical, so we just log a warning.
    LUISA_WARNING_WITH_LOCATION("DebugBreakStmt is not supported.");
}

}// namespace luisa::compute
