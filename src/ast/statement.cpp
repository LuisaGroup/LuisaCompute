#include <luisa/ast/statement.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>

namespace luisa::compute {

uint64_t Statement::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        static auto seed = hash_value("__hash_statement"sv);
        _hash = hash_combine({static_cast<uint64_t>(_tag), _compute_hash()}, seed);
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

void StmtVisitor::visit(const AutoDiffStmt *stmt) {
    // reports error by default since it should be
    // handled by the IR when reaching the backend
    LUISA_ERROR_WITH_LOCATION("AutoDiffStmt is not supported.");
}

}// namespace luisa::compute

