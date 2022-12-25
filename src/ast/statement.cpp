//
// Created by Mike Smith on 2021/11/1.
//

#include <ast/statement.h>

namespace luisa::compute {

uint64_t Statement::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        static auto seed = hash_value("__hash_statement"sv);
        std::array a{static_cast<uint64_t>(_tag), _compute_hash()};
        _hash = hash64(&a, sizeof(a), seed);
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
    std::array a{hl, hr};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t IfStmt::_compute_hash() const noexcept {
    std::array a{_condition->hash(), _true_branch.hash(), _false_branch.hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t LoopStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t ExprStmt::_compute_hash() const noexcept {
    return _expr->hash();
}

uint64_t SwitchStmt::_compute_hash() const noexcept {
    std::array a{_body.hash(), _expr->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t SwitchCaseStmt::_compute_hash() const noexcept {
    std::array a{_body.hash(), _expr->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t SwitchDefaultStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t ForStmt::_compute_hash() const noexcept {
    std::array a{_body.hash(), _var->hash(), _cond->hash(), _step->hash()};
    return hash64(&a, sizeof(a), hash64_default_seed);
}

uint64_t CommentStmt::_compute_hash() const noexcept {
    return hash_value(_comment);
}

}// namespace luisa::compute
