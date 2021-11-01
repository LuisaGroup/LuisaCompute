//
// Created by Mike Smith on 2021/11/1.
//

#include <ast/statement.h>

namespace luisa::compute {

uint64_t Statement::hash() const noexcept {
    if (!_hash_computed) {
        using namespace std::string_view_literals;
        _hash = hash64(_tag, hash64(_compute_hash(), hash64("__hash_statement")));
        _hash_computed = true;
    }
    return _hash;
}

}// namespace luisa::compute
