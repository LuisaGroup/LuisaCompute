//
// Created by Mike Smith on 2021/8/6.
//

#include <core/logging.h>
#include <ast/op.h>

namespace luisa::compute {

CallOpSet::Iterator::Iterator(const CallOpSet &set) noexcept : _set{set} {
    while (_index != call_op_count && !_set.test(static_cast<CallOp>(_index))) {
        _index++;
    }
}

CallOp CallOpSet::Iterator::operator*() const noexcept {
    return static_cast<CallOp>(_index);
}

CallOpSet::Iterator &CallOpSet::Iterator::operator++() noexcept {
    if (_index == call_op_count) {
        LUISA_ERROR_WITH_LOCATION(
            "Walking past the end of CallOpSet.");
    }
    _index++;
    while (_index != call_op_count && !_set.test(static_cast<CallOp>(_index))) {
        _index++;
    }
    return (*this);
}

CallOpSet::Iterator CallOpSet::Iterator::operator++(int) noexcept {
    auto self = *this;
    this->operator++();
    return self;
}

bool CallOpSet::Iterator::operator==(CallOpSet::Iterator::End) const noexcept {
    return _index == call_op_count;
}

}// namespace luisa::compute
