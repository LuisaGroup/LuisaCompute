//
// Created by Mike on 2021/12/10.
//

#include <core/mathematics.h>
#include <core/pool.h>
#include <core/logging.h>
#include <core/first_fit.h>

namespace luisa {

namespace detail {
[[nodiscard]] static auto &first_fit_node_pool() noexcept {
    static Pool<FirstFit::Node> pool;
    return pool;
}
}// namespace detail

inline FirstFit::Node::Node() noexcept = default;

FirstFit::FirstFit(size_t size, size_t alignment) noexcept
    : _alignment{next_pow2(alignment)} {
    _free_list._next = detail::first_fit_node_pool().create();
    _free_list._size = size;
    _free_list._next->_next = nullptr;
    _free_list._next->_offset = 0u;
    _free_list._next->_size = size;
}

FirstFit::~FirstFit() noexcept { _destroy(); }

FirstFit::FirstFit(FirstFit &&another) noexcept
    : _alignment{another._alignment} {
}

FirstFit &FirstFit::operator=(FirstFit &&rhs) noexcept {
    if (this != &rhs) {
        _destroy();
        _free_list._next = rhs._free_list._next;
        _free_list._size = rhs._free_list._size;
        _alignment = rhs._alignment;
        rhs._free_list._size = 0u;// indicates move
    }
    return *this;
}

FirstFit::Node *FirstFit::allocate(size_t size) noexcept {
    // walk the free list
    for (auto p = &_free_list; p->_next != nullptr; p = p->_next) {
        // found available node
        if (auto node = p->_next; node->_size > size) {
            // compute aligned size
            auto mask = _alignment - 1u;
            auto aligned_size = (size & mask) == 0u ? size : (size & ~mask) + _alignment;
            // has remaining size, split the node
            if (node->_size > aligned_size) {
                auto alloc_node = detail::first_fit_node_pool().create();
                alloc_node->_offset = node->_offset;
                alloc_node->_size = aligned_size;
                node->_offset += aligned_size;
                node->_size -= aligned_size;
                return alloc_node;
            }
            // no more remaining size, use the whole node
            p->_next = node->_next;
            return node;
        }
    }
    // none available
    return nullptr;
}

void FirstFit::free(FirstFit::Node *node) noexcept {
    if (node != nullptr) [[likely]] {
        auto first = _free_list._next;
        size_t node_end = node->_offset + node->_size;
        if (first == nullptr || node_end < first->_offset) {// insert as first, no merge
            node->_next = first;
            _free_list._next = node;
            return;
        }
        if (node_end == first->_offset) {// insert as first, merge
            first->_offset = node->_offset;
            first->_size += node->_size;
            detail::first_fit_node_pool().recycle(node);
            return;
        }
        // should not be the first node
        for (auto p = first; p != nullptr; p = p->_next) {
            // found the node after which we should insert...
            auto next = p->_next;
            auto prev_end = p->_offset + p->_size;
            if (prev_end < node->_offset) {// no merge with prev
                // no merge with next
                if (next == nullptr || node_end < next->_offset) {
                    node->_next = p->_next;
                    p->_next = node;
                    return;
                }
                // merge with next
                if (node_end == next->_offset) {
                    next->_offset = node->_offset;
                    next->_size += node->_size;
                    detail::first_fit_node_pool().recycle(node);
                    return;
                }
            } else if (prev_end == node->_offset) {// merge with prev
                // no merge with next
                if (next == nullptr || node_end < next->_offset) {
                    p->_size += node->_size;
                    detail::first_fit_node_pool().recycle(node);
                    return;
                }
                // merge with prev & next
                if (node_end == next->_offset) {
                    p->_size += node->_size + next->_size;
                    p->_next = next->_next;
                    detail::first_fit_node_pool().recycle(node);
                    detail::first_fit_node_pool().recycle(next);
                    return;
                }
            }
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid node for first-fit free list "
            "(offset = {}, size = {}). Free list dump: {}.",
            node->_offset, node->_size, dump_free_list());
    }
}

inline void FirstFit::_destroy() noexcept {
    if (_free_list._size != 0u) {
        if (_free_list._next == nullptr ||
            _free_list._next->_next != nullptr ||
            _free_list._next->_offset != 0u ||
            _free_list._next->_size != _free_list._size) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION("Leaks in first-fit free list.");
        }
        auto p = _free_list._next;
        while (p != nullptr) {
            auto node = p;
            p = p->_next;
            detail::first_fit_node_pool().recycle(node);
        }
    }
}

luisa::string FirstFit::dump_free_list() const noexcept {
    luisa::string message{fmt::format("[head (size = {})]", size())};
    for (auto p = _free_list._next; p != nullptr; p = p->_next) {
        message.append(fmt::format(" -> [{}, {})", p->_offset, p->_offset + p->_size));
    }
    return message;
}

}// namespace luisa
