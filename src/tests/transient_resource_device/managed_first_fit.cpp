#include "managed_first_fit.h"
#include <luisa/core/mathematics.h>
#include <luisa/core/logging.h>
namespace luisa::utils
{
ManagedFirstFit::Node::Node() noexcept = default;

ManagedFirstFit::ManagedFirstFit(size_t size, size_t alignment) noexcept
    : _alignment{ luisa::next_pow2(alignment) }
    , _node_pool{ 256 }
{
    _free_list._next = _node_pool.create();
    _free_list._size = size;
    _free_list._next->_next = nullptr;
    _free_list._next->_offset = 0u;
    _free_list._next->_size = size;
}

ManagedFirstFit::~ManagedFirstFit() noexcept { _destroy(); }

ManagedFirstFit::ManagedFirstFit(ManagedFirstFit&& another) noexcept
    : _alignment{ another._alignment }
    , _node_pool{ std::move(another._node_pool) }
{
}

void ManagedFirstFit::clean_all() noexcept
{
    _free_list._offset = 0;
	_node_pool.destroy_all();
    _free_list._next = _node_pool.create();
    _free_list._next->_next = nullptr;
    _free_list._next->_offset = 0u;
    _free_list._next->_size = _free_list._size;
}

ManagedFirstFit::Node* ManagedFirstFit::allocate(size_t size) noexcept
{
    auto mask = _alignment - 1u;
    auto aligned_size = (size + mask) & ~mask;
    // walk the free list
    for (auto p = &_free_list; p->_next != nullptr; p = p->_next)
    {
        // found available node
        if (auto node = p->_next; node->_size >= aligned_size)
        {
            // compute aligned size
            // has remaining size, split the node
            if (node->_size > aligned_size)
            {
                auto alloc_node = _node_pool.create();
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
ManagedFirstFit::Node* ManagedFirstFit::allocate_best_fit(size_t size) noexcept
{
    auto mask = _alignment - 1u;
    auto aligned_size = (size + mask) & ~mask;
    struct FitResult {
        Node* p;
        Node* node;
        size_t remained;
    };
    FitResult result{
        nullptr, nullptr, std::numeric_limits<size_t>::max()
    };
    // walk the free list
    for (auto p = &_free_list; p->_next != nullptr; p = p->_next)
    {
        // found available node
        if (auto node = p->_next; node->_size >= aligned_size)
        {
            auto remained = node->_size - aligned_size;
            if (remained > 0)
            {
                // This one is better
                if (remained < result.remained)
                {
                    result.remained = remained;
                    result.p = p;
                    result.node = node;
                }
            }
            else
            {
                result.remained = 0;
                result.p = p;
                result.node = node;
                break;
            }
            // no more remaining size, use the whole node
            // p->_next = node->_next;
        }
    }
    if (result.node == nullptr)
        return nullptr;
    if (result.remained == 0)
    {
        result.p->_next = result.node->_next;
        return result.node;
    }
    auto alloc_node = _node_pool.create();
    alloc_node->_offset = result.node->_offset;
    alloc_node->_size = aligned_size;
    result.node->_offset += aligned_size;
    result.node->_size -= aligned_size;
    return alloc_node;
}

void ManagedFirstFit::free(ManagedFirstFit::Node* node) noexcept
{
    if (node != nullptr) [[likely]]
    {
        auto first = _free_list._next;
        size_t node_end = node->_offset + node->_size;
        if (first == nullptr || node_end < first->_offset)
        { // insert as first, no merge
            node->_next = first;
            _free_list._next = node;
            return;
        }
        if (node_end == first->_offset)
        { // insert as first, merge
            first->_offset = node->_offset;
            first->_size += node->_size;
            _node_pool.destroy(node);
            return;
        }
        // should not be the first node
        for (auto p = first; p != nullptr; p = p->_next)
        {
            // found the node after which we should insert...
            auto next = p->_next;
            auto prev_end = p->_offset + p->_size;
            if (prev_end < node->_offset)
            { // no merge with prev
                // no merge with next
                if (next == nullptr || node_end < next->_offset)
                {
                    node->_next = p->_next;
                    p->_next = node;
                    return;
                }
                // merge with next
                if (node_end == next->_offset)
                {
                    next->_offset = node->_offset;
                    next->_size += node->_size;
                    _node_pool.destroy(node);
                    return;
                }
            }
            else if (prev_end == node->_offset)
            { // merge with prev
                // no merge with next
                if (next == nullptr || node_end < next->_offset)
                {
                    p->_size += node->_size;
                    _node_pool.destroy(node);
                    return;
                }
                // merge with prev & next
                if (node_end == next->_offset)
                {
                    p->_size += node->_size + next->_size;
                    p->_next = next->_next;
                    _node_pool.destroy(node);
                    _node_pool.destroy(next);
                    return;
                }
            }
        }
        LUISA_ERROR_WITH_LOCATION(
            "Invalid node for first-fit free list "
            "(offset = {}, size = {}).",
            node->_offset, node->_size
        );
    }
}

inline void ManagedFirstFit::_destroy() noexcept
{
    if (_free_list._size != 0u)
    {
        if (_free_list._next == nullptr ||
            _free_list._next->_next != nullptr ||
            _free_list._next->_offset != 0u ||
            _free_list._next->_size != _free_list._size) [[unlikely]]
        {
            LUISA_WARNING_WITH_LOCATION("Leaks in first-fit free list.");
        }
        auto p = _free_list._next;
        while (p != nullptr)
        {
            auto node = p;
            p = p->_next;
            _node_pool.destroy(node);
        }
    }
}
} // namespace luisa::utils