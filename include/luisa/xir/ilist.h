#pragma once

#include <luisa/core/stl/iterator.h>
#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

template<typename T, typename Base>
class IntrusiveNode;

template<typename T, typename Base>
class IntrusiveForwardNode;

namespace detail {

template<typename>
struct extract_intrusive_list_node {};

template<template<typename> typename List, typename T>
struct extract_intrusive_list_node<List<T>> {
    using type = T;
};

template<typename T>
using extract_intrusive_list_node_t = typename extract_intrusive_list_node<T>::type;

template<typename DerivedList>
class IntrusiveListImpl : public concepts::Noncopyable {

private:
    using Node = extract_intrusive_list_node_t<DerivedList>;

    template<typename T, typename Advance>
    class IteratorBase {

    private:
        friend class IntrusiveListImpl;
        T *_current = nullptr;
        explicit IteratorBase(T *current) noexcept : _current{current} {}

    public:
        [[nodiscard]] auto &operator*() const noexcept { return *_current; }
        [[nodiscard]] auto operator->() const noexcept { return _current; }
        [[nodiscard]] auto operator==(luisa::default_sentinel_t) const noexcept { return _current->is_sentinel(); }
        IteratorBase &operator++() noexcept {
            Advance::advance(_current);
            return *this;
        }
    };

    struct ForwardAdvance {
        template<typename U>
        void static advance(U &current) noexcept { current = current->next(); }
    };
    struct BackwardAdvance {
        template<typename U>
        void static advance(U &current) noexcept { current = current->prev(); }
    };

public:
    using Iterator = IteratorBase<Node, ForwardAdvance>;
    using ConstIterator = IteratorBase<const Node, ForwardAdvance>;
    using ReverseIterator = IteratorBase<Node, BackwardAdvance>;
    using ConstReverseIterator = IteratorBase<const Node, BackwardAdvance>;

private:
    [[nodiscard]] auto _get_head_sentinel() noexcept { return static_cast<DerivedList *>(this)->head_sentinel(); }
    [[nodiscard]] auto _get_head_sentinel() const noexcept { return static_cast<const DerivedList *>(this)->head_sentinel(); }
    [[nodiscard]] auto _get_tail_sentinel() noexcept { return static_cast<DerivedList *>(this)->tail_sentinel(); }
    [[nodiscard]] auto _get_tail_sentinel() const noexcept { return static_cast<const DerivedList *>(this)->tail_sentinel(); }

public:
    void insert_front(Node *node) noexcept { _get_head_sentinel()->insert_after_self(node); }
    void insert_back(Node *node) noexcept { _get_tail_sentinel()->insert_before_self(node); }

public:
    [[nodiscard]] bool empty() const noexcept { return _get_head_sentinel()->next() == _get_tail_sentinel(); }
    [[nodiscard]] auto &front() noexcept { return *_get_head_sentinel()->next(); }
    [[nodiscard]] auto &back() noexcept { return *_get_tail_sentinel()->prev(); }
    [[nodiscard]] const auto &front() const noexcept { return *_get_head_sentinel()->next(); }
    [[nodiscard]] const auto &back() const noexcept { return *_get_tail_sentinel()->prev(); }

    [[nodiscard]] auto begin() noexcept { return Iterator{_get_head_sentinel()->next()}; }
    [[nodiscard]] auto begin() const noexcept { return ConstIterator{_get_head_sentinel()->next()}; }
    [[nodiscard]] auto end() const noexcept { return luisa::default_sentinel; }

    [[nodiscard]] auto rbegin() noexcept { return ReverseIterator{_get_tail_sentinel()->prev()}; }
    [[nodiscard]] auto rbegin() const noexcept { return ConstReverseIterator{_get_tail_sentinel()->prev()}; }
    [[nodiscard]] auto rend() const noexcept { return luisa::default_sentinel; }

    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }

    [[nodiscard]] auto crbegin() const noexcept { return this->rbegin(); }
    [[nodiscard]] auto crend() const noexcept { return this->rend(); }
};

}// namespace detail

template<typename>
class IntrusiveList;

template<typename>
class InlineIntrusiveList;

template<typename T, typename Base = PooledObject>
class IntrusiveNode : public Base {

public:
    using Super = IntrusiveNode;
    static_assert(std::is_base_of_v<PooledObject, Base>);

protected:
    using Base::Base;

private:
    friend IntrusiveList<T>;
    friend InlineIntrusiveList<T>;

    T *_prev = nullptr;
    T *_next = nullptr;

public:
    [[nodiscard]] auto prev() noexcept { return static_cast<T *>(_prev); }
    [[nodiscard]] auto prev() const noexcept { return static_cast<const T *>(_prev); }
    [[nodiscard]] auto next() noexcept { return static_cast<T *>(_next); }
    [[nodiscard]] auto next() const noexcept { return static_cast<const T *>(_next); }
    [[nodiscard]] auto is_linked() const noexcept { return _prev != nullptr && _next != nullptr; }
    [[nodiscard]] auto is_head_sentinel() const noexcept { return _prev == nullptr; }
    [[nodiscard]] auto is_tail_sentinel() const noexcept { return _next == nullptr; }
    [[nodiscard]] auto is_sentinel() const noexcept { return is_head_sentinel() || is_tail_sentinel(); }

public:
    virtual void remove_self() noexcept {
        if (is_linked()) {
            assert(!is_sentinel() && "Removing a sentinel node from a list.");
            _prev->_next = _next;
            _next->_prev = _prev;
            _prev = nullptr;
            _next = nullptr;
        }
    }
    virtual void insert_before_self(T *node) noexcept {
        assert(!node->is_linked() && "Inserting a linked node into a list.");
        assert(!is_head_sentinel() && "Inserting before a head sentinel.");
        node->_prev = _prev;
        node->_next = static_cast<T *>(this);
        _prev = node;
    }
    virtual void insert_after_self(T *node) noexcept {
        assert(!node->is_linked() && "Inserting a linked node into a list.");
        assert(!is_tail_sentinel() && "Inserting after a tail sentinel.");
        node->_next = _next;
        node->_prev = static_cast<T *>(this);
        _next = node;
    }
};

template<typename Node>
class IntrusiveList : public detail::IntrusiveListImpl<IntrusiveList<Node>> {

private:
    Node *_head_sentinel = nullptr;
    Node *_tail_sentinel = nullptr;

public:
    explicit IntrusiveList(Pool *pool) noexcept {
        _head_sentinel = pool->create<Node>();
        _tail_sentinel = pool->create<Node>();
        _head_sentinel->_next = _tail_sentinel;
        _tail_sentinel->_prev = _head_sentinel;
    }
    [[nodiscard]] auto head_sentinel() noexcept { return _head_sentinel; }
    [[nodiscard]] auto head_sentinel() const noexcept { return const_cast<const Node *>(_head_sentinel); }
    [[nodiscard]] auto tail_sentinel() noexcept { return _tail_sentinel; }
    [[nodiscard]] auto tail_sentinel() const noexcept { return const_cast<const Node *>(_tail_sentinel); }
};

template<typename Node>
class InlineIntrusiveList : public detail::IntrusiveListImpl<InlineIntrusiveList<Node>> {

private:
    Node _head_sentinel;
    Node _tail_sentinel;

public:
    explicit InlineIntrusiveList(Pool *pool) noexcept
        : _head_sentinel{pool}, _tail_sentinel{pool} {
        _head_sentinel._next = &_tail_sentinel;
        _tail_sentinel._prev = &_head_sentinel;
    }
    [[nodiscard]] auto head_sentinel() noexcept { return &_head_sentinel; }
    [[nodiscard]] auto head_sentinel() const noexcept { return const_cast<const Node *>(&_head_sentinel); }
    [[nodiscard]] auto tail_sentinel() noexcept { return &_tail_sentinel; }
    [[nodiscard]] auto tail_sentinel() const noexcept { return const_cast<const Node *>(&_tail_sentinel); }
};

template<typename Node>
class IntrusiveForwardList {

private:
    template<typename, typename>
    friend class IntrusiveForwardNode;
    Node *_head = nullptr;

private:
    template<typename U>
    class IteratorBase {

    private:
        friend class IntrusiveForwardList;
        U *_current = nullptr;
        explicit IteratorBase(U *current) noexcept : _current{current} {}

    public:
        [[nodiscard]] auto &operator*() const noexcept { return *_current; }
        [[nodiscard]] auto operator->() const noexcept { return _current; }
        [[nodiscard]] auto operator==(luisa::default_sentinel_t) const noexcept { return _current == nullptr; }
        IteratorBase &operator++() noexcept {
            _current = _current->next();
            return *this;
        }
    };

public:
    [[nodiscard]] auto empty() const noexcept { return _head == nullptr; }
    [[nodiscard]] auto head() noexcept { return _head; }
    [[nodiscard]] auto head() const noexcept { return const_cast<const Node *>(_head); }
    [[nodiscard]] auto &front() noexcept { return *head(); }
    [[nodiscard]] const auto &front() const noexcept { return *head(); }
    void insert_front(Node *node) noexcept { node->add_to_list(*this); }

public:
    using Iterator = IteratorBase<Node>;
    using ConstIterator = IteratorBase<const Node>;
    [[nodiscard]] auto begin() noexcept { return Iterator{_head}; }
    [[nodiscard]] auto begin() const noexcept { return ConstIterator{_head}; }
    [[nodiscard]] auto end() const noexcept { return luisa::default_sentinel; }
    [[nodiscard]] auto cbegin() const noexcept { return begin(); }
    [[nodiscard]] auto cend() const noexcept { return end(); }
};

// intrusive node for singly-doubly linked lists
template<typename T, typename Base = PooledObject>
class IntrusiveForwardNode : public Base {

public:
    using Super = IntrusiveForwardNode;
    static_assert(std::is_base_of_v<PooledObject, Base>);

protected:
    using Base::Base;

private:
    template<typename>
    friend class IntrusiveForwardList;

    T *_next = nullptr;      // pointer to the next node
    T **_prev_next = nullptr;// pointer to the next pointer of the previous node

public:
    [[nodiscard]] auto is_linked() const noexcept { return _prev_next != nullptr; }

    virtual void add_to_list(IntrusiveForwardList<T> &list) noexcept {
        assert(_next == nullptr && _prev_next == nullptr && "Adding a linked node to a list.");
        _next = list._head;
        _prev_next = &list._head;
        if (_next != nullptr) {
            _next->_prev_next = &_next;
        }
        list._head = static_cast<T *>(this);
    }

    virtual void remove_self() noexcept {
        if (_prev_next != nullptr) {
            *_prev_next = _next;
            if (_next != nullptr) {
                _next->_prev_next = _prev_next;
            }
            _next = nullptr;
            _prev_next = nullptr;
        }
    }
};

}// namespace luisa::compute::xir
