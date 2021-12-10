//
// Created by Mike on 2021/12/10.
//

#pragma once

#include <cstddef>
#include <core/allocator.h>

namespace luisa {

class FirstFit {

public:
    class Node {

    private:
        Node *_next{nullptr};
        size_t _offset{0u};
        size_t _size{0u};

    private:
        friend class FirstFit;

    public:
        Node() noexcept;
        Node(Node &&) noexcept = delete;
        Node(const Node &) noexcept = delete;
        Node &operator=(Node &&) noexcept = delete;
        Node &operator=(const Node &) noexcept = delete;
        [[nodiscard]] auto offset() const noexcept { return _offset; }
        [[nodiscard]] auto size() const noexcept { return _size; }
    };

private:
    Node _free_list;
    size_t _alignment;

private:
    void _destroy() noexcept;

public:
    explicit FirstFit(size_t size, size_t alignment = 64u) noexcept;
    ~FirstFit() noexcept;
    FirstFit(FirstFit &&) noexcept;
    FirstFit(const FirstFit &) noexcept = delete;
    FirstFit &operator=(FirstFit &&) noexcept;
    FirstFit &operator=(const FirstFit &) noexcept = delete;
    [[nodiscard]] Node *allocate(size_t size) noexcept;
    void free(Node *node) noexcept;
    [[nodiscard]] auto size() const noexcept { return _free_list._size; }
    [[nodiscard]] auto alignment() const noexcept { return _alignment; }
    [[nodiscard]] luisa::string dump_free_list() const noexcept;
};

}
