//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <core/stl.h>
#include <runtime/command.h>

namespace luisa::compute {

class CommandList : concepts::Noncopyable {

public:
    class Iterator {

    private:
        const Command *_command{nullptr};

    public:
        explicit Iterator(const Command *cmd) noexcept : _command{cmd} {}
        decltype(auto) operator++() noexcept {
            _command = _command->next();
            return (*this);
        }
        auto operator++(int) noexcept {
            auto self = *this;
            _command = _command->next();
            return self;
        }
        [[nodiscard]] decltype(auto) operator*() const noexcept { return _command; }
        [[nodiscard]] auto operator==(std::default_sentinel_t) const noexcept { return _command == nullptr; }
    };

private:
    Command *_head{nullptr};
    Command *_tail{nullptr};

private:
    void _recycle() noexcept;

public:
    CommandList() noexcept = default;
    ~CommandList() noexcept;
    CommandList(CommandList &&) noexcept;
    CommandList &operator=(CommandList &&rhs) noexcept;

    void append(Command *cmd) noexcept;
    [[nodiscard]] auto begin() const noexcept { return Iterator{_head}; }
    [[nodiscard]] auto end() const noexcept { return std::default_sentinel; }
    [[nodiscard]] auto empty() const noexcept { return _head == nullptr; }
};

}// namespace luisa::compute
