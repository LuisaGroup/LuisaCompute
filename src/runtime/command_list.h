//
// Created by Mike Smith on 2021/3/18.
//

#pragma once

#include <vector>

#include <functional>
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
        [[nodiscard]] auto operator==(Iterator rhs) const noexcept { return _command == rhs._command; }
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
    [[nodiscard]] auto end() const noexcept { return Iterator{nullptr}; }
    [[nodiscard]] auto empty() const noexcept { return _head == nullptr; }
    [[nodiscard]] auto remove_all() noexcept;
    [[nodiscard]] std::vector<Command *> get_all() const noexcept;
};

}// namespace luisa::compute
