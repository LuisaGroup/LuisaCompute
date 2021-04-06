//
// Created by Mike Smith on 2021/3/5.
//

#pragma once

#include <string>
#include <ast/function.h>

namespace luisa::compute {

class Codegen {

public:
    class Scratch {

    private:
        std::string _buffer;

    public:
        Scratch() noexcept;
        Scratch &operator<<(bool x) noexcept;
        Scratch &operator<<(float x) noexcept;
        Scratch &operator<<(int x) noexcept;
        Scratch &operator<<(uint x) noexcept;
        Scratch &operator<<(size_t x) noexcept;
        Scratch &operator<<(std::string_view s) noexcept;
        Scratch &operator<<(const char *s) noexcept;
        Scratch &operator<<(const std::string &s) noexcept;
        [[nodiscard]] std::string_view view() const noexcept;
        [[nodiscard]] const char *c_str() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        void pop_back() noexcept;
        void clear() noexcept;
        [[nodiscard]] char back() const noexcept;
    };

protected:
    Scratch &_scratch;

public:
    explicit Codegen(Scratch &scratch) noexcept : _scratch{scratch} {}
    virtual void emit(Function f) = 0;
};

}// namespace luisa::compute
