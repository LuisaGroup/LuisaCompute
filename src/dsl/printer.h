//
// Created by Mike Smith on 2022/2/13.
//

#pragma once

#include <mutex>
#include <thread>

#include <core/logging.h>
#include <ast/function_builder.h>
#include <runtime/buffer.h>
#include <runtime/event.h>
#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/builtin.h>
#include <dsl/operators.h>

namespace luisa::compute {

class Device;
class Stream;

/// Printer in kernel
class LC_DSL_API Printer {

public:
    struct Item {
        uint size;
        luisa::move_only_function<void(const uint *)> f;
        Item(uint size, luisa::move_only_function<void(const uint *)> f) noexcept
            : size{size}, f{std::move(f)} {}
    };

private:
    Buffer<uint> _buffer;// count & records (desc_id, arg0, arg1, ...)
    luisa::vector<uint> _host_buffer;
    luisa::vector<Item> _items;
    luisa::logger _logger;
    bool _reset_called{false};

private:
    void _log_to_buffer(Expr<uint>, uint) noexcept {}

    template<typename Curr, typename... Other>
    void _log_to_buffer(Expr<uint> offset, uint index, const Curr &curr, const Other &...other) noexcept {
        if constexpr (is_dsl_v<Curr>) {
            index++;
            using T = expr_value_t<Curr>;
            if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, uint>) {
                _buffer.write(offset + index, cast<uint>(curr));
            } else if constexpr (std::is_same_v<T, float>) {
                _buffer.write(offset + index, as<uint>(curr));
            } else {
                static_assert(always_false_v<T>, "unsupported type for printing in kernel.");
            }
        }
        _log_to_buffer(offset, index, other...);
    }

    /// Log in kernel
    template<typename... Args>
    void _log(luisa::log_level level, luisa::string fmt, const Args &...args) noexcept;

public:
    /// Create printer object on device. Will create a buffer in it.
    explicit Printer(Device &device, luisa::string_view name = "device", size_t capacity = 16_mb) noexcept;
    /// Reset the printer. Must be called before any shader dispatch that uses this printer.
    [[nodiscard]] luisa::unique_ptr<Command> reset() noexcept;
    /// Retrieve and print the logs. Will automatically reset the printer for future use.
    [[nodiscard]] std::tuple<luisa::unique_ptr<Command> /* download */,
                             luisa::move_only_function<void()> /* print */,
                             luisa::unique_ptr<Command> /* reset */>
    retrieve() noexcept;

    /// Log in kernel at debug level.
    template<typename... Args>
    void verbose(luisa::string fmt, Args &&...args) noexcept {
        _log(luisa::log_level::debug, std::move(fmt), std::forward<Args>(args)...);
    }
    /// Log in kernel at information level.
    template<typename... Args>
    void info(luisa::string fmt, Args &&...args) noexcept {
        _log(luisa::log_level::info, std::move(fmt), std::forward<Args>(args)...);
    }
    /// Log in kernel at warning level.
    template<typename... Args>
    void warning(luisa::string fmt, Args &&...args) noexcept {
        _log(luisa::log_level::warn, std::move(fmt), std::forward<Args>(args)...);
    }
    /// Log in kernel at error level.
    template<typename... Args>
    void error(luisa::string fmt, Args &&...args) noexcept {
        _log(luisa::log_level::err, std::move(fmt), std::forward<Args>(args)...);
    }
    /// Log in kernel at debug level with dispatch id.
    template<typename... Args>
    void verbose_with_location(luisa::string fmt, Args &&...args) noexcept {
        auto p = dispatch_id();
        verbose(std::move(fmt.append(" [dispatch_id = ({}, {}, {})]")),
                std::forward<Args>(args)..., p.x, p.y, p.z);
    }
    /// Log in kernel at information level with dispatch id.
    template<typename... Args>
    void info_with_location(luisa::string fmt, Args &&...args) noexcept {
        auto p = dispatch_id();
        info(std::move(fmt.append(" [dispatch_id = ({}, {}, {})]")),
             std::forward<Args>(args)..., p.x, p.y, p.z);
    }
    /// Log in kernel at warning level with dispatch id.
    template<typename... Args>
    void warning_with_location(luisa::string fmt, Args &&...args) noexcept {
        auto p = dispatch_id();
        warning(std::move(fmt.append(" [dispatch_id = ({}, {}, {})]")),
                std::forward<Args>(args)..., p.x, p.y, p.z);
    }
    /// Log in kernel at error level with dispatch id.
    template<typename... Args>
    void error_with_location(luisa::string fmt, Args &&...args) noexcept {
        auto p = dispatch_id();
        error(std::move(fmt.append(" [dispatch_id = ({}, {}, {})]")),
              std::forward<Args>(args)..., p.x, p.y, p.z);
    }
};

template<typename... Args>
void Printer::_log(luisa::log_level level, luisa::string fmt, const Args &...args) noexcept {
    auto count = (1u /* desc_id */ + ... + static_cast<uint>(is_dsl_v<Args>));
    auto size = static_cast<uint>(_buffer.size() - 1u);
    auto offset = _buffer.atomic(size).fetch_add(count);
    auto item = static_cast<uint>(_items.size());
    if_(offset < size, [&] { _buffer.write(offset, item); });
    if_(offset + count <= size, [&] { _log_to_buffer(offset, 0u, args...); });
    // create decoder...
    auto counter = 0u;
    auto convert = [&counter]<typename T>(const T &arg) noexcept {
        if constexpr (is_dsl_v<T>) {
            return ++counter;
        } else if constexpr (requires { luisa::string_view{arg}; }) {
            return luisa::string{arg};
        } else {
            static_assert(std::is_trivial_v<std::remove_cvref_t<T>>);
            return arg;
        }
    };
    auto decode = [this, level, f = std::move(fmt), args = std::tuple{convert(args)...}](const uint *data) noexcept {
        auto decode_arg = [&args, data]<size_t i>() noexcept {
            using Arg = std::tuple_element_t<i, std::tuple<Args...>>;
            if constexpr (is_dsl_v<Arg>) {
                auto raw = data[std::get<i>(args)];
                using T = expr_value_t<Arg>;
                if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, uint>) {
                    return static_cast<T>(raw);
                } else {
                    return luisa::bit_cast<T>(raw);
                }
            } else {
                return std::get<i>(args);
            }
        };
        auto do_print = [&]<size_t... i>(std::index_sequence<i...>) noexcept {
            _logger.log(level, f, decode_arg.template operator()<i>()...);
        };
        do_print(std::index_sequence_for<Args...>{});
    };
    _items.emplace_back(count, decode);
}

}// namespace luisa::compute
