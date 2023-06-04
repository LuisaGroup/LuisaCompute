//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <utility>

#include <core/spin_mutex.h>
#include <runtime/rhi/resource.h>
#include <runtime/command_list.h>
#include <runtime/rhi/stream_tag.h>

namespace luisa::compute {
class LC_RUNTIME_API Stream final : public Resource {

public:
    struct Commit {};
    struct Synchronize {};

    class LC_RUNTIME_API Delegate {

    private:
        Stream *_stream;
        CommandList _command_list;

    private:
        void _commit() noexcept;

    private:
        friend class Stream;
        explicit Delegate(Stream *s) noexcept;
        Delegate(Delegate &&) noexcept;

    public:
        ~Delegate() noexcept;
        Delegate(const Delegate &) noexcept = delete;
        Delegate &operator=(Delegate &&) noexcept = delete;
        Delegate &operator=(const Delegate &) noexcept = delete;
        Delegate operator<<(luisa::unique_ptr<Command> &&cmd) && noexcept;
        Delegate operator<<(luisa::move_only_function<void()> &&f) && noexcept;
        template<typename T>
            requires(StreamEvent<std::remove_cvref_t<T>>::value)
        Stream &operator<<(T &&t) && noexcept {
            _commit();
            return *_stream << std::forward<T>(t);
        }
        Stream &operator<<(CommandList::Commit &&commit) && noexcept;
        Stream &operator<<(Synchronize &&) && noexcept;
        Stream &operator<<(Commit &&) && noexcept;

        // compound commands
        template<typename... T>
        decltype(auto) operator<<(std::tuple<T...> args) && noexcept {
            auto encode = [&]<size_t... i>(std::index_sequence<i...>) noexcept -> decltype(auto) {
                return (std::move(*this) << ... << std::move(std::get<i>(args)));
            };
            return encode(std::index_sequence_for<T...>{});
        }
    };

private:
    friend class Device;
    friend class DStorageExt;
    StreamTag _stream_tag{};

private:
    explicit Stream(DeviceInterface *device, StreamTag stream_tag) noexcept;
    explicit Stream(DeviceInterface *device, StreamTag stream_tag, const ResourceCreationInfo &stream_handle) noexcept;
    void _dispatch(CommandList &&command_buffer) noexcept;
    void _synchronize() noexcept;

public:
    Stream() noexcept = default;
    ~Stream() noexcept override;
    Stream(Stream &&) noexcept = default;
    Stream(Stream const &) noexcept = delete;
    Stream &operator=(Stream &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Stream &operator=(Stream const &) noexcept = delete;
    using Resource::operator bool;
    Delegate operator<<(luisa::unique_ptr<Command> &&cmd) noexcept;
    Delegate operator<<(luisa::move_only_function<void()> &&f) noexcept;
    template<typename T>
        requires(StreamEvent<std::remove_cvref_t<T>>::value)
    Stream &operator<<(T &&t) noexcept {
        StreamEvent<std::remove_cvref_t<T>>::execute(device(), handle(), std::forward<T>(t));
        return *this;
    }
    Stream &operator<<(CommandList::Commit &&commit) noexcept;
    Stream &operator<<(Synchronize &&) noexcept;
    void synchronize() noexcept { _synchronize(); }
    [[nodiscard]] auto stream_tag() const noexcept { return _stream_tag; }

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> &&args) noexcept {
        // No Delegate{this}<< here, may boom GCC
        Delegate delegate{this};
        return std::move(delegate) << std::move(args);
    }
};

[[nodiscard]] constexpr auto commit() noexcept { return Stream::Commit{}; }
[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute
