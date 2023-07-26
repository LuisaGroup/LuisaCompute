#pragma once

#include <utility>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/command_list.h>

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
            requires std::is_rvalue_reference_v<T &&> && is_stream_event_v<T>
        Stream &operator<<(T &&t) && noexcept {
            _commit();
            luisa::invoke(std::forward<T>(t), _stream->device(), _stream->handle());
            return *_stream;
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
        requires std::is_rvalue_reference_v<T &&> && is_stream_event_v<T>
    Stream &operator<<(T &&t) noexcept {
        luisa::invoke(std::forward<T>(t), device(), handle());
        return *this;
    }
    Stream &operator<<(CommandList::Commit &&commit) noexcept;
    Stream &operator<<(Synchronize &&) noexcept;
    void synchronize() noexcept { _synchronize(); }
    [[nodiscard]] auto stream_tag() const noexcept { return _stream_tag; }

    // compound commands
    template<typename... T>
    decltype(auto) operator<<(std::tuple<T...> &&args) noexcept {
        // FIXME: Delegate{this} << without a temporary definition may boom GCC
        Delegate delegate{this};
        return std::move(delegate) << std::move(args);
    }
};

[[nodiscard]] constexpr auto commit() noexcept { return Stream::Commit{}; }
[[nodiscard]] constexpr auto synchronize() noexcept { return Stream::Synchronize{}; }

}// namespace luisa::compute

