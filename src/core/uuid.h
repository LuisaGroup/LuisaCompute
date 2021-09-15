//
// Created by Mike Smith on 2021/8/22.
//

#pragma once

#include <span>
#include <uuid.h>

#include <core/concepts.h>
#include <core/logging.h>
#include <core/hash.h>

namespace luisa {

using uuids::operator<<;

// clang-format off
class alignas(16) uuid : public uuids::uuid {

private:
    using uuids::uuid::uuid;
    explicit uuid(uuids::uuid uuid) noexcept : uuids::uuid{uuid} {}

public:
    uuid() noexcept = default;
    uuid(uuid &&) noexcept = default;
    uuid(const uuid &) noexcept = default;
    uuid &operator=(uuid &&) noexcept = default;
    uuid &operator=(const uuid &) noexcept = default;
    [[nodiscard]] explicit operator bool() const noexcept { return !is_nil(); }
    [[nodiscard]] auto view() const noexcept { return as_bytes(); }
    [[nodiscard]] auto hash() const noexcept { return hash64(view()); }
    [[nodiscard]] auto string(bool capitalized = false) const noexcept {
        auto s = uuids::to_string(*this);
        if (capitalized) {
            for (auto &&c : s) { c = static_cast<char>(std::toupper(c)); }
        }
        return s;
    }
    [[nodiscard]] auto string_without_dash(bool capitalized = false) const noexcept {
        auto s = this->string(capitalized);
        s.erase(std::remove(s.begin(), s.end(), '-'), s.end());
        return s;
    }
    [[nodiscard]] static auto generate() noexcept {
        return uuid{uuids::uuid_system_generator{}()};
    }
    template<typename S>
    [[nodiscard]] static auto from_string(S &&s) noexcept {
        std::string_view sv{std::forward<S>(s)};
        auto u = uuids::uuid::from_string(sv);
        if (!u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("Invalid u string: {}.", sv);
        }
        return uuid{*u};
    }
    [[nodiscard]] static auto from_hash64(uint64_t h) noexcept {
        std::array<uint64_t, 2u> src{h, hash64(h)};
        uuid u;
        std::memcpy(&u, &src, sizeof(src));
        return u;
    }
    template<typename T>
    [[nodiscard]] static auto from(T &&data) noexcept {
        if constexpr (concepts::string_viewable<T>) {
            return from_string(std::forward<T>(data));
        } else if constexpr (std::is_same_v<std::remove_cvref_t<T>, uint64_t>) {
            return from_hash64(data);
        } else {
            return uuid{std::forward<T>(data)};
        }
    }
};
// clang-format on

static_assert(sizeof(uuid) == 16u);

}// namespace luisa
