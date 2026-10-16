#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <luisa/core/binary_io.h>
#include <luisa/core/stl/vector.h>

namespace luisa::test {

class MemoryBinaryStream final : public BinaryStream {

private:
    luisa::vector<std::byte> _data;
    size_t _position{};

public:
    explicit MemoryBinaryStream(
        luisa::span<const std::byte> data) noexcept
        : _data{data.begin(), data.end()} {}

    [[nodiscard]] size_t length() const noexcept override {
        return _data.size();
    }

    [[nodiscard]] size_t pos() const noexcept override {
        return _position;
    }

    void read(luisa::span<std::byte> destination) noexcept override {
        if (_position > _data.size() ||
            destination.size() > _data.size() - _position) {
            std::memset(
                destination.data(), 0, destination.size());
            _position = _data.size();
            return;
        }
        std::memcpy(
            destination.data(), _data.data() + _position,
            destination.size());
        _position += destination.size();
    }
};

class MemoryBinaryIO final : public BinaryIO {

private:
    mutable std::unordered_map<
        std::string, std::vector<std::byte>>
        _cache;

public:
    mutable size_t cache_read_count{};
    mutable size_t cache_write_count{};

    void clear_shader_cache() const noexcept override {
        _cache.clear();
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(
        luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(
        luisa::string_view name) const noexcept override {
        cache_read_count++;
        auto iterator = _cache.find(std::string{name});
        if (iterator == _cache.end()) { return nullptr; }
        auto &&data = iterator->second;
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{
                data.data(), data.size()});
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(
        luisa::string_view) const noexcept override {
        return nullptr;
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept override {
        cache_write_count++;
        _cache[std::string{name}] = {
            data.begin(), data.end()};
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }
};

}// namespace luisa::test
