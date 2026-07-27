#include "ut/ut.hpp"

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <luisa/core/binary_io.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

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

[[nodiscard]] int run_cached_kernel(
    const char *program_path, const BinaryIO *binary_io,
    int value, bool enable_cache) noexcept {
    Context context{program_path};
    DeviceConfig config{.binary_io = binary_io};
    auto device = context.create_device("fallback", &config);
    auto output = device.create_buffer<int>(1u);
    Kernel1D kernel = [](
                          BufferVar<int> result,
                          Int parameter) noexcept {
        result->write(0u, parameter * 3 + 1);
    };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = enable_cache});
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, value).dispatch(1u)
           << output.copy_to(&result)
           << synchronize();
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    MemoryBinaryIO binary_io;

    "fallback object cache reuses code across devices and keeps uniforms dynamic"_test =
        [&] {
            auto cold_result = run_cached_kernel(
                program_path, &binary_io, 7, true);
            expect(cold_result == 22);
            expect(binary_io.cache_write_count == 2u);
            auto cold_read_count = binary_io.cache_read_count;
            auto cold_write_count = binary_io.cache_write_count;

            auto hot_result = run_cached_kernel(
                program_path, &binary_io, 11, true);
            expect(hot_result == 34);
            expect(
                binary_io.cache_read_count ==
                cold_read_count + 2u);
            expect(
                binary_io.cache_write_count ==
                cold_write_count);

            auto reads_before_disabled =
                binary_io.cache_read_count;
            auto writes_before_disabled =
                binary_io.cache_write_count;
            auto uncached_result = run_cached_kernel(
                program_path, &binary_io, 13, false);
            expect(uncached_result == 40);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled);
        };
}
