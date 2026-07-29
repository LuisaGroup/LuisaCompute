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
    mutable std::string _last_written_name;

public:
    mutable size_t cache_read_count{};
    mutable size_t cache_write_count{};

    void clear_shader_cache() const noexcept override {
        _cache.clear();
        _last_written_name.clear();
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
        _last_written_name = name;
        _cache[std::string{name}] = {
            data.begin(), data.end()};
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    [[nodiscard]] size_t cache_entry_count() const noexcept {
        return _cache.size();
    }

    void corrupt_last_written_entry() const noexcept {
        auto iterator = _cache.find(_last_written_name);
        if (iterator != _cache.end() &&
            !iterator->second.empty()) {
            iterator->second.back() ^= std::byte{0xffu};
        }
    }
};

[[nodiscard]] int run_cached_kernel(
    const char *program_path,
    const BinaryIO *binary_io,
    int value,
    bool enable_cache,
    bool enable_fast_math) noexcept {
    Context context{program_path};
    DeviceConfig config{.binary_io = binary_io};
    auto device = context.create_device("hip", &config);
    auto output = device.create_buffer<int>(1u);
    Kernel1D kernel = [](
                          BufferVar<int> result,
                          Int parameter) noexcept {
        result->write(0u, parameter * 3 + 1);
    };
    auto shader = device.compile(
        kernel,
        ShaderOption{
            .enable_cache = enable_cache,
            .enable_fast_math = enable_fast_math});
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, value).dispatch(1u)
           << output.copy_to(&result)
           << synchronize();
    return result;
}

[[nodiscard]] int run_cached_bound_kernel(
    const char *program_path,
    const BinaryIO *binary_io,
    int value) noexcept {
    Context context{program_path};
    DeviceConfig config{.binary_io = binary_io};
    auto device = context.create_device("hip", &config);
    auto input = device.create_buffer<int>(1u);
    auto output = device.create_buffer<int>(1u);
    Kernel1D kernel = [&input](
                          BufferVar<int> result) noexcept {
        result->write(0u, input->read(0u) * 5 + 2);
    };
    auto shader = device.compile(
        kernel,
        ShaderOption{.enable_cache = true});
    auto stream = device.create_stream();
    auto result = 0;
    stream << input.copy_from(&value)
           << shader(output).dispatch(1u)
           << output.copy_to(&result)
           << synchronize();
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    auto program_path =
        argc > 0 && argv != nullptr ? argv[0] : "";
    MemoryBinaryIO binary_io;

    "HIP shader cache is reusable, option-safe, and corruption-tolerant"_test =
        [&] {
            auto cold_result = run_cached_kernel(
                program_path, &binary_io, 7, true, true);
            expect(cold_result == 22);
            expect(binary_io.cache_read_count == 1u);
            expect(binary_io.cache_write_count == 1u);
            expect(binary_io.cache_entry_count() == 1u);

            auto hot_result = run_cached_kernel(
                program_path, &binary_io, 11, true, true);
            expect(hot_result == 34);
            expect(binary_io.cache_read_count == 2u);
            expect(binary_io.cache_write_count == 1u);
            expect(binary_io.cache_entry_count() == 1u);

            auto reads_before_disabled =
                binary_io.cache_read_count;
            auto writes_before_disabled =
                binary_io.cache_write_count;
            auto uncached_result = run_cached_kernel(
                program_path, &binary_io, 13, false, true);
            expect(uncached_result == 40);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled);

            auto strict_result = run_cached_kernel(
                program_path, &binary_io, 17, true, false);
            expect(strict_result == 52);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled + 1u);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled + 1u);
            expect(binary_io.cache_entry_count() == 2u);

            binary_io.corrupt_last_written_entry();
            auto repaired_result = run_cached_kernel(
                program_path, &binary_io, 19, true, false);
            expect(repaired_result == 58);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled + 2u);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled + 2u);
            expect(binary_io.cache_entry_count() == 2u);

            auto bound_cold_result =
                run_cached_bound_kernel(
                    program_path, &binary_io, 23);
            expect(bound_cold_result == 117);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled + 3u);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled + 3u);
            expect(binary_io.cache_entry_count() == 3u);

            // The cached module is reusable, but the bound resource handle
            // belongs to the current device and must be reconstructed from
            // the current Function rather than persisted in the cache.
            auto bound_hot_result =
                run_cached_bound_kernel(
                    program_path, &binary_io, 29);
            expect(bound_hot_result == 147);
            expect(
                binary_io.cache_read_count ==
                reads_before_disabled + 4u);
            expect(
                binary_io.cache_write_count ==
                writes_before_disabled + 3u);
            expect(binary_io.cache_entry_count() == 3u);
        };
}
