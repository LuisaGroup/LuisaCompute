#include "ut/ut.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
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

    void reset() const noexcept {
        _cache.clear();
        cache_read_count = 0u;
        cache_write_count = 0u;
    }

    void clear_shader_cache() const noexcept override {
        reset();
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(
        luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(
        luisa::string_view name) const noexcept override {
        ++cache_read_count;
        auto iterator = _cache.find(std::string{name});
        if (iterator == _cache.end()) { return nullptr; }
        auto &data = iterator->second;
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
        ++cache_write_count;
        _cache[std::string{name}] = {
            data.begin(), data.end()};
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view,
        luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    [[nodiscard]] size_t entry_count_with_suffix(
        std::string_view suffix) const noexcept {
        auto count = size_t{0u};
        for (auto &&[name, data] : _cache) {
            static_cast<void>(data);
            count += name.ends_with(suffix) ? 1u : 0u;
        }
        return count;
    }
};

class ScopedEnvironmentVariable {

private:
    std::string _name;
    std::string _old_value;
    bool _had_old_value{};

public:
    ScopedEnvironmentVariable(
        const char *name, const char *value)
        : _name{name} {
        if (auto old_value = std::getenv(name)) {
            _old_value = old_value;
            _had_old_value = true;
        }
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    ~ScopedEnvironmentVariable() noexcept {
#ifdef _WIN32
        _putenv_s(
            _name.c_str(),
            _had_old_value ? _old_value.c_str() : "");
#else
        if (_had_old_value) {
            setenv(
                _name.c_str(), _old_value.c_str(), 1);
        } else {
            unsetenv(_name.c_str());
        }
#endif
    }
};

template<typename Shader>
[[nodiscard]] int execute(
    Device &device, const Shader &shader,
    int input) noexcept {
    auto output = device.create_buffer<int>(1u);
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, input).dispatch(1u)
           << output.copy_to(&result)
           << synchronize();
    return result;
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1 || argv == nullptr ||
        std::string_view{argv[1]} != "vk") {
        LUISA_INFO(
            "Usage: {} vk",
            argc > 0 && argv != nullptr ?
                argv[0] :
                "test_vk_shader_cache");
        return 2;
    }

    ScopedEnvironmentVariable require_native{
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
    MemoryBinaryIO binary_io;
    Context context{argv[0]};
    DeviceConfig config{.binary_io = &binary_io};
    auto device = context.create_device("vk", &config);

    // Device-owned helper shaders may use the same BinaryIO. Begin the
    // contract at the user-compute boundary after device initialization.
    binary_io.reset();

    Kernel1D kernel = [](
                          BufferVar<int> output,
                          Int input) noexcept {
        output.write(0u, input * 3 + 1);
    };
    ShaderOption option{.enable_cache = true};

    "native Vulkan persists SPIR-V and PSO on the cold compile"_test = [&] {
        auto cold = device.compile(kernel, option);
        expect(execute(device, cold, 7) == 22);
        expect(binary_io.entry_count_with_suffix(".spv") == 1u);
        expect(binary_io.entry_count_with_suffix(".vk") == 1u)
            << "the first native compile must persist the driver pipeline "
               "cache instead of deferring it until the second process";
        expect(binary_io.cache_write_count == 2u);

        auto writes_after_cold = binary_io.cache_write_count;
        auto hot = device.compile(kernel, option);
        expect(execute(device, hot, 11) == 34);
        expect(binary_io.cache_write_count == writes_after_cold)
            << "a valid hot SPIR-V/PSO pair must not be rewritten";
        expect(binary_io.entry_count_with_suffix(".spv") == 1u);
        expect(binary_io.entry_count_with_suffix(".vk") == 1u);
    };
}
