// Cold-start benchmark for parallel Vulkan shader compilation (plan P5.1).
//
// This whole benchmark is a measurement tool of the Vulkan optimization work and
// is therefore gated by the private compile-time macro LUISA_VULKAN_PROFILE_SUBMIT
// (default 0): with the macro off the file compiles to a `main` that only prints
// that requirement, so a normal build carries neither this benchmark nor a silent
// no-op that looks like it ran. The macro is a source-level opt-in - deliberately
// not an XMake/CMake option and not an environment variable; see
// src/tests/CMakeLists.txt and src/tests/xmake.lua.
//
// What it measures
// ----------------
// The wall time of creating N *distinct* compute shaders from scratch through
// `VulkanShaderCompileExt::compile_shaders()` (`--workers W`) or through a serial
// loop of `Device::compile` (`--workers 0`), plus the same N creations served
// entirely from a warm shader cache (`--warm`).
//
// Every configuration gets a **cold, isolated** `BinaryIO`: a fresh in-memory
// implementation per round, so no artifact of a previous round can be read and no
// file on disk can leak in. The shaders stay alive for the whole measured window
// (destruction is not part of the number) and each one is verified afterwards by
// dispatching it once and checking its numeric result, which is also what proves
// that the handles a batch returns are usable shaders.
//
// Artifact identity
// -----------------
// `--manifest <path>` writes one line per persisted artifact -
// `<name> <bytes> <md5>` - sorted by name. Running the benchmark with `--workers 0`
// and with `--workers N` into two manifests and comparing them is the check that a
// batch produces byte-identical cache artifacts to a serial loop (plan P5.1
// requirement: "generated cache artifacts must match serial compilation").
//
// Usage
// -----
//   benchmark_vk_shader_compile vk [--shaders 216] [--workers 0|1|N] [--rounds 3]
// [--warm] [--no-verify] [--manifest FILE] [--quiet]
//
// `LUISA_VULKAN_PROFILE_COMPILATION=1` additionally prints the backend's own phase
// timings for every compile; it is deliberately *not* set by this program, because
// it changes what the measured window does (it formats and logs).
// The gate of the instrument (same default as src/backends/vk/submit_profile.h).
#if !defined(LUISA_VULKAN_PROFILE_SUBMIT)
#define LUISA_VULKAN_PROFILE_SUBMIT 0
#endif

#if LUISA_VULKAN_PROFILE_SUBMIT

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <luisa/backends/ext/vk_shader_compile_ext.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/vstl/md5.h>

#include "ut/ut.hpp"
#include "batch_shader_dispatch.h"
#include "test_device.h"

using namespace luisa;
using namespace luisa::compute;

namespace {

// ------------------------------------------------------------ BinaryIO ---
class MemoryBinaryStream final : public BinaryStream {
private:
    luisa::vector<std::byte> _data;
    size_t _position{};
public:
    explicit MemoryBinaryStream(luisa::span<const std::byte> data) noexcept
        : _data{data.begin(), data.end()} {}
    [[nodiscard]] size_t length() const noexcept override { return _data.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _position; }
    void read(luisa::span<std::byte> destination) noexcept override {
        if (_position > _data.size() ||
            destination.size() > _data.size() - _position) {
            std::memset(destination.data(), 0, destination.size());
            _position = _data.size();
            return;
        }
        std::memcpy(destination.data(), _data.data() + _position,
                    destination.size());
        _position += destination.size();
    }
    ~MemoryBinaryStream() noexcept override = default;
};

class MemoryBinaryIO final : public BinaryIO {
private:
    mutable std::unordered_map<std::string, std::vector<std::byte>> _cache;
public:
    mutable size_t cache_reads{};
    mutable size_t cache_writes{};
    void clear_shader_cache() const noexcept override { _cache.clear(); }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view name) const noexcept override {
        auto iterator = _cache.find(std::string{name});
        if (iterator == _cache.end()) { return nullptr; }
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{iterator->second.data(),
                                         iterator->second.size()});
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view name) const noexcept override {
        ++cache_reads;
        return read_shader_bytecode(name);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(luisa::string_view) const noexcept override {
        return nullptr;
    }
    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        ++cache_writes;
        _cache[std::string{name}] = {data.begin(), data.end()};
        return {};
    }
    luisa::filesystem::path write_shader_cache(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        return write_shader_bytecode(name, data);
    }
    luisa::filesystem::path write_internal_shader(
        luisa::string_view, luisa::span<const std::byte>) const noexcept override {
        return {};
    }
    [[nodiscard]] size_t entry_count() const noexcept { return _cache.size(); }
    [[nodiscard]] std::vector<std::string> sorted_names() const {
        std::vector<std::string> names;
        names.reserve(_cache.size());
        for (auto &&entry : _cache) { names.emplace_back(entry.first); }
        std::sort(names.begin(), names.end());
        return names;
    }
    [[nodiscard]] std::vector<std::byte> const *find(std::string_view name) const {
        auto iterator = _cache.find(std::string{name});
        return iterator == _cache.end() ? nullptr : &iterator->second;
    }
};

// -------------------------------------------------------------- kernels ---
struct KernelEntry {
    // The DSL kernel owns the AST; `function` is the handle a backend call takes.
    Kernel1D<Buffer<float>> kernel;
    Function function;
    uint32_t constant{0u};
    uint32_t shape{0u};
};

// N distinct kernels. `constant` is baked into the AST, so each one has its own
// cache key and its own artifact; the four shapes keep the generated code from
// being one identical body N times.
luisa::vector<KernelEntry> make_kernels(uint32_t count) {
    auto make_one = [](uint32_t k, uint32_t shape) {
        auto constant = static_cast<float>(k) + 0.5f;
        switch (shape) {
            case 0u:
                return Kernel1D<Buffer<float>>{[constant, k](BufferFloat buffer) noexcept {
                    auto x = cast<float>(dispatch_id().x);
                    buffer.write(dispatch_id().x,
                                 sin(x * constant) + cast<float>(k));
                }};
            case 1u:
                return Kernel1D<Buffer<float>>{[constant, k](BufferFloat buffer) noexcept {
                    auto x = cast<float>(dispatch_id().x) * constant;
                    auto sum = def<float>(0.0f);
                    for (auto j = 0u; j < 4u; j++) {
                        sum += sqrt(x + cast<float>(j) + 1.0f);
                    }
                    buffer.write(dispatch_id().x, sum + cast<float>(k * 3u));
                }};
            case 2u:
                return Kernel1D<Buffer<float>>{[constant, k](BufferFloat buffer) noexcept {
                    auto x = cast<float>(dispatch_id().x);
                    auto v = select(x, x * 0.5f, x > constant);
                    buffer.write(dispatch_id().x,
                                 v * exp2(1.0f) + cast<float>(k));
                }};
            default:
                return Kernel1D<Buffer<float>>{[constant, k](BufferFloat buffer) noexcept {
                    auto x = cast<float>(dispatch_id().x);
                    auto bits = cast<uint>(x) * (k + 7u);
                    buffer.write(dispatch_id().x,
                                 cast<float>(bits ^ (k * 13u)) + constant);
                }};
        }
    };
    luisa::vector<KernelEntry> kernels;
    kernels.reserve(count);
    for (auto i = 0u; i < count; i++) {
        auto k = i + 1u;
        auto shape = i % 4u;
        auto kernel = make_one(k, shape);
        kernels.emplace_back(KernelEntry{
            .kernel = kernel,
            .function = kernel.function()->function(),
            .constant = k,
            .shape = shape});
    }
    return kernels;
}

// Expected value of kernel `i` at dispatch index 0 (the same arithmetic as the
// kernel body, in host code). Only index 0..3 are checked, which is enough to
// catch a cross-wired handle: two different handles cannot produce the same four
// values.
float expected_value(uint32_t k, uint32_t shape, uint32_t index) noexcept {
    auto constant = static_cast<float>(k) + 0.5f;
    auto x = static_cast<float>(index);
    switch (shape) {
        case 0u: return std::sin(x * constant) + static_cast<float>(k);
        case 1u: {
            // The kernel scales x by its constant before the loop.
            auto scaled = x * constant;
            auto sum = 0.0f;
            for (auto j = 0u; j < 4u; j++) {
                sum += std::sqrt(scaled + static_cast<float>(j) + 1.0f);
            }
            return sum + static_cast<float>(k * 3u);
        }
        case 2u: {
            auto v = x > constant ? x * 0.5f : x;
            return v * 2.0f + static_cast<float>(k);
        }
        default: {
            auto bits = static_cast<uint32_t>(x) * (k + 7u);
            return static_cast<float>(bits ^ (k * 13u)) + constant;
        }
    }
}

struct Options {
    uint32_t shaders{216u};
    // Sentinel: `--workers auto` lets the extension pick its default count.
    static constexpr uint32_t kAutoWorkers = ~0u;
    uint32_t workers{0u};
    uint32_t rounds{3u};
    bool warm{false};
    bool verify{true};
    bool quiet{false};
    luisa::string manifest;
    luisa::string artifact_dir;
};

struct Measurement {
    double milliseconds{0.0};
    size_t artifacts{0u};
};

}// namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <backend> [--shaders N] [--workers W] [--rounds R] "
                     "[--warm] [--manifest FILE] [--no-verify] [--quiet]\n",
                     argv[0]);
        return 2;
    }
    Options options;
    for (auto i = 2; i < argc; i++) {
        std::string_view argument{argv[i]};
        auto next = [&]() -> uint32_t {
            return i + 1 < argc ? static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)) : 0u;
        };
        if (argument == "--shaders") {
            options.shaders = std::max(1u, next());
        } else if (argument == "--workers") {
            if (i + 1 < argc && std::string_view{argv[i + 1]} == "auto") {
                ++i;
                options.workers = Options::kAutoWorkers;
            } else {
                options.workers = next();
            }
        } else if (argument == "--rounds") {
            options.rounds = std::max(1u, next());
        } else if (argument == "--warm") {
            options.warm = true;
        } else if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--manifest") {
            if (i + 1 < argc) { options.manifest = argv[++i]; }
        } else if (argument == "--artifact-dir") {
            if (i + 1 < argc) { options.artifact_dir = argv[++i]; }
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    // The benchmark creates one device per round, each with its own fresh in-memory
    // BinaryIO. `create_device_from_ut` is what validates the backend argument; the
    // Vulkan backend allows one live device at a time, so every device is destroyed
    // at the end of its round.
    auto kernels = make_kernels(options.shaders);
    std::optional<Measurement> cold;
    std::optional<Measurement> warm;
    std::vector<double> cold_ms;
    std::vector<double> warm_ms;
    // Kept alive past the loop so the manifest of the last round can be written.
    std::unique_ptr<MemoryBinaryIO> binary_io;
    for (auto round = 0u; round < options.rounds; round++) {
        binary_io = std::make_unique<MemoryBinaryIO>();
        {
            DeviceConfig config;
            config.binary_io = binary_io.get();
            auto device_context = luisa::test::create_device_from_ut(
                argc, argv, &config);
            if (!device_context) { return 0; }
            auto &device = device_context->device;
            auto stream = device.create_stream();
            auto dispatch_size = 4u;
            luisa::vector<float> host(dispatch_size);
            auto buffer = device.create_buffer<float>(dispatch_size);
            auto verify_one = [&](uint64_t handle, size_t arg_count,
                                  size_t uniform_size, uint32_t index) noexcept {
                if (!options.verify) { return; }
                luisa::test::dispatch_one_buffer_shader(
                    stream, handle, arg_count, uniform_size, buffer,
                    dispatch_size, host);
                for (auto element = 0u; element < dispatch_size; element++) {
                    auto e = expected_value(kernels[index].constant,
                                            kernels[index].shape, element);
                    if (std::abs(host[element] - e) > 1e-3f) {
                        LUISA_WARNING(
                            "kernel {} element {}: got {}, expected {}",
                            index, element, host[element], e);
                        LUISA_ERROR("shader handle verification failed");
                    }
                }
            };
            Clock clock;
            if (options.workers == 0u) {
                // Serial control: a plain `Device::compile` loop.
                luisa::vector<Shader1D<Buffer<float>>> shaders;
                shaders.reserve(options.shaders);
                clock.tic();
                for (auto i = 0u; i < options.shaders; i++) {
                    ShaderOption option;
                    shaders.emplace_back(device.compile(
                        Kernel1D<Buffer<float>>{
                            kernels[i].function.shared_builder()},
                        option));
                }
                auto ms = clock.toc();
                for (auto i = 0u; i < options.shaders; i++) {
                    verify_one(shaders[i].handle(),
                               luisa::test::batch_shader_arg_count(kernels[i].function),
                               luisa::test::batch_shader_uniform_size(kernels[i].function),
                               i);
                }
                cold_ms.emplace_back(ms);
                cold = Measurement{.milliseconds = ms,
                                   .artifacts = binary_io->entry_count()};
            } else {
                auto *ext = device.extension<VulkanShaderCompileExt>();
                if (ext == nullptr) {
                    LUISA_ERROR("This backend does not provide VulkanShaderCompileExt.");
                }
                luisa::vector<VulkanShaderCompileRequest> requests;
                requests.reserve(options.shaders);
                for (auto &kernel : kernels) {
                    requests.emplace_back(VulkanShaderCompileRequest{
                        .option = ShaderOption{}, .kernel = kernel.function});
                }
                luisa::vector<VulkanShaderCompileResult> results;
                clock.tic();
                auto status = ext->compile_shaders(
                    luisa::span<const VulkanShaderCompileRequest>{requests},
                    results,
                    options.workers == Options::kAutoWorkers ? 0u : options.workers);
                auto ms = clock.toc();
                if (status != VulkanShaderCompileStatus::SUCCESS) {
                    LUISA_ERROR("Batch compilation answered '{}': {}",
                                vulkan_shader_compile_status_name(status),
                                results.empty() ? luisa::string_view{} : luisa::string_view{results.front().message});
                }
                if (results.size() != options.shaders) {
                    LUISA_ERROR("Batch returned {} results for {} requests.",
                                results.size(), options.shaders);
                }
                for (auto i = 0u; i < options.shaders; i++) {
                    if (results[i].status != VulkanShaderCompileStatus::SUCCESS ||
                        results[i].shader.handle == 0u) {
                        LUISA_ERROR("Request {} failed: {}", i,
                                    luisa::string_view{results[i].message});
                    }
                    verify_one(results[i].shader.handle,
                               luisa::test::batch_shader_arg_count(kernels[i].function),
                               luisa::test::batch_shader_uniform_size(kernels[i].function),
                               i);
                }
                cold_ms.emplace_back(ms);
                cold = Measurement{.milliseconds = ms,
                                   .artifacts = binary_io->entry_count()};
                if (!options.quiet) {
                    auto stats = ext->stats();
                    LUISA_INFO(
                        "pool: workers={} batches={} requests={} parallel={} "
                        "serial={} cache_hits={} fallbacks={} order_waits={} "
                        "order_wait_ns={}",
                        stats.worker_count, stats.batches, stats.requests,
                        stats.parallel_prepares, stats.serial_prepares,
                        stats.cache_hits, stats.serial_fallbacks,
                        stats.publish_order_waits, stats.publish_order_wait_ns);
                }
            }
            if (options.warm) {
                // The same device and the same BinaryIO again: every compile is a
                // cache hit, so this is the "warm create" number the acceptance rule
                // compares (it must not regress materially).
                luisa::vector<Shader1D<Buffer<float>>> shaders;
                shaders.reserve(options.shaders);
                Clock warm_clock;
                warm_clock.tic();
                for (auto i = 0u; i < options.shaders; i++) {
                    shaders.emplace_back(device.compile(
                        Kernel1D<Buffer<float>>{
                            kernels[i].function.shared_builder()}));
                }
                warm_ms.emplace_back(warm_clock.toc());
            }
            static_cast<void>(device);
        }
    }
    // One artifact per distinct kernel: the batch wrote the same cache entries a
    // serial loop writes.
    if (!options.manifest.empty() && binary_io != nullptr) {
        FILE *file = std::fopen(options.manifest.c_str(), "wb");
        if (file == nullptr) {
            LUISA_ERROR("Could not open '{}' for writing.", options.manifest);
        }
        for (auto &name : binary_io->sorted_names()) {
            auto *data = binary_io->find(name);
            auto md5 = vstd::MD5(vstd::span<const uint8_t>{
                reinterpret_cast<const uint8_t *>(data->data()), data->size()});
            auto line = std::string{name} + " " + std::to_string(data->size()) +
                        " " + std::string{md5.to_string(false)} + "\n";
            std::fwrite(line.data(), 1u, line.size(), file);
        }
        std::fclose(file);
    }
    // Raw artifacts of the last round, one file per entry. This is what a
    // byte-level comparison of two configurations reads (the manifest only carries
    // name/size/md5).
    if (!options.artifact_dir.empty() && binary_io != nullptr) {
        auto directory = luisa::filesystem::path{
            luisa::string_view{options.artifact_dir}};
        std::error_code error;
        luisa::filesystem::create_directories(directory, error);
        for (auto &name : binary_io->sorted_names()) {
            auto *data = binary_io->find(name);
            auto path = directory / luisa::filesystem::path{luisa::string_view{name}};
            FILE *file = std::fopen(path.string().c_str(), "wb");
            if (file == nullptr) {
                LUISA_ERROR("Could not open '{}' for writing.", path.string());
            }
            std::fwrite(data->data(), 1u, data->size(), file);
            std::fclose(file);
        }
    }
    auto median = [](std::vector<double> values) {
        if (values.empty()) { return 0.0; }
        std::sort(values.begin(), values.end());
        return values[values.size() / 2u];
    };
    auto cold_median = median(cold_ms);
    auto warm_median = median(warm_ms);
    std::printf("RESULT shaders=%u workers=%u rounds=%u artifacts=%zu "
                "cold_ms=%.3f cold_per_shader_ms=%.3f warm_ms=%.3f "
                "warm_per_shader_ms=%.3f\n",
                options.shaders, options.workers, options.rounds,
                cold.has_value() ? cold->artifacts : 0u,
                cold_median, cold_median / static_cast<double>(options.shaders),
                warm_median, warm_median / static_cast<double>(options.shaders));
    for (auto value : cold_ms) { std::printf("  cold round: %.3f ms\n", value); }
    for (auto value : warm_ms) { std::printf("  warm round: %.3f ms\n", value); }
    return 0;
}


#else// !LUISA_VULKAN_PROFILE_SUBMIT

// Nothing was measured. This benchmark belongs to the gated profiling/optimization
// tooling (see src/backends/vk/submit_profile.h), so a default build has no
// benchmark to run and must not look as if it had run one.
#include <cstdio>

int main() {
    std::fprintf(stderr,
                 "benchmark_vk_shader_compile: nothing was measured. This binary "
                 "needs the compile-time macro LUISA_VULKAN_PROFILE_SUBMIT=1 in the "
                 "backend library *and* in this benchmark; it is a private "
                 "source-level opt-in, deliberately not an XMake/CMake option and not "
                 "an environment variable.\n");
    return 0;
}

#endif// LUISA_VULKAN_PROFILE_SUBMIT
