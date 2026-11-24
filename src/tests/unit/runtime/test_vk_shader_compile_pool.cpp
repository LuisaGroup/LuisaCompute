// Test for ordered parallel Vulkan shader compilation (plan P5.1).
// This test covers:
// - the serial path is unchanged: a device that never batches creates no worker
//   thread, answers `vk_shader_compile_pool_state=not_created`, and `Device::compile`
//   keeps working before and after a pool shutdown
// - a batch publishes in request order: `results[i]` is the shader for
//   `requests[i]`, every returned handle is a working shader of this device, and
//   every numeric dispatch result is correct
// - worker counts 1 and >= 2 produce the same shaders and the same artifacts
// - the artifacts a batch writes are byte-identical to the ones a serial compile
//   writes (same names, same sizes, same bytes), which is the cache contract the
//   phase must preserve
// - the compatibility HLSL-to-SPIR-V route (a printing kernel) is reported with
//   `serial_fallback` and still produces a valid shader
// - `compile_only` requests persist their artifact and create no shader object
// - ordinary `Device::create_shader` calls from several threads are safe and
//   ordered: every handle works, and no batch is involved
// - `shutdown_pool()` while a batch is in flight drains that batch (all handles
// valid) and rejects every later batch with `POOL_SHUTDOWN` without creating
// anything
// - the executor is selected on the *device-constructing* thread and never
// re-selected: a device created with no marl scheduler bound reports
// `vk_shader_compile_executor=std_thread` and owns its worker threads, while a
// device created with a scheduler bound reports `fiber` and runs preparation as
// marl tasks on that captured scheduler (no thread of its own). Both executors
// publish in request order, count the same things and shut down the same way
// - a batch submitted *from a scheduler fiber* completes: the publisher waits for
// its lanes through marl events, so it suspends its fiber instead of blocking the
// OS worker under it, and a second batch contending for the batch lock from a fiber
// does not starve the first batch's lanes
// - a batch started from one of the pool's own preparation lanes - through the public
// `BinaryIO` shader-cache callback, which is where a consumer's nested compile really
// happens - is compiled serially on that lane and reports `FELL_BACK_TO_SERIAL`,
// while the outer batch still prepares on a lane, publishes in request order, counts
// both prepares, and both shaders work
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ut/ut.hpp"
#include "batch_shader_dispatch.h"
#include "test_device.h"

#include <luisa/backends/ext/vk_shader_compile_ext.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/fiber.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

int g_argc = 0;
char **g_argv = nullptr;

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

class MemoryBinaryIO : public BinaryIO {
private:
    mutable std::unordered_map<std::string, std::vector<std::byte>> _entries;
public:
    void clear_shader_cache() const noexcept override { _entries.clear(); }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view name) const noexcept override {
        auto iterator = _entries.find(std::string{name});
        if (iterator == _entries.end()) { return nullptr; }
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{iterator->second.data(),
                                         iterator->second.size()});
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view name) const noexcept override {
        return read_shader_bytecode(name);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(luisa::string_view) const noexcept override {
        return nullptr;
    }
    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        _entries[std::string{name}] = {data.begin(), data.end()};
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
    [[nodiscard]] size_t size() const noexcept { return _entries.size(); }
    [[nodiscard]] std::vector<std::string> spv_names() const {
        std::vector<std::string> names;
        for (auto &&entry : _entries) {
            if (entry.first.ends_with(".spv")) { names.emplace_back(entry.first); }
        }
        std::sort(names.begin(), names.end());
        return names;
    }
    [[nodiscard]] std::vector<std::byte> const *find(std::string_view name) const {
        auto iterator = _entries.find(std::string{name});
        return iterator == _entries.end() ? nullptr : &iterator->second;
    }
};

// A `BinaryIO` that starts a nested `compile_shaders()` batch from inside the first
// shader-cache read the pool performs - the public `BinaryIO` seam. That read runs on
// a *preparation lane* of the pool, i.e. on a flow that belongs to the batch waiting
// for it, which is exactly the shape a nested batch has to recognise: waiting for a
// lane (or for the batch gate) from there would wait for itself.
//
// `configure_nested()` is called after the `Device` exists - the extension is a view
// over it - and before the outer batch, so nothing the device does while it
// constructs itself can start a batch. Only the *first* cache read after that
// triggers: the nested batch's own reads re-enter these two overrides, and the
// one-shot flag is set before the nested call, so they cannot start further batches.
// Everything else is inherited unchanged, so the artifacts are stored exactly as a
// plain `MemoryBinaryIO` stores them.
class NestedCompileBinaryIO final : public MemoryBinaryIO {
private:
    struct NestedBatch {
        // Null means "not armed yet"; `configure_nested` sets both members.
        VulkanShaderCompileExt *ext{nullptr};
        Function kernel;
    };
    mutable NestedBatch _nested;
    // One-shot guard. Exchanged *before* the nested batch runs, so a nested cache read
    // that comes back through the overrides below cannot start another batch.
    mutable std::atomic<bool> _nested_started{false};
    mutable std::atomic<uint32_t> _cache_reads{0u};
    // The answer of the nested batch, for the fixture to assert on.
    mutable VulkanShaderCompileStatus _nested_status{
        VulkanShaderCompileStatus::INVALID_ARGUMENT};
    mutable luisa::vector<VulkanShaderCompileResult> _nested_results;
    // The flow that started the nested batch: a preparation lane, not the caller.
    mutable std::thread::id _nested_thread{};

    // Every cache read goes through here: trigger the nested batch if it is armed and
    // has not run yet, then forward the read to the in-memory store unchanged.
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    load_after_nested_batch(luisa::string_view name) const noexcept {
        _cache_reads.fetch_add(1u, std::memory_order_relaxed);
        if (_nested.ext != nullptr &&
            !_nested_started.exchange(true, std::memory_order_acq_rel)) {
            _nested_thread = std::this_thread::get_id();
            VulkanShaderCompileRequest request{
                .option = ShaderOption{}, .kernel = _nested.kernel};
            _nested_results.clear();
            _nested_status = _nested.ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{&request, 1u},
                _nested_results, 1u);
        }
        // Qualified: the derived overrides must not be re-entered for this read.
        return MemoryBinaryIO::read_shader_bytecode(name);
    }

public:
    // Arm the nested batch: one request for `kernel`, one lane, started by this
    // BinaryIO's next cache read. The caller keeps `kernel`'s AST alive.
    void configure_nested(VulkanShaderCompileExt *ext, Function kernel) noexcept {
        _nested.ext = ext;
        _nested.kernel = kernel;
        _nested_started.store(false, std::memory_order_relaxed);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view name) const noexcept override {
        return load_after_nested_batch(name);
    }
    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view name) const noexcept override {
        return load_after_nested_batch(name);
    }
    [[nodiscard]] bool nested_started() const noexcept {
        return _nested_started.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint32_t cache_reads() const noexcept {
        return _cache_reads.load(std::memory_order_relaxed);
    }
    [[nodiscard]] VulkanShaderCompileStatus nested_status() const noexcept {
        return _nested_status;
    }
    [[nodiscard]] luisa::vector<VulkanShaderCompileResult> const &nested_results() const noexcept {
        return _nested_results;
    }
    [[nodiscard]] std::thread::id nested_thread() const noexcept { return _nested_thread; }
};

// -------------------------------------------------------------- kernels ---
// `count` distinct element-wise kernels; the constant is baked into the AST, so
// every one of them has its own cache key and its own artifact.
luisa::vector<Kernel1D<Buffer<float>>> make_kernels(uint32_t count) {
    luisa::vector<Kernel1D<Buffer<float>>> kernels;
    kernels.reserve(count);
    for (auto i = 0u; i < count; i++) {
        auto k = i + 1u;
        kernels.emplace_back([k](BufferFloat buffer) noexcept {
            auto x = cast<float>(dispatch_id().x);
            buffer.write(dispatch_id().x, x * cast<float>(k) + 0.5f);
        });
    }
    return kernels;
}

[[nodiscard]] VulkanShaderCompileExt *compile_ext(Device &device) {
    auto *ext = device.extension<VulkanShaderCompileExt>();
    expect(ext != nullptr) << "the Vulkan backend must provide VulkanShaderCompileExt";
    return ext;
}

// One dispatch of one shader handle, then the numeric check of its result. The
// argument count and uniform size are the values a `Shader1D` object would store
// for the same kernel (`Device::compile` computes exactly these).
struct Dispatcher {
    Stream &stream;
    Buffer<float> &buffer;
    luisa::vector<float> &host;
    uint32_t dispatch_size;
    void verify(uint64_t handle, Function kernel, uint32_t k,
                uint32_t index) noexcept {
        luisa::test::dispatch_one_buffer_shader(
            stream, handle,
            luisa::test::batch_shader_arg_count(kernel),
            luisa::test::batch_shader_uniform_size(kernel),
            buffer, dispatch_size, host);
        for (auto element = 0u; element < dispatch_size; element++) {
            auto expected =
                static_cast<float>(element) * static_cast<float>(k) + 0.5f;
            expect(std::abs(host[element] - expected) < 1e-4f)
                << "shader " << index << " element " << element;
        }
    }
};

}// namespace

static void test_serial_path_is_unchanged() {
    "vk_shader_compile_serial_path_unchanged"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        // A device that only ever compiles serially answers the pool diagnostics
        // without creating a worker.
        expect(device.query("vk_shader_compile_parallel") == "1");
        expect(device.query("vk_shader_compile_pool_state") == "not_created");
        expect(device.query("vk_shader_compile_worker_count") == "0");
        auto kernels = make_kernels(2u);
        auto shader = device.compile(kernels[0]);
        expect(shader.handle() != 0u) << "a serial compile must still create a shader";
        expect(device.query("vk_shader_compile_pool_state") == "not_created")
            << "a serial compile must not create the pool";
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        dispatcher.verify(shader.handle(), kernels[0].function()->function(), 1u, 0u);
        // Shutting the (never created) pool down must be harmless, and the serial
        // path must keep working afterwards.
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        ext->shutdown_pool();
        ext->shutdown_pool();// idempotent
        auto stats = ext->stats();
        expect(stats.pool_shutdown);
        expect(!stats.pool_created);
        auto shader2 = device.compile(kernels[1]);
        expect(shader2.handle() != 0u)
            << "a pool shutdown must not affect the serial path";
        dispatcher.verify(shader2.handle(), kernels[1].function()->function(), 2u, 1u);
    };
}

static void test_batch_order_and_results() {
    "vk_shader_compile_batch_matches_serial"_test = [] {
        constexpr auto kShaderCount = 24u;
        auto kernels = make_kernels(kShaderCount);
        // Serial reference: one artifact per kernel, written by `Device::compile`.
        MemoryBinaryIO serial_io;
        {
            DeviceConfig config;
            config.binary_io = &serial_io;
            auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
            if (!context) { return; }
            auto &device = context->device;
            luisa::vector<Shader1D<Buffer<float>>> shaders;
            shaders.reserve(kShaderCount);
            for (auto &kernel : kernels) { shaders.emplace_back(device.compile(kernel)); }
            expect(shaders.size() == kShaderCount);
        }
        auto serial_names = serial_io.spv_names();
        expect(serial_names.size() == kShaderCount)
            << "the serial reference must persist one artifact per kernel";
        // Batch: the same kernels, prepared on workers and published in order.
        MemoryBinaryIO batch_io;
        {
            DeviceConfig config;
            config.binary_io = &batch_io;
            auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
            if (!context) { return; }
            auto &device = context->device;
            auto *ext = compile_ext(device);
            if (ext == nullptr) { return; }
            luisa::vector<VulkanShaderCompileRequest> requests;
            requests.reserve(kShaderCount);
            for (auto &kernel : kernels) {
                requests.emplace_back(VulkanShaderCompileRequest{
                    .option = ShaderOption{}, .kernel = kernel.function()->function()});
            }
            luisa::vector<VulkanShaderCompileResult> results;
            auto status = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests},
                results, 4u);
            expect(status == VulkanShaderCompileStatus::SUCCESS);
            expect(results.size() == kShaderCount);
            auto stats = ext->stats();
            expect(stats.worker_count == 4u);
            expect(stats.batches == 1u);
            expect(stats.requests == kShaderCount);
            expect(stats.parallel_prepares == kShaderCount);
            expect(stats.serial_fallbacks == 0u);
            expect(device.query("vk_shader_compile_pool_state") == "running");
            expect(device.query("vk_shader_compile_worker_count") == "4");
            auto stream = device.create_stream();
            auto buffer = device.create_buffer<float>(4u);
            luisa::vector<float> host(4u);
            Dispatcher dispatcher{stream, buffer, host, 4u};
            for (auto i = 0u; i < kShaderCount; i++) {
                expect(results[i].status == VulkanShaderCompileStatus::SUCCESS)
                    << "request " << i;
                expect(!results[i].serial_fallback) << "request " << i;
                expect(results[i].shader.handle != 0u) << "request " << i;
                // Request order: the handle at index i must be kernel i's shader,
                // which its numeric result proves.
                dispatcher.verify(results[i].shader.handle,
                                  kernels[i].function()->function(), i + 1u, i);
            }
        }
        // Artifact identity: same names, same sizes, same bytes as the serial loop.
        auto batch_names = batch_io.spv_names();
        expect(batch_names == serial_names)
            << "a batch must persist exactly the serial artifacts";
        auto identical = true;
        for (auto &name : serial_names) {
            auto *serial = serial_io.find(name);
            auto *batch = batch_io.find(name);
            if (serial == nullptr || batch == nullptr ||
                serial->size() != batch->size() ||
                std::memcmp(serial->data(), batch->data(), serial->size()) != 0) {
                identical = false;
                LUISA_WARNING("artifact '{}' differs from the serial one", name);
            }
        }
        expect(identical) << "batch artifacts must be byte-identical to serial ones";
    };
}

static void test_worker_counts_agree() {
    "vk_shader_compile_worker_counts"_test = [] {
        constexpr auto kShaderCount = 8u;
        auto kernels = make_kernels(kShaderCount);
        luisa::vector<std::vector<std::byte>> reference;
        std::vector<std::string> reference_names;
        for (auto workers : {1u, 2u, 5u}) {
            MemoryBinaryIO binary_io;
            DeviceConfig config;
            config.binary_io = &binary_io;
            auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
            if (!context) { return; }
            auto &device = context->device;
            auto *ext = compile_ext(device);
            if (ext == nullptr) { return; }
            luisa::vector<VulkanShaderCompileRequest> requests;
            for (auto &kernel : kernels) {
                requests.emplace_back(VulkanShaderCompileRequest{
                    .option = ShaderOption{}, .kernel = kernel.function()->function()});
            }
            luisa::vector<VulkanShaderCompileResult> results;
            expect(ext->compile_shaders(
                       luisa::span<const VulkanShaderCompileRequest>{requests},
                       results, workers) == VulkanShaderCompileStatus::SUCCESS);
            expect(ext->stats().worker_count == workers);
            auto stream = device.create_stream();
            auto buffer = device.create_buffer<float>(4u);
            luisa::vector<float> host(4u);
            Dispatcher dispatcher{stream, buffer, host, 4u};
            for (auto i = 0u; i < kShaderCount; i++) {
                expect(results[i].status == VulkanShaderCompileStatus::SUCCESS);
                dispatcher.verify(results[i].shader.handle,
                                  kernels[i].function()->function(), i + 1u, i);
            }
            auto names = binary_io.spv_names();
            if (reference.empty()) {
                reference_names = names;
                for (auto &name : names) {
                    auto *data = binary_io.find(name);
                    reference.emplace_back(data->begin(), data->end());
                }
            } else {
                expect(names == reference_names)
                    << "worker count " << workers << " changed the artifact set";
                auto identical = names.size() == reference.size();
                for (auto i = 0u; identical && i < names.size(); i++) {
                    auto *data = binary_io.find(names[i]);
                    identical = data->size() == reference[i].size() &&
                                std::memcmp(data->data(), reference[i].data(),
                                            data->size()) == 0;
                }
                expect(identical)
                    << "worker count " << workers << " changed the artifact bytes";
            }
        }
    };
}

static void test_hlsl_route_falls_back_to_serial() {
    "vk_shader_compile_hlsl_route_serial_fallback"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        expect(!ext->capabilities().parallel_hlsl_compatibility_route)
            << "the compatibility HLSL route is documented as serial";
        // Printing forces the compatibility HLSL-to-SPIR-V route, which this
        // backend does not prepare on a worker.
        auto printer_kernel = [](BufferFloat buffer) noexcept {
            auto x = cast<float>(dispatch_id().x);
            device_log("value {}", x);
            buffer.write(dispatch_id().x, x);
        };
        auto normal_kernel = [](BufferFloat buffer) noexcept {
            buffer.write(dispatch_id().x, 1.0f);
        };
        // The DSL kernels own the ASTs the requests point at.
        Kernel1D<Buffer<float>> normal{normal_kernel};
        Kernel1D<Buffer<float>> printer{printer_kernel};
        luisa::vector<VulkanShaderCompileRequest> requests;
        requests.emplace_back(VulkanShaderCompileRequest{
            .option = ShaderOption{},
            .kernel = normal.function()->function()});
        requests.emplace_back(VulkanShaderCompileRequest{
            .option = ShaderOption{},
            .kernel = printer.function()->function()});
        luisa::vector<VulkanShaderCompileResult> results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   results, 4u) == VulkanShaderCompileStatus::SUCCESS);
        expect(results.size() == 2u);
        expect(!results[0].serial_fallback)
            << "a plain kernel must be prepared on a worker";
        expect(results[1].serial_fallback)
            << "the HLSL compatibility route must be reported as a serial fallback";
        expect(results[0].status == VulkanShaderCompileStatus::SUCCESS);
        expect(results[1].status == VulkanShaderCompileStatus::SUCCESS);
        expect(results[1].shader.handle != 0u)
            << "a serial fallback must still produce a shader";
        expect(ext->stats().serial_fallbacks == 1u);
        // The device is still usable for ordinary work.
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        luisa::test::dispatch_one_buffer_shader(
            stream, results[0].shader.handle,
            luisa::test::batch_shader_arg_count(normal.function()->function()),
            luisa::test::batch_shader_uniform_size(normal.function()->function()),
            buffer, 4u, host);
        for (auto i = 0u; i < 4u; i++) { expect(std::abs(host[i] - 1.0f) < 1e-6f); }
    };
}

static void test_compile_only_batch() {
    "vk_shader_compile_compile_only"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        auto kernels = make_kernels(2u);
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto i = 0u; i < 2u; i++) {
            ShaderOption option;
            option.compile_only = true;
            option.enable_cache = false;
            option.name = luisa::format("p5_1_aot_{}.spv", i);
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = option,
                .kernel = kernels[i].function()->function()});
        }
        luisa::vector<VulkanShaderCompileResult> results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   results, 2u) == VulkanShaderCompileStatus::SUCCESS);
        for (auto i = 0u; i < 2u; i++) {
            expect(results[i].status == VulkanShaderCompileStatus::SUCCESS);
            expect(!results[i].shader.valid())
                << "a compile-only request must not create a shader object";
            expect(results[i].shader.compile_ok);
            auto name = luisa::format("p5_1_aot_{}.spv", i);
            expect(binary_io.find(name) != nullptr)
                << "a compile-only request in a batch must persist its artifact";
        }
    };
}

static void test_concurrent_ordinary_create() {
    "vk_shader_compile_concurrent_create_shader"_test = [] {
        constexpr auto kThreads = 4u;
        constexpr auto kPerThread = 4u;
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        // `kThreads * kPerThread` distinct kernels, compiled by ordinary
        // `Device::compile` calls from several threads at once: preparation overlaps,
        // publication is serialized device-wide.
        auto kernels = make_kernels(kThreads * kPerThread);
        luisa::vector<uint64_t> handles(kernels.size(), 0u);
        std::atomic<uint32_t> failures{0u};
        luisa::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (auto t = 0u; t < kThreads; t++) {
            threads.emplace_back([&, t] {
                for (auto j = 0u; j < kPerThread; j++) {
                    auto index = t * kPerThread + j;
                    auto shader = device.compile(kernels[index]);
                    if (shader.handle() == 0u) { failures.fetch_add(1u); }
                    handles[index] = shader.handle();
                    // Keep the shader alive until the thread is done with it.
                    static_cast<void>(shader);
                }
            });
        }
        for (auto &thread : threads) { thread.join(); }
        expect(failures.load() == 0u) << "concurrent create_shader returned no handle";
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        // The handles above are owned by shader objects that already died at the end
        // of each thread, so this only proves creation succeeded; the dispatch checks
        // happen in the single-threaded tests above. Here we re-compile one kernel per
        // thread sequentially to prove the device is still healthy.
        for (auto index = 0u; index < kernels.size(); index++) {
            auto shader = device.compile(kernels[index]);
            dispatcher.verify(shader.handle(), kernels[index].function()->function(),
                              index + 1u, index);
        }
    };
}

static void test_shutdown_with_live_work() {
    "vk_shader_compile_shutdown_with_live_work"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        auto kernels = make_kernels(48u);
        luisa::vector<VulkanShaderCompileRequest> requests;
        requests.reserve(kernels.size());
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        VulkanShaderCompileStatus status{VulkanShaderCompileStatus::INVALID_ARGUMENT};
        luisa::vector<VulkanShaderCompileResult> results;
        std::thread batch{[&] {
            status = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests},
                results, 4u);
        }};
        // Let the batch start, then stop the pool underneath it. `stop()` waits for
        // the batch mutex, so the batch finishes; the workers drain their tickets
        // first. No ticket may be lost, and no result may be missing.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ext->shutdown_pool();
        batch.join();
        expect(status == VulkanShaderCompileStatus::SUCCESS)
            << "an in-flight batch must complete when the pool is shut down";
        expect(results.size() == kernels.size());
        for (auto i = 0u; i < results.size(); i++) {
            expect(results[i].status == VulkanShaderCompileStatus::SUCCESS)
                << "request " << i << " after shutdown";
            expect(results[i].shader.handle != 0u) << "request " << i;
        }
        expect(device.query("vk_shader_compile_pool_state") == "shutdown");
        // Every later batch is refused, and it creates nothing.
        auto before = binary_io.size();
        luisa::vector<VulkanShaderCompileRequest> late_requests{
            VulkanShaderCompileRequest{
                .option = ShaderOption{},
                .kernel = kernels.front().function()->function()}};
        luisa::vector<VulkanShaderCompileResult> late_results;
        auto late_status = ext->compile_shaders(
            luisa::span<const VulkanShaderCompileRequest>{late_requests},
            late_results, 4u);
        expect(late_status == VulkanShaderCompileStatus::POOL_SHUTDOWN);
        expect(late_results.size() == 1u);
        expect(late_results[0].status == VulkanShaderCompileStatus::POOL_SHUTDOWN);
        expect(!late_results[0].shader.valid());
        expect(binary_io.size() == before)
            << "a refused batch must not write an artifact";
        // The drained batch's handles are usable shaders.
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        dispatcher.verify(results[0].shader.handle, kernels[0].function()->function(), 1u, 0u);
        dispatcher.verify(results.back().shader.handle,
                          kernels.back().function()->function(), 48u, 47u);
    };
}

static void test_repeated_batch_uses_the_cache() {
    "vk_shader_compile_repeated_batch"_test = [] {
        constexpr auto kShaderCount = 8u;
        auto kernels = make_kernels(kShaderCount);
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        luisa::vector<VulkanShaderCompileResult> cold_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   cold_results, 3u) == VulkanShaderCompileStatus::SUCCESS);
        auto stats_after_cold = ext->stats();
        expect(stats_after_cold.cache_hits == 0u) << "the first batch must be cold";
        expect(stats_after_cold.batches == 1u);
        auto artifacts_after_cold = binary_io.size();
        // Second batch on the same device: every request is a cache hit, served by
        // the same worker set, and it writes nothing new.
        luisa::vector<VulkanShaderCompileResult> warm_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   warm_results, 3u) == VulkanShaderCompileStatus::SUCCESS);
        auto stats_after_warm = ext->stats();
        expect(stats_after_warm.batches == 2u);
        expect(stats_after_warm.requests == 2u * kShaderCount);
        expect(stats_after_warm.cache_hits >= kShaderCount)
            << "the second batch must load every artifact from the cache";
        expect(stats_after_warm.worker_count == 3u)
            << "a second batch must reuse the pool instead of recreating it";
        expect(binary_io.size() == artifacts_after_cold)
            << "a warm batch must not rewrite the artifacts";
        // Both batches' handles must be usable shaders of this device.
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kShaderCount; i++) {
            dispatcher.verify(warm_results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
        // A request that disables the cache entirely (`enable_cache == false`) must
        // still compile and publish, without reading or writing the BinaryIO.
        {
            ShaderOption nc;
            nc.enable_cache = false;
            luisa::vector<VulkanShaderCompileRequest> no_cache_requests{
                VulkanShaderCompileRequest{
                    .option = nc,
                    .kernel = kernels[0].function()->function()}};
            luisa::vector<VulkanShaderCompileResult> no_cache_results;
            auto artifacts_before = binary_io.size();
            expect(ext->compile_shaders(
                       luisa::span<const VulkanShaderCompileRequest>{no_cache_requests},
                       no_cache_results, 2u) == VulkanShaderCompileStatus::SUCCESS);
            expect(no_cache_results.size() == 1u);
            expect(no_cache_results[0].status == VulkanShaderCompileStatus::SUCCESS);
            expect(no_cache_results[0].shader.valid());
            expect(binary_io.size() == artifacts_before)
                << "a cache-disabled request must not write an artifact";
            dispatcher.verify(no_cache_results[0].shader.handle,
                              kernels[0].function()->function(), 1u, 0u);
        }
        // And a later serial compile of the same kernel is still a cache hit with
        // the same artifact set (the warm path shares one cache with the pool).
        auto serial_shader = device.compile(kernels[0]);
        dispatcher.verify(serial_shader.handle(),
                          kernels[0].function()->function(), 1u, 0u);
        expect(binary_io.size() == artifacts_after_cold);
    };
}

// A batch that is started from one of the pool's own preparation lanes must be
// compiled serially on that lane: the lane belongs to the batch the caller is waiting
// for, so waiting for a lane - or for the batch gate - would wait for itself. The
// seam used here is the public `BinaryIO` shader-cache callback, which the pool calls
// from the lane, and on the first read of a real batch it starts a nested batch.
static void test_nested_prepare_serial_fallback() {
    "vk_shader_compile_nested_prepare_serial_fallback"_test = [] {
        NestedCompileBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        // Nothing is bound to this thread, so the pool owns a `std::thread` worker and
        // the outer request is really prepared on a lane of the pool.
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        expect(device.query("vk_shader_compile_executor") == "std_thread")
            << "this fixture needs the executor that owns its own preparation lane";
        expect(device.query("vk_shader_compile_pool_state") == "not_created")
            << "constructing the device must not start the pool";
        auto kernels = make_kernels(2u);
        // Two distinct kernels, so the outer request and the nested batch share no
        // cache key and no artifact. Arming happens after the device exists (the
        // extension is a view over it) and before the outer batch.
        binary_io.configure_nested(ext, kernels[1].function()->function());
        luisa::vector<VulkanShaderCompileRequest> requests{
            VulkanShaderCompileRequest{
                .option = ShaderOption{},
                .kernel = kernels[0].function()->function()}};
        luisa::vector<VulkanShaderCompileResult> results;
        // One lane, one request: the cache read of that lane's prepare is the first
        // read after arming, so it starts the nested batch from inside the lane.
        auto status = ext->compile_shaders(
            luisa::span<const VulkanShaderCompileRequest>{requests}, results, 1u);
        expect(status == VulkanShaderCompileStatus::SUCCESS)
            << "the outer batch must complete, got status "
            << static_cast<uint32_t>(status);
        expect(results.size() == 1u);
        expect(results[0].status == VulkanShaderCompileStatus::SUCCESS)
            << "the outer request must publish a shader";
        expect(results[0].shader.handle != 0u);
        expect(!results[0].serial_fallback)
            << "the outer request must be prepared on a lane, not on the caller";
        // The nested batch: really started by a preparation lane, exactly once, from
        // the cache read rather than from the caller.
        expect(binary_io.nested_started())
            << "the outer prepare must read the shader cache through the BinaryIO seam";
        expect(binary_io.cache_reads() >= 2u);
        expect(binary_io.nested_thread() != std::this_thread::get_id())
            << "the nested batch must be started by a pool lane, not by the caller";
        auto nested_status = binary_io.nested_status();
        expect(nested_status == VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL)
            << "a batch started on a lane of the batch that owns it must be compiled "
               "serially on that lane, got status "
            << static_cast<uint32_t>(nested_status);
        auto &nested_results = binary_io.nested_results();
        expect(nested_results.size() == 1u);
        expect(nested_results[0].status == VulkanShaderCompileStatus::SUCCESS);
        expect(nested_results[0].serial_fallback)
            << "the nested request must be reported as prepared on the calling flow";
        expect(nested_results[0].shader.handle != 0u)
            << "a nested serial fallback must still produce a shader";
        expect(binary_io.size() >= 2u)
            << "both shaders must persist their artifact";
        // The counters say where each request was really prepared: the outer request
        // on a lane of the pool, the nested request on the flow that asked for it.
        auto stats = ext->stats();
        expect(stats.worker_count == 1u);
        expect(stats.pool_created);
        expect(stats.batches == 2u) << "the outer and the nested batch must both count";
        expect(stats.requests == 2u);
        expect(stats.parallel_prepares == 1u)
            << "the outer request must still be prepared in parallel";
        expect(stats.serial_prepares == 1u)
            << "only the nested request must be prepared serially";
        // Both handles are working shaders of this device and answer with the constant
        // of their own kernel, so the nested batch did not disturb the outer
        // publication or its result.
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        dispatcher.verify(results[0].shader.handle,
                          kernels[0].function()->function(), 1u, 0u);
        dispatcher.verify(nested_results[0].shader.handle,
                          kernels[1].function()->function(), 2u, 1u);
    };
}

static void test_empty_batch() {
    "vk_shader_compile_empty_batch"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto *ext = compile_ext(context->device);
        if (ext == nullptr) { return; }
        luisa::vector<VulkanShaderCompileRequest> requests;
        luisa::vector<VulkanShaderCompileResult> results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   results, 2u) == VulkanShaderCompileStatus::INVALID_ARGUMENT);
        expect(results.empty());
        expect(ext->stats().batches == 0u);
        expect(!ext->stats().pool_created)
            << "an empty batch must not create a worker";
    };
}

// ------------------------------------------------------- executor choice ---
// The pool's executor is fixed by the scheduler that is bound to the
// *device-constructing* thread (see `shader_compile_pool.h`). These tests pin that
// decision down from the outside - through the diagnostics `Device::query` exposes,
// so the public ABI stays untouched - and then run a batch on each executor.
//
// Every device here is created inside a scope that is closed *before* the scheduler
// that provided the fibers: the pool enqueues each batch's lane tasks on that
// scheduler, so it has to outlive the device.

static void test_std_thread_executor_selection() {
    "vk_shader_compile_executor_std_thread"_test = [] {
        MemoryBinaryIO binary_io;
        DeviceConfig config;
        config.binary_io = &binary_io;
        // Nothing is bound to this thread, so nothing can be captured and the pool
        // must own its workers.
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        expect(device.query("vk_shader_compile_executor") == "std_thread")
            << "a device constructed without a bound scheduler must own its threads";
        expect(device.query("vk_shader_compile_scheduler_workers") == "0")
            << "no scheduler may be captured when none is bound";
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto kShaderCount = 6u;
        auto kernels = make_kernels(kShaderCount);
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        luisa::vector<VulkanShaderCompileResult> results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   results, 2u) == VulkanShaderCompileStatus::SUCCESS);
        auto stats = ext->stats();
        expect(stats.pool_created) << "a batch must have created the workers";
        expect(stats.worker_count == 2u);
        expect(stats.parallel_prepares == kShaderCount);
        expect(device.query("vk_shader_compile_executor") == "std_thread")
            << "running a batch must not re-select the executor";
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kernels.size(); i++) {
            expect(results[i].status == VulkanShaderCompileStatus::SUCCESS);
            dispatcher.verify(results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
        ext->shutdown_pool();
        expect(device.query("vk_shader_compile_pool_state") == "shutdown");
        luisa::vector<VulkanShaderCompileResult> late_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   late_results, 2u) == VulkanShaderCompileStatus::POOL_SHUTDOWN);
        expect(late_results.size() == kShaderCount);
        expect(!late_results.front().shader.valid());
    };
}

static void test_single_threaded_scheduler_never_creates_owned_threads() {
    "vk_shader_compile_executor_single_threaded_scheduler"_test = [] {
        MemoryBinaryIO binary_io;
        luisa::fiber::scheduler scheduler{0u};
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        expect(device.query("vk_shader_compile_executor") == "fiber")
            << "any scheduler bound on the constructor thread selects the scheduler executor";
        expect(device.query("vk_shader_compile_scheduler_workers") == "0");
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        auto kernels = make_kernels(2u);
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        luisa::vector<VulkanShaderCompileResult> results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   results, 2u) == VulkanShaderCompileStatus::FELL_BACK_TO_SERIAL)
            << "a scheduler without workers has no parallel preparation capacity";
        auto stats = ext->stats();
        expect(stats.worker_count == 0u && !stats.pool_created)
            << "the single-threaded scheduler path must not create owned std::threads";
        expect(stats.serial_prepares == requests.size());
        expect(stats.parallel_prepares == 0u);
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kernels.size(); i++) {
            expect(results[i].status == VulkanShaderCompileStatus::SUCCESS);
            expect(results[i].serial_fallback);
            dispatcher.verify(results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
    };
}

static void test_fiber_executor_batch() {
    "vk_shader_compile_executor_fiber"_test = [] {
        MemoryBinaryIO binary_io;
        // Declared before the device and destroyed after it (reverse declaration
        // order), which is the lifetime contract of a captured scheduler.
        luisa::fiber::scheduler scheduler{4u};
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        constexpr auto kShaderCount = 12u;
        auto kernels = make_kernels(kShaderCount);
        expect(device.query("vk_shader_compile_executor") == "fiber")
            << "a scheduler bound to the constructing thread must be captured";
        expect(device.query("vk_shader_compile_scheduler_workers") == "4")
            << "the captured scheduler's worker threads must be reported";
        expect(device.query("vk_shader_compile_pool_state") == "not_created")
            << "capturing an executor must not create a lane";
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        luisa::vector<VulkanShaderCompileResult> cold_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   cold_results, 3u) == VulkanShaderCompileStatus::SUCCESS);
        auto cold = ext->stats();
        expect(cold.worker_count == 3u) << "the requested width must be reported";
        expect(cold.pool_created) << "a batch must have used lanes";
        expect(cold.batches == 1u);
        expect(cold.requests == kShaderCount);
        expect(cold.parallel_prepares == kShaderCount)
            << "every request must be prepared on a lane";
        expect(cold.serial_prepares == 0u);
        expect(cold.serial_fallbacks == 0u);
        expect(cold.cache_hits == 0u) << "the first batch must be cold";
        expect(device.query("vk_shader_compile_pool_state") == "running");
        expect(device.query("vk_shader_compile_worker_count") == "3");
        expect(device.query("vk_shader_compile_executor") == "fiber");
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kernels.size(); i++) {
            expect(cold_results[i].status == VulkanShaderCompileStatus::SUCCESS)
                << "request " << i;
            expect(!cold_results[i].serial_fallback) << "request " << i;
            expect(cold_results[i].shader.handle != 0u) << "request " << i;
            // Request order across lanes: handle i must be kernel i's shader.
            dispatcher.verify(cold_results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
        // A second batch reuses the same lanes and is served from the cache: the
        // fiber executor must not change what the pool counts.
        luisa::vector<VulkanShaderCompileResult> warm_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   warm_results, 3u) == VulkanShaderCompileStatus::SUCCESS);
        auto warm = ext->stats();
        expect(warm.batches == 2u);
        expect(warm.requests == 2u * kShaderCount);
        expect(warm.cache_hits >= kShaderCount)
            << "the second batch must load every artifact from the cache";
        expect(warm.worker_count == 3u);
        for (auto i = 0u; i < kernels.size(); i++) {
            dispatcher.verify(warm_results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
        // A batch submitted from a plain, *unbound* thread must still be prepared on
        // the captured scheduler: the pool enqueues on the scheduler it captured at
        // construction, not on whatever happens to be bound to the calling thread.
        VulkanShaderCompileStatus unbound_status{
            VulkanShaderCompileStatus::INVALID_ARGUMENT};
        luisa::vector<VulkanShaderCompileResult> unbound_results;
        std::thread unbound{[&] {
            unbound_status = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests},
                unbound_results, 2u);
        }};
        unbound.join();
        expect(unbound_status == VulkanShaderCompileStatus::SUCCESS)
            << "an unbound calling thread must still reach the captured scheduler";
        expect(ext->stats().worker_count == 2u);
        expect(ext->stats().parallel_prepares == 3u * kShaderCount);
        for (auto i = 0u; i < kernels.size(); i++) {
            expect(unbound_results[i].shader.handle != 0u) << "request " << i;
            dispatcher.verify(unbound_results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
        // Shutdown: this executor owns no thread, but the batch contract is the same.
        ext->shutdown_pool();
        ext->shutdown_pool();// idempotent
        expect(device.query("vk_shader_compile_pool_state") == "shutdown");
        luisa::vector<VulkanShaderCompileResult> late_results;
        expect(ext->compile_shaders(
                   luisa::span<const VulkanShaderCompileRequest>{requests},
                   late_results, 3u) == VulkanShaderCompileStatus::POOL_SHUTDOWN);
        expect(late_results.size() == kShaderCount);
        for (auto &result : late_results) {
            expect(result.status == VulkanShaderCompileStatus::POOL_SHUTDOWN);
            expect(!result.shader.valid());
        }
    };
}

static void test_fiber_executor_publisher_on_fiber() {
    "vk_shader_compile_publisher_on_fiber"_test = [] {
        MemoryBinaryIO binary_io;
        luisa::fiber::scheduler scheduler{4u};
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto kShaderCount = 8u;
        auto kernels = make_kernels(kShaderCount);
        luisa::vector<VulkanShaderCompileRequest> requests;
        for (auto &kernel : kernels) {
            requests.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{}, .kernel = kernel.function()->function()});
        }
        // The publisher is a scheduler fiber, not a plain thread: it waits for the
        // lanes through marl events, so it must suspend its fiber instead of blocking
        // the OS worker underneath it.
        luisa::vector<VulkanShaderCompileResult> results;
        VulkanShaderCompileStatus status{VulkanShaderCompileStatus::INVALID_ARGUMENT};
        auto done = luisa::fiber::async([&] {
            status = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests},
                results, 2u);
        });
        done.wait();
        expect(status == VulkanShaderCompileStatus::SUCCESS);
        expect(results.size() == kShaderCount);
        expect(ext->stats().worker_count == 2u);
        expect(ext->stats().parallel_prepares == kShaderCount);
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kernels.size(); i++) {
            expect(results[i].status == VulkanShaderCompileStatus::SUCCESS)
                << "request " << i;
            dispatcher.verify(results[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
        }
    };
}

static void test_fiber_executor_concurrent_batches() {
    "vk_shader_compile_fiber_concurrent_batches"_test = [] {
        MemoryBinaryIO binary_io;
        luisa::fiber::scheduler scheduler{4u};
        DeviceConfig config;
        config.binary_io = &binary_io;
        auto context = luisa::test::create_device_from_ut(g_argc, g_argv, &config);
        if (!context) { return; }
        auto &device = context->device;
        auto *ext = compile_ext(device);
        if (ext == nullptr) { return; }
        constexpr auto kShaderCount = 4u;
        auto kernels = make_kernels(2u * kShaderCount);
        luisa::vector<VulkanShaderCompileRequest> requests_a;
        luisa::vector<VulkanShaderCompileRequest> requests_b;
        for (auto i = 0u; i < kShaderCount; i++) {
            requests_a.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{},
                .kernel = kernels[i].function()->function()});
            requests_b.emplace_back(VulkanShaderCompileRequest{
                .option = ShaderOption{},
                .kernel = kernels[kShaderCount + i].function()->function()});
        }
        // Two batches at once, one from a plain thread and one from a scheduler
        // fiber. The second caller contends for the batch lock, and a fiber must
        // suspend there instead of blocking an OS worker the first batch's lanes may
        // need - so both batches have to complete.
        luisa::vector<VulkanShaderCompileResult> results_a;
        luisa::vector<VulkanShaderCompileResult> results_b;
        VulkanShaderCompileStatus status_a{VulkanShaderCompileStatus::INVALID_ARGUMENT};
        VulkanShaderCompileStatus status_b{VulkanShaderCompileStatus::INVALID_ARGUMENT};
        auto b = luisa::fiber::async([&] {
            status_b = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests_b},
                results_b, 2u);
        });
        std::thread a{[&] {
            status_a = ext->compile_shaders(
                luisa::span<const VulkanShaderCompileRequest>{requests_a},
                results_a, 2u);
        }};
        a.join();
        b.wait();
        expect(status_a == VulkanShaderCompileStatus::SUCCESS);
        expect(status_b == VulkanShaderCompileStatus::SUCCESS);
        expect(results_a.size() == kShaderCount);
        expect(results_b.size() == kShaderCount);
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<float>(4u);
        luisa::vector<float> host(4u);
        Dispatcher dispatcher{stream, buffer, host, 4u};
        for (auto i = 0u; i < kShaderCount; i++) {
            expect(results_a[i].shader.handle != 0u) << "batch A request " << i;
            dispatcher.verify(results_a[i].shader.handle,
                              kernels[i].function()->function(), i + 1u, i);
            expect(results_b[i].shader.handle != 0u) << "batch B request " << i;
            dispatcher.verify(results_b[i].shader.handle,
                              kernels[kShaderCount + i].function()->function(),
                              kShaderCount + i + 1u, kShaderCount + i);
        }
    };
}

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_serial_path_is_unchanged();
    test_batch_order_and_results();
    test_worker_counts_agree();
    test_hlsl_route_falls_back_to_serial();
    test_compile_only_batch();
    test_concurrent_ordinary_create();
    test_shutdown_with_live_work();
    test_repeated_batch_uses_the_cache();
    test_nested_prepare_serial_fallback();
    test_empty_batch();
    test_std_thread_executor_selection();
    test_single_threaded_scheduler_never_creates_owned_threads();
    test_fiber_executor_batch();
    test_fiber_executor_publisher_on_fiber();
    test_fiber_executor_concurrent_batches();
    return 0;
}