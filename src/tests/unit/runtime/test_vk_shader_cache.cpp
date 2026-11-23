// Test for the Vulkan shader-artifact and pipeline-cache surface (plan P1.3).
// This test covers:
// - SPIR-V bytecode caching: a cold compile persists exactly one `.spv`
//   artifact, a hot compile rewrites nothing
// - P1.3: the VkPipelineCache is device-scoped. One Device owns exactly one
//   VkPipelineCache, created when the Device is; a compile never creates a
//   per-shader cache and never writes a per-shader PSO artifact
// - P1.3 cold -> warm: the device cache is persisted as exactly one versioned
//   global artifact at device teardown (not one file per shader) and a second
//   Device seeded from it reports the seed as loaded while the SPIR-V cache and
//   the dispatch results stay intact
// - P1.3 fail-closed input handling: a truncated and a foreign-device global
//   artifact are rejected with the reason reported through `Device::query`, and
//   the Device still comes up with a valid (empty-seed) cache and keeps the
//   SPIR-V bytecode cache
// - P1.3 legacy seeds: a pre-P1.3 per-shader `.vk` blob is still an add-only
//   input, merged with vkMergePipelineCaches into the device cache; a corrupt
//   one is rejected without disturbing the cache
// - P1.3 ownership: releasing a compiled shader never destroys the device cache
//   (the handle, the state and the next compile/dispatch all stay valid)
// - the driver-optimization policy changes the pipeline creation but not the
//   cache, and no per-shader artifact is written for either policy
//
// The strict native XIR-to-SPIR-V guard is deliberately not enabled here: the
// legacy-seed half of the contract only exists on the compatibility
// HLSL-to-SPIR-V route, which a `native_include` block selects. The route
// contract itself is covered by test_vk_spirv_codegen_path and
// test_vk_native_route_guard.

#include "ut/ut.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <luisa/core/binary_io.h>
#include <luisa/core/stl/string.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/vstl/md5.h>

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;

namespace {

// ---------------------------------------------------------------- BinaryIO ---

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
};

class MemoryBinaryIO final : public BinaryIO {

private:
    mutable std::unordered_map<std::string, std::vector<std::byte>> _cache;

public:
    // Diagnostics the assertions use: every cache read/write name, in order.
    mutable size_t cache_read_count{};
    mutable size_t cache_write_count{};
    mutable std::vector<std::string> read_names;
    mutable std::vector<std::string> write_names;

    void reset() const noexcept {
        _cache.clear();
        clear_counters();
    }

    void clear_counters() const noexcept {
        cache_read_count = 0u;
        cache_write_count = 0u;
        read_names.clear();
        write_names.clear();
    }

    void clear_shader_cache() const noexcept override { reset(); }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_bytecode(luisa::string_view) const noexcept override {
        return nullptr;
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_shader_cache(luisa::string_view name) const noexcept override {
        ++cache_read_count;
        read_names.emplace_back(std::string{name});
        auto iterator = _cache.find(std::string{name});
        if (iterator == _cache.end()) { return nullptr; }
        auto &data = iterator->second;
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{data.data(), data.size()});
    }

    [[nodiscard]] luisa::unique_ptr<BinaryStream>
    read_internal_shader(luisa::string_view) const noexcept override {
        return nullptr;
    }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view, luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    luisa::filesystem::path write_shader_cache(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        ++cache_write_count;
        write_names.emplace_back(std::string{name});
        _cache[std::string{name}] = {data.begin(), data.end()};
        return {};
    }

    luisa::filesystem::path write_internal_shader(
        luisa::string_view, luisa::span<const std::byte>) const noexcept override {
        return {};
    }

    [[nodiscard]] size_t entry_count_with_suffix(luisa::string_view suffix) const noexcept {
        auto count = size_t{0u};
        for (auto &&[name, data] : _cache) {
            static_cast<void>(data);
            count += name.ends_with(suffix) ? 1u : 0u;
        }
        return count;
    }

    [[nodiscard]] std::vector<std::string>
    entries_with_suffix(std::string_view suffix) const {
        std::vector<std::string> result;
        for (auto &&[name, data] : _cache) {
            static_cast<void>(data);
            if (std::string_view{name}.ends_with(suffix)) {
                result.emplace_back(name);
            }
        }
        return result;
    }

    [[nodiscard]] bool has(std::string_view name) const {
        return _cache.find(std::string{name}) != _cache.end();
    }

    [[nodiscard]] std::vector<std::byte> const *
    find(std::string_view name) const {
        auto iterator = _cache.find(std::string{name});
        return iterator == _cache.end() ? nullptr : &iterator->second;
    }

    void put(std::string_view name, std::vector<std::byte> data) const {
        _cache[std::string{name}] = std::move(data);
    }

    void erase(std::string_view name) const {
        _cache.erase(std::string{name});
    }
};

// ------------------------------------------------ PSO artifact framing ---
//
// Mirrors `lc::vk::detail::PipelineCacheArtifactHeader` +
// `encode_pipeline_cache_artifact` (src/backends/vk/shader_binary_contract.h).
// The test must be able to build a *legacy* per-shader blob and a hostile
// global blob, i.e. exactly the framing the backend writes, so a drift between
// this mirror and the backend contract is what the static_asserts below catch.
struct PsoArtifactHeader {
    uint64_t magic{0x314f53504b56434cull};// "LCVKPSO1"
    uint32_t version{1u};
    uint32_t reserved{};
    uint64_t payload_size{};
    std::byte payload_md5[16]{};
};
static_assert(sizeof(PsoArtifactHeader) == 40u);
static_assert(sizeof(vstd::MD5) == 16u);

[[nodiscard]] std::vector<std::byte>
frame_pso_artifact(std::vector<std::byte> const &payload) {
    auto md5 = vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(payload.data()), payload.size()}};
    PsoArtifactHeader header{};
    header.payload_size = payload.size();
    std::memcpy(header.payload_md5, &md5, sizeof(vstd::MD5));
    std::vector<std::byte> artifact(sizeof(header) + payload.size());
    std::memcpy(artifact.data(), &header, sizeof(header));
    std::memcpy(artifact.data() + sizeof(header), payload.data(), payload.size());
    return artifact;
}

[[nodiscard]] std::optional<std::vector<std::byte>>
unframe_pso_artifact(std::vector<std::byte> const &artifact) {
    if (artifact.size() < sizeof(PsoArtifactHeader)) { return std::nullopt; }
    PsoArtifactHeader header{};
    std::memcpy(&header, artifact.data(), sizeof(header));
    if (header.magic != 0x314f53504b56434cull || header.version != 1u ||
        header.reserved != 0u ||
        artifact.size() - sizeof(header) != header.payload_size) {
        return std::nullopt;
    }
    std::vector<std::byte> payload(
        artifact.begin() + sizeof(header), artifact.end());
    auto md5 = vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(payload.data()), payload.size()}};
    if (std::memcmp(&md5, header.payload_md5, sizeof(vstd::MD5)) != 0) {
        return std::nullopt;
    }
    return payload;
}

// VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID,
// deviceID, pipelineCacheUUID[16].
constexpr size_t kCacheHeaderByteSize = 4u * sizeof(uint32_t) + 16u;
static_assert(kCacheHeaderByteSize == 32u);

// --------------------------------------------------------------- kernels ---

constexpr size_t kRunCount = 1u;

Kernel1D plain_kernel = [](BufferVar<int> output, Int input) noexcept {
    output.write(0u, input * 3 + 1);
};

// Same body; the `native_include` block forces the compatibility
// HLSL-to-SPIR-V route, which is the only route that has a legacy per-shader
// PSO seed.
[[nodiscard]] ShaderOption legacy_route_option() {
    ShaderOption option{.enable_cache = true};
    option.native_include = R"(
uint lc_pso_cache_test_marker(uint value) { return value; }
)";
    return option;
}

template<typename Shader>
[[nodiscard]] int execute(Device &device, const Shader &shader,
                          int input) noexcept {
    auto output = device.create_buffer<int>(kRunCount);
    auto stream = device.create_stream();
    auto result = 0;
    stream << shader(output, input).dispatch(kRunCount)
           << output.copy_to(luisa::span{&result, 1})
           << synchronize();
    return result;
}

[[nodiscard]] std::string query(Device &device, luisa::string_view key) {
    auto value = device.query(key);
    return {value.data(), value.size()};
}

[[nodiscard]] std::string find_legacy_seed_name(
    std::vector<std::string> const &names) {
    for (auto const &name : names) {
        if (name.ends_with(".vk")) { return name; }
    }
    return {};
}

}// namespace

int main(int argc, char *argv[]) {
    // Boost.UT's string_view comparisons select overloaded logical operators,
    // which do not short-circuit the argc/null guards around argv access.
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr ||
        std::strcmp(argv[1], "vk") != 0) {
        LUISA_INFO(
            "Usage: {} vk",
            argc > 0 && argv != nullptr ? argv[0] : "test_vk_shader_cache");
        return 2;
    }

    MemoryBinaryIO binary_io;
    Context context{argv[0]};
    DeviceConfig config{.binary_io = &binary_io};

    auto global_artifact_name = std::string{};
    auto legacy_seed_name = std::string{};
    auto legacy_seed_payload = std::vector<std::byte>{};
    auto cold_device_cache_handle = std::string{};

    "vk_pso_cache_is_device_scoped_and_persists_once"_test = [&] {
        // ---- phase 1: cold device -------------------------------------
        auto device = context.create_device("vk", &config);
        // Device-owned helper shaders may use the same BinaryIO. Begin the
        // contract at the user-compute boundary after device initialization.
        binary_io.reset();

        expect(query(device, "vk_pipeline_cache_state") == "ready")
            << "the Device must own a usable VkPipelineCache (P1.3)";
        auto handle = query(device, "vk_pipeline_cache_handle");
        expect(!handle.empty() && handle != "0")
            << "the device cache must expose a live handle (P1.3)";
        cold_device_cache_handle = handle;
        expect(query(device, "vk_pipeline_cache_seed") == "absent")
            << "a cold cache directory has nothing to seed from";
        global_artifact_name = query(device, "vk_pipeline_cache_artifact_name");
        expect(!global_artifact_name.empty())
            << "the device cache must report the one artifact it reads/writes";
        expect(binary_io.find(global_artifact_name) == nullptr)
            << "the artifact does not exist before the first teardown";

        // Every Shader/Buffer/Stream handle keeps the Device alive, so the whole
        // compile/dispatch part of this phase lives in its own scope: the device
        // can only be torn down (and the cache written) once they are gone.
        {
            ShaderOption option{.enable_cache = true};

            // ---- SPIR-V cache: one artifact, no per-shader PSO artifact ----
            auto writes_before_cold = binary_io.cache_write_count;
            {
                auto cold = device.compile(plain_kernel, option);
                expect(execute(device, cold, 7) == 22);
            }
            expect(binary_io.entry_count_with_suffix(".spv") == 1u)
                << "the cold compile must persist the SPIR-V artifact";
            expect(binary_io.entry_count_with_suffix(".vk") == 0u)
                << "P1.3: a compile must not write a per-shader PSO artifact";
            expect(binary_io.cache_write_count == writes_before_cold + 1u)
                << "exactly one artifact (the SPIR-V module) is written";
            expect(query(device, "vk_pipeline_cache_seed") == "absent")
                << "a compile must not create a cache of its own";

            auto writes_after_cold = binary_io.cache_write_count;
            {
                auto hot = device.compile(plain_kernel, option);
                expect(execute(device, hot, 11) == 34);
            }
            expect(binary_io.cache_write_count == writes_after_cold)
                << "a valid hot SPIR-V artifact must not be rewritten";
            expect(binary_io.entry_count_with_suffix(".vk") == 0u)
                << "a hot compile must not add a per-shader PSO artifact either";

            // ---- ownership: a shader release never destroys the device cache
            {
                auto transient = device.compile(plain_kernel, option);
                expect(execute(device, transient, 13) == 40);
            }// ~Shader here releases the backend shader (and its pipeline ref)
            expect(query(device, "vk_pipeline_cache_handle") ==
                   cold_device_cache_handle)
                << "releasing a shader must not destroy or re-create the device "
                   "cache (P1.3 ownership)";
            expect(query(device, "vk_pipeline_cache_state") == "ready");
            {
                auto after_release = device.compile(plain_kernel, option);
                expect(execute(device, after_release, 17) == 52)
                    << "the device cache must still be usable after a shader "
                       "release";
            }

            // ---- driver policy: same cache, different pipeline ----
            auto writes_before_fast = binary_io.cache_write_count;
            {
                auto fast_option = option;
                fast_option.enable_driver_optimization = false;
                auto cold_fast = device.compile(plain_kernel, fast_option);
                expect(execute(device, cold_fast, 19) == 58);
            }
            expect(binary_io.entry_count_with_suffix(".spv") == 1u)
                << "driver policy must reuse the identical SPIR-V artifact";
            expect(binary_io.entry_count_with_suffix(".vk") == 0u)
                << "the alternate driver policy must not add a PSO artifact";
            expect(binary_io.cache_write_count == writes_before_fast)
                << "the alternate driver policy must not write anything: the "
                   "pipeline went into the device cache";

            // ---- legacy route: compiles still only use the device cache ----
            auto spv_before_legacy = binary_io.entry_count_with_suffix(".spv");
            auto writes_before_legacy = binary_io.cache_write_count;
            {
                auto legacy_option = legacy_route_option();
                auto legacy_cold = device.compile(plain_kernel, legacy_option);
                expect(execute(device, legacy_cold, 23) == 70);
            }
            expect(binary_io.entry_count_with_suffix(".spv") ==
                   spv_before_legacy + 1u)
                << "the HLSL route persists its own SPIR-V artifact";
            expect(binary_io.entry_count_with_suffix(".vk") == 0u)
                << "the HLSL route must not write a per-shader PSO artifact "
                   "either";
            expect(binary_io.cache_write_count == writes_before_legacy + 1u)
                << "only the SPIR-V artifact of the HLSL route is written";

            // The second (hot) HLSL compile is the one that consults the legacy
            // seed name: the cold one returned before that lookup because the
            // SPIR-V artifact it needs did not exist yet.
            binary_io.clear_counters();
            {
                auto legacy_option = legacy_route_option();
                auto legacy_hot = device.compile(plain_kernel, legacy_option);
                expect(execute(device, legacy_hot, 29) == 88);
            }
            legacy_seed_name = find_legacy_seed_name(binary_io.read_names);
            expect(!legacy_seed_name.empty())
                << "a hot HLSL-route compile must look up its legacy per-shader "
                   "PSO seed name";
            expect(binary_io.find(legacy_seed_name) == nullptr)
                << "the legacy seed must not exist when nothing ever wrote one";
            expect(query(device, "vk_pipeline_cache_merged_seeds") == "0")
                << "no merge happens while no legacy seed exists";
        }

        // ---- teardown: exactly one global artifact ----------------------
        auto writes_before_teardown = binary_io.cache_write_count;
        device = Device{};
        expect(binary_io.cache_write_count == writes_before_teardown + 1u)
            << "device teardown writes the device cache exactly once";
        expect(binary_io.has(global_artifact_name))
            << "the device cache must be persisted under its own artifact name";
        auto artifact = binary_io.find(global_artifact_name);
        expect(artifact != nullptr);
        auto payload = artifact == nullptr ?
                           std::optional<std::vector<std::byte>>{} :
                           unframe_pso_artifact(*artifact);
        expect(payload.has_value())
            << "the global artifact must use the same validated PSO framing";
        if (payload) {
            legacy_seed_payload = *payload;
            expect(legacy_seed_payload.size() > kCacheHeaderByteSize)
                << "a device that compiled several pipelines must persist more "
                   "than the bare cache header";
        }
        expect(binary_io.entry_count_with_suffix(".vk") == 0u)
            << "no per-shader PSO artifact may exist anywhere in this flow";
        expect(binary_io.entry_count_with_suffix(".bin") == 1u)
            << "exactly one device-cache artifact exists";
    };

    "vk_pso_cache_warm_start_and_legacy_seed_merge"_test = [&] {
        auto device = context.create_device("vk", &config);
        expect(query(device, "vk_pipeline_cache_seed") == "loaded")
            << "the second Device must be seeded from the artifact the first one "
               "wrote (P1.3 cold -> warm)";
        expect(query(device, "vk_pipeline_cache_seed_bytes") != "0")
            << "a loaded seed must report a non-zero payload size";
        expect(query(device, "vk_pipeline_cache_handle") != "0")
            << "a seeded Device still owns a live cache handle";

        {
            // The SPIR-V artifacts and the device cache are both warm now: a
            // recompile writes nothing and must still dispatch correctly.
            ShaderOption option{.enable_cache = true};
            binary_io.clear_counters();
            {
                auto warm = device.compile(plain_kernel, option);
                expect(execute(device, warm, 31) == 94);
            }
            expect(binary_io.cache_write_count == 0u)
                << "a fully warm compile must not write any artifact";
            expect(query(device, "vk_pipeline_cache_merged_seeds") == "0")
                << "no legacy seed was present yet";

            // ---- legacy per-shader seed: add-only input, merged -----------
            expect(!legacy_seed_name.empty() && !legacy_seed_payload.empty())
                << "the cold phase must have produced a legacy seed target";
            if (!legacy_seed_name.empty() && !legacy_seed_payload.empty()) {
                binary_io.put(legacy_seed_name,
                              frame_pso_artifact(legacy_seed_payload));
                {
                    auto legacy_option = legacy_route_option();
                    auto seeded = device.compile(plain_kernel, legacy_option);
                    expect(execute(device, seeded, 37) == 112);
                }
                expect(query(device, "vk_pipeline_cache_merged_seeds") == "1")
                    << "a valid legacy per-shader blob must be merged into the "
                       "device cache (P1.3 add-only seed)";
                expect(query(device, "vk_pipeline_cache_rejected_seeds") == "0")
                    << "a valid legacy seed must not be counted as rejected";
                expect(binary_io.entry_count_with_suffix(".vk") == 1u)
                    << "the legacy file stays where it was: seeds are never "
                       "rewritten";
                expect(binary_io.has(legacy_seed_name))
                    << "the legacy seed must not be deleted";

                // A corrupt seed of the same name is rejected, not merged, and it
                // must not damage the cache or the dispatch path.
                auto corrupt = frame_pso_artifact(legacy_seed_payload);
                corrupt.resize(corrupt.size() - 4u);// framing no longer adds up
                binary_io.put(legacy_seed_name, std::move(corrupt));
                {
                    auto legacy_option = legacy_route_option();
                    auto rejected = device.compile(plain_kernel, legacy_option);
                    expect(execute(device, rejected, 41) == 124);
                }
                expect(query(device, "vk_pipeline_cache_merged_seeds") == "1")
                    << "a rejected legacy seed must not be merged";
                expect(query(device, "vk_pipeline_cache_rejected_seeds") == "1")
                    << "a corrupt legacy seed must be reported as rejected";
                expect(binary_io.cache_write_count == 0u)
                    << "no artifact may be written while replaying warm compiles";
                // Restore a good legacy seed so the next phase starts from a
                // known file set.
                binary_io.put(legacy_seed_name,
                              frame_pso_artifact(legacy_seed_payload));
            }
        }
        device = Device{};
    };

    "vk_pso_cache_rejects_malformed_global_artifact"_test = [&] {
        // ---- truncated global artifact --------------------------------
        auto artifact = binary_io.find(global_artifact_name);
        expect(artifact != nullptr)
            << "the warm phase must have left the global artifact in place";
        std::vector<std::byte> truncated;
        if (artifact != nullptr) {
            truncated = *artifact;
            truncated.resize(truncated.size() - 8u);
        }
        binary_io.put(global_artifact_name, truncated);
        {
            auto device = context.create_device("vk", &config);
            expect(query(device, "vk_pipeline_cache_seed") == "rejected")
                << "a truncated global artifact must be rejected (P1.3 fail "
                   "closed)";
            expect(query(device, "vk_pipeline_cache_seed_reason") == "bad-framing")
                << "the rejection reason must name the failed check";
            expect(query(device, "vk_pipeline_cache_state") == "ready")
                << "a rejected seed must still leave a valid (empty) cache";
            // The SPIR-V bytecode cache is untouched by a rejected PSO blob.
            ShaderOption option{.enable_cache = true};
            binary_io.clear_counters();
            auto shader = device.compile(plain_kernel, option);
            expect(execute(device, shader, 43) == 130)
                << "dispatch must keep working after a rejected PSO artifact";
            expect(binary_io.cache_write_count == 0u)
                << "the rejected PSO artifact must not invalidate the SPIR-V "
                   "bytecode cache";
            device = Device{};
        }

        // ---- foreign-device global artifact ---------------------------
        // Valid framing and a valid payload MD5, but the
        // VkPipelineCacheHeaderVersionOne does not belong to this device, so
        // Device::is_pso_same must reject it.
        if (!legacy_seed_payload.empty()) {
            auto foreign = legacy_seed_payload;
            // deviceID lives at offset 12 of the cache header.
            foreign[12] = static_cast<std::byte>(
                static_cast<uint8_t>(foreign[12]) ^ 0xFFu);
            binary_io.put(global_artifact_name,
                          frame_pso_artifact(foreign));
            auto device = context.create_device("vk", &config);
            expect(query(device, "vk_pipeline_cache_seed") == "rejected")
                << "a foreign-device payload must be rejected";
            expect(query(device, "vk_pipeline_cache_seed_reason") ==
                   "foreign-device")
                << "the rejection reason must name the identity check";
            auto legacy_option = legacy_route_option();
            binary_io.clear_counters();
            auto shader = device.compile(plain_kernel, legacy_option);
            expect(execute(device, shader, 47) == 142)
                << "the HLSL route must still work with a rejected PSO artifact";
            // The legacy per-shader seed is still valid (it was taken from this
            // device's own cache) and must still be merged.
            expect(query(device, "vk_pipeline_cache_merged_seeds") == "1")
                << "a rejected global artifact must not disable the legacy seed "
                   "path";
            device = Device{};
        }
    };
}