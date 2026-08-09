// Vulkan native XIR-to-SPIR-V code-generation path tests.
// This test covers JIT/AOT routing, shader identity, SSA/control-flow structure,
// arithmetic edge cases, word-storage ABI, callable interfaces, ray metadata,
// and Vulkan resource synchronization boundaries.

#include "ut/ut.hpp"
#include "test_device.h"

#include <volk.h>

#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/backends/ext/raster_ext.hpp>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/verifier.h>
#include "indirect_dispatch_layout.h"
#include "spirv_codegen/entry.h"
#include "spirv_codegen/utils.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct FourBools {
    bool4 values;
};
LUISA_STRUCT(FourBools, values) {};
static_assert(sizeof(FourBools) == 4u);

struct WordBackedMixedComposite {
    uint32_t tag;
    bool4 flags;
    uint32_t payload;
};
LUISA_STRUCT(WordBackedMixedComposite, tag, flags, payload) {};
static_assert(sizeof(WordBackedMixedComposite) == 12u);

struct WordBackedSignedAtomicComposite {
    bool4 flags;
    int32_t values[2];
};
LUISA_STRUCT(WordBackedSignedAtomicComposite, flags, values) {};
static_assert(sizeof(WordBackedSignedAtomicComposite) == 12u);

struct NestedConstantLeaf {
    uint32_t code;
    int32_t offsets[2];
};
LUISA_STRUCT(NestedConstantLeaf, code, offsets) {};
static_assert(sizeof(NestedConstantLeaf) == 12u);

struct NestedConstantRecord {
    NestedConstantLeaf leaves[2];
    uint32_t order[3];
    float scale;
};
LUISA_STRUCT(NestedConstantRecord, leaves, order, scale) {};
static_assert(sizeof(NestedConstantRecord) == 40u);

struct WideVectorStorageRecord {
    float4 prefix;
    double4 payload;
    uint32_t suffix;
};
LUISA_STRUCT(WideVectorStorageRecord, prefix, payload, suffix) {};
static_assert(offsetof(WideVectorStorageRecord, payload) == 16u);
static_assert(sizeof(WideVectorStorageRecord) == 64u);

struct NestedMatrixStorageRecord {
    float4 prefix;
    float2x2 transforms[2];
    uint32_t suffix;
};
LUISA_STRUCT(NestedMatrixStorageRecord, prefix, transforms, suffix) {};
static_assert(offsetof(NestedMatrixStorageRecord, transforms) == 16u);
static_assert(offsetof(NestedMatrixStorageRecord, suffix) == 48u);
static_assert(sizeof(NestedMatrixStorageRecord) == 64u);

struct SpirvCallableAggregate {
    float2 pair;
    uint32_t tag;
    float weight;
};
LUISA_STRUCT(SpirvCallableAggregate, pair, tag, weight) {};
static_assert(sizeof(SpirvCallableAggregate) == 16u);

namespace {

void set_environment_variable(const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class NativeCommandVolkConfigExt : public VulkanDeviceConfigExt {
public:
    void init_volk(PFN_vkGetInstanceProcAddr handler) noexcept override {
        volkInitializeCustom(handler);
    }

    void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice,
        VkDevice,
        VkAllocationCallbacks *,
        VkPipelineCacheHeaderVersionOne const &,
        VkQueue,
        VkQueue,
        VkQueue,
        uint32_t,
        uint32_t,
        uint32_t,
        IDxcCompiler3 *,
        IDxcLibrary *,
        IDxcUtils *) noexcept override {
        // volkInitializeCustom() deliberately loads only loader-level entry
        // points. The private Volk copy linked into this test needs the exact
        // backend instance before volkLoadDeviceTable() can resolve commands
        // through vkGetDeviceProcAddr.
        volkLoadInstanceOnly(instance);
        LUISA_ASSERT(vkGetDeviceProcAddr != nullptr,
                     "Failed to load vkGetDeviceProcAddr for Vulkan custom "
                     "command tests.");
    }
};

[[nodiscard]] luisa::test::DeviceContext create_native_command_device(
    int argc, char *argv[]) {
    LUISA_ASSERT(argc > 1 && argv != nullptr && argv[0] != nullptr &&
                     argv[1] != nullptr && std::string_view{argv[1]} == "vk",
                 "Vulkan custom-command tests require the vk backend.");
    Context context{argv[0]};
    DeviceConfig config{};
    config.extension = luisa::make_unique<NativeCommandVolkConfigExt>();
    auto device = context.create_device("vk", &config);
    luisa::test::log_test_backend("vk", device);
    return {std::move(context), std::move(device)};
}

class BindlessBufferFillCommand final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;
    VkBuffer _target;
    VkDeviceSize _size;
    uint32_t _value;

public:
    BindlessBufferFillCommand(
        BindlessArray const &array,
        Buffer<uint32_t> const &target,
        uint32_t value) noexcept
        : _target{std::bit_cast<VkBuffer>(target.native_handle())},
          _size{target.size_bytes()}, _value{value} {
        _usages.emplace_back(
            Argument::BindlessArray{array.handle()},
            ResourceUsageType::CopyDest);
    }

    [[nodiscard]] luisa::span<ResourceUsage>
    get_resource_usages() noexcept override {
        return _usages;
    }

    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }

    void execute(
        VkPhysicalDevice, VkDevice device, VkQueue,
        VkCommandBuffer command_buffer,
        VkDescriptorPool) const noexcept override {
        VolkDeviceTable table{};
        volkLoadDeviceTable(&table, device);
        LUISA_ASSERT(table.vkCmdFillBuffer != nullptr,
                     "vkCmdFillBuffer is unavailable.");
        table.vkCmdFillBuffer(
            command_buffer, _target, 0u, _size, _value);
    }
};

class BindlessTextureCopyCommand final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;
    VkImage _source;
    VkBuffer _target;
    VkExtent3D _extent;

public:
    BindlessTextureCopyCommand(
        BindlessArray const &array,
        Image<float> const &source,
        Buffer<float4> const &target) noexcept
        : _source{std::bit_cast<VkImage>(source.native_handle())},
          _target{std::bit_cast<VkBuffer>(target.native_handle())},
          _extent{source.size().x, source.size().y, 1u} {
        _usages.emplace_back(
            Argument::BindlessArray{array.handle()},
            ResourceUsageType::CopySource);
        _usages.emplace_back(
            Argument::Buffer{
                target.handle(), 0u, target.size_bytes()},
            ResourceUsageType::CopyDest);
    }

    [[nodiscard]] luisa::span<ResourceUsage>
    get_resource_usages() noexcept override {
        return _usages;
    }

    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }

    void execute(
        VkPhysicalDevice, VkDevice device, VkQueue,
        VkCommandBuffer command_buffer,
        VkDescriptorPool) const noexcept override {
        VolkDeviceTable table{};
        volkLoadDeviceTable(&table, device);
        LUISA_ASSERT(table.vkCmdCopyImageToBuffer != nullptr,
                     "vkCmdCopyImageToBuffer is unavailable.");
        auto region = VkBufferImageCopy{
            .bufferOffset = 0u,
            .bufferRowLength = 0u,
            .bufferImageHeight = 0u,
            .imageSubresource = VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0u,
                .baseArrayLayer = 0u,
                .layerCount = 1u},
            .imageOffset = VkOffset3D{0, 0, 0},
            .imageExtent = _extent};
        table.vkCmdCopyImageToBuffer(
            command_buffer, _source,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            _target, 1u, &region);
    }
};

class BindlessConfigStateExt final : public NativeCommandVolkConfigExt {
private:
    luisa::vector<VKCustomCmd::ResourceUsage> _before_states;
    luisa::vector<VKCustomCmd::ResourceUsage> _after_states;

public:
    void clear_states() noexcept {
        _before_states.clear();
        _after_states.clear();
    }

    void set_before_state(
        BindlessArray const &array,
        VkPipelineStageFlagBits2 stage,
        VkAccessFlagBits2 access,
        VkImageLayout layout) {
        _before_states.clear();
        _before_states.emplace_back(
            Argument::BindlessArray{array.handle()},
            stage, access, layout);
    }

    void set_after_state(
        BindlessArray const &array,
        VKCustomCmd::ResourceUsageType type) {
        _after_states.clear();
        _after_states.emplace_back(
            Argument::BindlessArray{array.handle()}, type);
    }

    [[nodiscard]] luisa::span<VKCustomCmd::ResourceUsage const>
    before_states(uint64_t) noexcept override {
        return _before_states;
    }

    [[nodiscard]] luisa::span<VKCustomCmd::ResourceUsage const>
    after_states(uint64_t) noexcept override {
        return _after_states;
    }
};

class ConfiguredTextureMipCopyCommand final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;
    VkImage _source;
    VkBuffer _target;
    uint32_t _mip_level;
    VkExtent3D _extent;

public:
    ConfiguredTextureMipCopyCommand(
        Image<float> const &source,
        Buffer<float4> const &target,
        uint32_t mip_level) noexcept
        : _source{std::bit_cast<VkImage>(source.native_handle())},
          _target{std::bit_cast<VkBuffer>(target.native_handle())},
          _mip_level{mip_level},
          _extent{
              std::max(source.size().x >> mip_level, 1u),
              std::max(source.size().y >> mip_level, 1u), 1u} {
        _usages.emplace_back(
            Argument::Buffer{
                target.handle(), 0u, target.size_bytes()},
            ResourceUsageType::CopyDest);
    }

    [[nodiscard]] luisa::span<ResourceUsage>
    get_resource_usages() noexcept override {
        return _usages;
    }

    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }

    void execute(
        VkPhysicalDevice, VkDevice device, VkQueue,
        VkCommandBuffer command_buffer,
        VkDescriptorPool) const noexcept override {
        VolkDeviceTable table{};
        volkLoadDeviceTable(&table, device);
        LUISA_ASSERT(table.vkCmdCopyImageToBuffer != nullptr,
                     "vkCmdCopyImageToBuffer is unavailable.");
        auto region = VkBufferImageCopy{
            .bufferOffset = 0u,
            .bufferRowLength = 0u,
            .bufferImageHeight = 0u,
            .imageSubresource = VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = _mip_level,
                .baseArrayLayer = 0u,
                .layerCount = 1u},
            .imageOffset = VkOffset3D{0, 0, 0},
            .imageExtent = _extent};
        // The source layout is the public after_states contract being tested.
        // This command intentionally declares only its direct destination.
        table.vkCmdCopyImageToBuffer(
            command_buffer, _source,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            _target, 1u, &region);
    }
};

class ConfiguredTextureMipOverwriteCommand final : public VKCustomCmd {
private:
    luisa::vector<ResourceUsage> _usages;
    VkBuffer _source;
    VkImage _target;
    uint32_t _mip_levels;
    uint32_t _target_mip;
    VkExtent3D _target_extent;

public:
    ConfiguredTextureMipOverwriteCommand(
        Buffer<float4> const &source,
        Image<float> const &target,
        uint32_t target_mip) noexcept
        : _source{std::bit_cast<VkBuffer>(source.native_handle())},
          _target{std::bit_cast<VkImage>(target.native_handle())},
          _mip_levels{target.mip_levels()},
          _target_mip{target_mip},
          _target_extent{
              std::max(target.size().x >> target_mip, 1u),
              std::max(target.size().y >> target_mip, 1u), 1u} {
        _usages.emplace_back(
            Argument::Buffer{
                source.handle(), 0u, source.size_bytes()},
            ResourceUsageType::CopySource);
    }

    [[nodiscard]] luisa::span<ResourceUsage>
    get_resource_usages() noexcept override {
        return _usages;
    }

    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }

    [[nodiscard]] uint3 max_dispatch_size() const noexcept override {
        return make_uint3(1u);
    }

    void execute(
        VkPhysicalDevice, VkDevice device, VkQueue,
        VkCommandBuffer command_buffer,
        VkDescriptorPool) const noexcept override {
        VolkDeviceTable table{};
        volkLoadDeviceTable(&table, device);
        LUISA_ASSERT(table.vkCmdPipelineBarrier2 != nullptr,
                     "vkCmdPipelineBarrier2 is unavailable.");
        LUISA_ASSERT(table.vkCmdCopyBufferToImage != nullptr,
                     "vkCmdCopyBufferToImage is unavailable.");
        auto image_barrier = VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = _target,
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0u,
                .levelCount = _mip_levels,
                .baseArrayLayer = 0u,
                .layerCount = 1u}};
        auto dependency = VkDependencyInfo{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1u,
            .pImageMemoryBarriers = &image_barrier};
        table.vkCmdPipelineBarrier2(command_buffer, &dependency);
        auto region = VkBufferImageCopy{
            .bufferOffset = 0u,
            .bufferRowLength = 0u,
            .bufferImageHeight = 0u,
            .imageSubresource = VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = _target_mip,
                .baseArrayLayer = 0u,
                .layerCount = 1u},
            .imageOffset = VkOffset3D{0, 0, 0},
            .imageExtent = _target_extent};
        // This models externally managed native work: the command performs
        // its own transition, and before_states publishes the resulting state
        // back to Luisa on the following submission.
        table.vkCmdCopyBufferToImage(
            command_buffer, _source, _target,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1u, &region);
    }
};

[[nodiscard]] auto dump_exists(std::string_view name) noexcept {
    std::error_code ec;
    return std::filesystem::exists(std::filesystem::path{name}, ec);
}

[[nodiscard]] auto read_text_file(const std::filesystem::path &path) {
    std::ifstream stream{path};
    return std::string{std::istreambuf_iterator<char>{stream},
                       std::istreambuf_iterator<char>{}};
}

[[nodiscard]] size_t count_substring(std::string_view text,
                                     std::string_view needle) noexcept {
    if (needle.empty()) { return 0u; }
    auto count = size_t{0u};
    for (auto offset = size_t{0u};;) {
        auto position = text.find(needle, offset);
        if (position == std::string_view::npos) { break; }
        count++;
        offset = position + needle.size();
    }
    return count;
}

[[nodiscard]] bool is_spirv_opcode_token(std::string_view token,
                                         std::string_view opcode) noexcept {
    if (token.compare(opcode) == 0) { return true; }
    if (token.size() != opcode.size() + 2u) { return false; }
    if (!token.starts_with("Op")) { return false; }
    return token.substr(2u).compare(opcode) == 0;
}

[[nodiscard]] size_t count_spirv_opcode(std::string_view disassembly,
                                        std::string_view opcode) noexcept {
    auto count = size_t{0u};
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        auto next_token = [&line](size_t &token_begin) noexcept {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            auto token = line.substr(token_begin, token_end - token_begin);
            token_begin = token_end;
            return token;
        };
        auto token_begin = size_t{0u};
        auto instruction_opcode = next_token(token_begin);
        if (instruction_opcode.ends_with(':')) {
            // glslang's readable dump uses either "result-id: Opcode ..."
            // for type declarations or "result-id: result-type Opcode ..."
            // for ordinary value-producing instructions.
            instruction_opcode = next_token(token_begin);
            if (!is_spirv_opcode_token(instruction_opcode, opcode)) {
                instruction_opcode = next_token(token_begin);
            }
        } else {
            auto saved_token_begin = token_begin;
            auto second_token = next_token(token_begin);
            if (second_token == "=") {
                // spirv-dis uses "%result-id = OpOpcode ...".
                instruction_opcode = next_token(token_begin);
            } else {
                token_begin = saved_token_begin;
            }
        }
        if (is_spirv_opcode_token(instruction_opcode, opcode)) {
            count++;
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    return count;
}

[[nodiscard]] size_t count_spirv_extended_instruction(
    std::string_view disassembly,
    std::string_view instruction) noexcept {
    auto count = size_t{0u};
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        if (count_spirv_opcode(line, "ExtInst") == 1u) {
            for (auto token_begin = size_t{0u}; token_begin < line.size();) {
                while (token_begin < line.size() &&
                       (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                    token_begin++;
                }
                auto token_end = token_begin;
                while (token_end < line.size() &&
                       line[token_end] != ' ' && line[token_end] != '\t') {
                    token_end++;
                }
                auto token = line.substr(token_begin, token_end - token_begin);
                auto parenthesized = token.size() == instruction.size() + 2u &&
                                     token.front() == '(' && token.back() == ')' &&
                                     token.substr(1u, instruction.size()) == instruction;
                if (token == instruction || parenthesized) {
                    count++;
                    break;
                }
                auto open = token.find('(');
                if (open != std::string_view::npos && token.ends_with(")") &&
                    token.substr(open + 1u,
                                 token.size() - open - 2u) == instruction) {
                    count++;
                    break;
                }
                token_begin = token_end;
            }
        }
        line_begin = line_end + (line_end < disassembly.size());
    }
    return count;
}

[[nodiscard]] bool spirv_opcode_has_operand(
    std::string_view disassembly, std::string_view opcode,
    std::string_view operand) noexcept {
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        auto found_opcode = false;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            auto token =
                line.substr(token_begin, token_end - token_begin);
            if (found_opcode && token == operand) { return true; }
            found_opcode |= is_spirv_opcode_token(token, opcode);
            token_begin = token_end;
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    return false;
}

[[nodiscard]] bool spirv_opcode_has_adjacent_operands(
    std::string_view disassembly, std::string_view opcode,
    std::string_view first, std::string_view second) noexcept {
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        for (auto i = size_t{0u}; i + 2u < tokens.size(); ++i) {
            if (is_spirv_opcode_token(tokens[i], opcode)) {
                for (auto j = i + 1u; j + 1u < tokens.size(); ++j) {
                    if (tokens[j] == first && tokens[j + 1u] == second) {
                        return true;
                    }
                }
                break;
            }
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    return false;
}

[[nodiscard]] std::optional<std::string>
spirv_unsigned_64_type_token(std::string_view disassembly) {
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        for (auto i = size_t{0u}; i + 2u < tokens.size(); ++i) {
            if (!is_spirv_opcode_token(tokens[i], "TypeInt")) { continue; }
            if (tokens[i + 1u].compare("64") != 0) { continue; }
            if (tokens[i + 2u].compare("0") != 0) { continue; }
            if (i >= 2u) {
                if (tokens[i - 1u].compare("=") == 0) {
                    return std::string{tokens[i - 2u]};
                }
            }
            if (i >= 1u) {
                if (tokens[i - 1u].ends_with(':')) {
                    auto token = std::string{tokens[i - 1u]};
                    token.pop_back();
                    return token;
                }
            }
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    return std::nullopt;
}

[[nodiscard]] std::string_view normalize_spirv_id_token(
    std::string_view token) noexcept {
    if (token.starts_with('%')) { token.remove_prefix(1u); }
    if (auto parenthesis = token.find('(');
        parenthesis != std::string_view::npos) {
        token = token.substr(0u, parenthesis);
    }
    return token;
}

[[nodiscard]] std::optional<std::string> spirv_id_named(
    std::string_view disassembly, std::string_view name) {
    auto quoted_name = luisa::format("\"{}\"", name);
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        for (auto i = size_t{0u}; i + 2u < tokens.size(); ++i) {
            if (is_spirv_opcode_token(tokens[i], "Name") &&
                tokens[i + 2u] == quoted_name) {
                return std::string{
                    normalize_spirv_id_token(tokens[i + 1u])};
            }
        }
        line_begin = line_end +
                     (line_end < disassembly.size() ? 1u : 0u);
    }
    return std::nullopt;
}

[[nodiscard]] bool spirv_id_has_decoration(
    std::string_view disassembly, std::string_view id,
    std::string_view decoration) noexcept {
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        for (auto i = size_t{0u}; i + 2u < tokens.size(); ++i) {
            if (is_spirv_opcode_token(tokens[i], "Decorate") &&
                normalize_spirv_id_token(tokens[i + 1u]) == id &&
                tokens[i + 2u] == decoration) {
                return true;
            }
        }
        line_begin = line_end +
                     (line_end < disassembly.size() ? 1u : 0u);
    }
    return false;
}

struct SpirvTextInstruction {
    std::string result;
    std::string result_type;
    std::vector<std::string> operands;
};

[[nodiscard]] std::string normalize_spirv_text_token(
    std::string_view token) {
    while (!token.empty() &&
           (token.back() == ',' || token.back() == ':')) {
        token.remove_suffix(1u);
    }
    return std::string{normalize_spirv_id_token(token)};
}

[[nodiscard]] std::vector<SpirvTextInstruction>
parse_spirv_text_instructions(
    std::string_view disassembly, std::string_view opcode) {
    std::vector<SpirvTextInstruction> instructions;
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto line = disassembly.substr(line_begin, line_end - line_begin);
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        for (auto i = size_t{0u}; i < tokens.size(); ++i) {
            if (!is_spirv_opcode_token(tokens[i], opcode)) { continue; }
            SpirvTextInstruction instruction;
            auto operand_begin = i + 1u;
            if (i >= 2u && tokens[i - 1u] == "=" &&
                i + 1u < tokens.size()) {
                instruction.result = normalize_spirv_text_token(
                    tokens[i - 2u]);
                instruction.result_type = normalize_spirv_text_token(
                    tokens[i + 1u]);
                operand_begin = i + 2u;
            } else if (i >= 2u &&
                       tokens[i - 2u].ends_with(':')) {
                instruction.result = normalize_spirv_text_token(
                    tokens[i - 2u]);
                instruction.result_type = normalize_spirv_text_token(
                    tokens[i - 1u]);
            }
            for (auto operand = operand_begin;
                 operand < tokens.size(); ++operand) {
                if (tokens[operand].starts_with(';')) { break; }
                instruction.operands.emplace_back(
                    normalize_spirv_text_token(tokens[operand]));
            }
            instructions.emplace_back(std::move(instruction));
            break;
        }
        line_begin = line_end +
                     (line_end < disassembly.size() ? 1u : 0u);
    }
    return instructions;
}

[[nodiscard]] bool spirv_u64_scaled_index_reaches_buffer_load(
    std::string_view disassembly,
    std::string_view uint64_type_token) {
    auto uint64_type = normalize_spirv_text_token(uint64_type_token);
    auto imuls = parse_spirv_text_instructions(disassembly, "IMul");
    auto iadds = parse_spirv_text_instructions(disassembly, "IAdd");
    auto udivs = parse_spirv_text_instructions(disassembly, "UDiv");
    auto access_chains = parse_spirv_text_instructions(
        disassembly, "AccessChain");
    auto loads = parse_spirv_text_instructions(disassembly, "Load");
    auto has_operand = [](const SpirvTextInstruction &instruction,
                          std::string_view id) noexcept {
        return std::ranges::any_of(
            instruction.operands,
            [&](auto &&operand) noexcept { return operand == id; });
    };
    for (auto &&scale : imuls) {
        if (scale.result.empty() ||
            scale.result_type != uint64_type) {
            continue;
        }
        for (auto &&biased : iadds) {
            if (biased.result.empty() ||
                biased.result_type != uint64_type ||
                !has_operand(biased, scale.result)) {
                continue;
            }
            for (auto &&word_index : udivs) {
                if (word_index.result.empty() ||
                    word_index.result_type != uint64_type ||
                    !has_operand(word_index, biased.result)) {
                    continue;
                }
                for (auto &&chain : access_chains) {
                    if (chain.result.empty() ||
                        !has_operand(chain, word_index.result)) {
                        continue;
                    }
                    if (std::ranges::any_of(
                            loads, [&](auto &&load) noexcept {
                                return has_operand(load, chain.result);
                            })) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

[[nodiscard]] bool spirv_entry_point_lists_callable_builtins(
    std::string_view disassembly) {
    auto tokenize = [](std::string_view line) {
        std::vector<std::string_view> tokens;
        for (auto token_begin = size_t{0u}; token_begin < line.size();) {
            while (token_begin < line.size() &&
                   (line[token_begin] == ' ' || line[token_begin] == '\t')) {
                token_begin++;
            }
            auto token_end = token_begin;
            while (token_end < line.size() &&
                   line[token_end] != ' ' && line[token_end] != '\t') {
                token_end++;
            }
            if (token_end != token_begin) {
                tokens.emplace_back(
                    line.substr(token_begin, token_end - token_begin));
            }
            token_begin = token_end;
        }
        return tokens;
    };
    std::optional<std::string_view> local_invocation_id;
    std::optional<std::string_view> workgroup_id;
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto tokens = tokenize(
            disassembly.substr(line_begin, line_end - line_begin));
        for (auto i = size_t{0u}; i + 3u < tokens.size(); i++) {
            if (!is_spirv_opcode_token(tokens[i], "Decorate") ||
                tokens[i + 2u] != "BuiltIn") {
                continue;
            }
            auto id = normalize_spirv_id_token(tokens[i + 1u]);
            if (tokens[i + 3u] == "LocalInvocationId") {
                local_invocation_id = id;
            } else if (tokens[i + 3u] == "WorkgroupId") {
                workgroup_id = id;
            }
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    if (!local_invocation_id || !workgroup_id) { return false; }
    for (auto line_begin = size_t{0u}; line_begin < disassembly.size();) {
        auto line_end = disassembly.find('\n', line_begin);
        if (line_end == std::string_view::npos) {
            line_end = disassembly.size();
        }
        auto tokens = tokenize(
            disassembly.substr(line_begin, line_end - line_begin));
        for (auto i = size_t{0u}; i < tokens.size(); i++) {
            if (!is_spirv_opcode_token(tokens[i], "EntryPoint")) { continue; }
            auto lists_local = false;
            auto lists_workgroup = false;
            for (auto j = i + 1u; j < tokens.size(); j++) {
                auto id = normalize_spirv_id_token(tokens[j]);
                lists_local |= id == *local_invocation_id;
                lists_workgroup |= id == *workgroup_id;
            }
            if (lists_local && lists_workgroup) { return true; }
        }
        line_begin = line_end + (line_end < disassembly.size() ? 1u : 0u);
    }
    return false;
}

struct SpirvU64SwitchCase {
    std::array<uint32_t, 2u> literal_words{};
    uint32_t target{0u};
};

struct SpirvU64SwitchShape {
    uint32_t selector{0u};
    uint32_t default_target{0u};
    std::vector<SpirvU64SwitchCase> cases;
    bool targets_are_labels{false};
};

[[nodiscard]] std::vector<SpirvU64SwitchShape>
inspect_spirv_u64_switches(
    luisa::span<const uint32_t> words) noexcept {
    std::unordered_map<uint32_t, uint32_t> integer_type_widths;
    std::unordered_map<uint32_t, uint32_t> value_types;
    std::unordered_set<uint32_t> label_ids;
    if (words.size() < 5u) { return {}; }
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (opcode == spv::Op::OpTypeInt && word_count == 4u) {
            integer_type_widths.emplace(
                words[offset + 1u], words[offset + 2u]);
        } else if (opcode == spv::Op::OpLabel && word_count == 2u) {
            label_ids.emplace(words[offset + 1u]);
        }
        auto has_result = false;
        auto has_result_type = false;
        spv::HasResultAndType(
            opcode, &has_result, &has_result_type);
        if (has_result && has_result_type && word_count >= 3u) {
            value_types.emplace(
                words[offset + 2u], words[offset + 1u]);
        }
        offset += word_count;
    }

    std::vector<SpirvU64SwitchShape> switches;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (opcode == spv::Op::OpSwitch && word_count >= 3u) {
            auto selector = words[offset + 1u];
            auto value_type = value_types.find(selector);
            auto type_width = value_type == value_types.end() ?
                                  integer_type_widths.end() :
                                  integer_type_widths.find(value_type->second);
            if (type_width != integer_type_widths.end() &&
                type_width->second == 64u) {
                // A 64-bit OpSwitch case is exactly low word, high word,
                // target ID. Reject any shape that could be misread as the
                // one-word literal layout.
                if ((word_count - 3u) % 3u != 0u) { return {}; }
                SpirvU64SwitchShape shape{
                    .selector = selector,
                    .default_target = words[offset + 2u]};
                for (auto operand = size_t{3u};
                     operand + 2u < word_count; operand += 3u) {
                    shape.cases.emplace_back(SpirvU64SwitchCase{
                        .literal_words = {
                            words[offset + operand],
                            words[offset + operand + 1u]},
                        .target = words[offset + operand + 2u]});
                }
                shape.targets_are_labels =
                    label_ids.contains(shape.default_target);
                for (auto &&case_value : shape.cases) {
                    shape.targets_are_labels &=
                        label_ids.contains(case_value.target);
                }
                switches.emplace_back(std::move(shape));
            }
        }
        offset += word_count;
    }
    return switches;
}

[[nodiscard]] size_t count_spirv_binary_opcode(
    luisa::span<const uint32_t> words, spv::Op expected) noexcept {
    if (words.size() < 5u) { return 0u; }
    auto count = size_t{0u};
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return 0u;
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        count += opcode == expected;
        offset += word_count;
    }
    return count;
}

struct SpirvIndirectRecordGuardFacts {
    bool exact_capacity_dataflow{false};
    bool record_stores_are_control_dependent{false};
};

struct SpirvBinaryOperation {
    uint32_t result;
    uint32_t lhs;
    uint32_t rhs;
};

struct SpirvConditionalBranch {
    uint32_t condition;
    uint32_t true_target;
};

struct SpirvBinaryAccessChain {
    uint32_t result;
    std::vector<uint32_t> indices;
};

struct SpirvBinaryStore {
    uint32_t pointer;
    uint32_t block;
};

[[nodiscard]] SpirvIndirectRecordGuardFacts
inspect_spirv_indirect_record_guard(
    luisa::span<const uint32_t> words) noexcept {
    std::unordered_map<uint32_t, uint32_t> constants;
    std::vector<uint32_t> array_lengths;
    std::vector<SpirvBinaryOperation> subtracts;
    std::vector<SpirvBinaryOperation> divides;
    std::vector<SpirvBinaryOperation> less_than;
    std::vector<SpirvBinaryOperation> multiplies;
    std::vector<SpirvBinaryOperation> adds;
    std::vector<SpirvConditionalBranch> branches;
    std::vector<SpirvBinaryAccessChain> access_chains;
    std::vector<SpirvBinaryStore> stores;
    auto current_block = uint32_t{0u};
    if (words.size() < 5u) { return {}; }
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto op = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (op == spv::Op::OpConstant && word_count == 4u) {
            constants.emplace(words[offset + 2u],
                              words[offset + 3u]);
        } else if (op == spv::Op::OpArrayLength &&
                   word_count == 5u) {
            array_lengths.emplace_back(words[offset + 2u]);
        } else if ((op == spv::Op::OpISub ||
                    op == spv::Op::OpUDiv ||
                    op == spv::Op::OpULessThan ||
                    op == spv::Op::OpIMul ||
                    op == spv::Op::OpIAdd) &&
                   word_count == 5u) {
            auto operation = SpirvBinaryOperation{
                .result = words[offset + 2u],
                .lhs = words[offset + 3u],
                .rhs = words[offset + 4u]};
            if (op == spv::Op::OpISub) {
                subtracts.emplace_back(operation);
            } else if (op == spv::Op::OpUDiv) {
                divides.emplace_back(operation);
            } else if (op == spv::Op::OpULessThan) {
                less_than.emplace_back(operation);
            } else if (op == spv::Op::OpIMul) {
                multiplies.emplace_back(operation);
            } else {
                adds.emplace_back(operation);
            }
        } else if (op == spv::Op::OpBranchConditional &&
                   word_count >= 4u) {
            branches.emplace_back(SpirvConditionalBranch{
                .condition = words[offset + 1u],
                .true_target = words[offset + 2u]});
        } else if (op == spv::Op::OpLabel && word_count == 2u) {
            current_block = words[offset + 1u];
        } else if ((op == spv::Op::OpAccessChain ||
                    op == spv::Op::OpInBoundsAccessChain) &&
                   word_count >= 5u) {
            SpirvBinaryAccessChain chain{
                .result = words[offset + 2u]};
            chain.indices.reserve(word_count - 4u);
            for (auto operand = size_t{4u};
                 operand < word_count; ++operand) {
                chain.indices.emplace_back(words[offset + operand]);
            }
            access_chains.emplace_back(std::move(chain));
        } else if (op == spv::Op::OpStore &&
                   word_count >= 3u && current_block != 0u) {
            stores.emplace_back(SpirvBinaryStore{
                .pointer = words[offset + 1u],
                .block = current_block});
        }
        offset += word_count;
    }

    auto constant_is = [&](uint32_t id, uint32_t value) noexcept {
        auto iter = constants.find(id);
        return iter != constants.end() && iter->second == value;
    };
    SpirvIndirectRecordGuardFacts facts;
    for (auto array_length : array_lengths) {
        for (auto &&subtract : subtracts) {
            if (subtract.lhs != array_length ||
                !constant_is(
                    subtract.rhs,
                    lc::IndirectDispatchLayout::header_word_count)) {
                continue;
            }
            for (auto &&divide : divides) {
                if (divide.lhs != subtract.result ||
                    !constant_is(
                        divide.rhs,
                        lc::IndirectDispatchLayout::record_word_count)) {
                    continue;
                }
                for (auto &&comparison : less_than) {
                    if (comparison.rhs != divide.result) { continue; }
                    for (auto &&branch : branches) {
                        if (branch.condition != comparison.result) {
                            continue;
                        }
                        facts.exact_capacity_dataflow = true;
                        for (auto &&multiply : multiplies) {
                            auto multiplies_index =
                                multiply.lhs == comparison.lhs &&
                                constant_is(
                                    multiply.rhs,
                                    lc::IndirectDispatchLayout::record_word_count);
                            multiplies_index |=
                                multiply.rhs == comparison.lhs &&
                                constant_is(
                                    multiply.lhs,
                                    lc::IndirectDispatchLayout::record_word_count);
                            if (!multiplies_index) { continue; }
                            for (auto &&base_add : adds) {
                                auto adds_header =
                                    base_add.lhs == multiply.result &&
                                    constant_is(
                                        base_add.rhs,
                                        lc::IndirectDispatchLayout::header_word_count);
                                adds_header |=
                                    base_add.rhs == multiply.result &&
                                    constant_is(
                                        base_add.lhs,
                                        lc::IndirectDispatchLayout::header_word_count);
                                if (!adds_header) { continue; }

                                std::unordered_set<uint32_t> record_words{
                                    base_add.result};
                                auto changed = true;
                                while (changed) {
                                    changed = false;
                                    for (auto &&add : adds) {
                                        if (record_words.contains(add.result)) {
                                            continue;
                                        }
                                        if (record_words.contains(add.lhs) ||
                                            record_words.contains(add.rhs)) {
                                            record_words.emplace(add.result);
                                            changed = true;
                                        }
                                    }
                                }
                                std::unordered_set<uint32_t> record_pointers;
                                for (auto &&chain : access_chains) {
                                    if (std::ranges::any_of(
                                            chain.indices,
                                            [&](uint32_t index) noexcept {
                                                return record_words.contains(index);
                                            })) {
                                        record_pointers.emplace(chain.result);
                                    }
                                }
                                auto record_store_count = size_t{0u};
                                auto all_record_stores_guarded = true;
                                for (auto &&store : stores) {
                                    if (!record_pointers.contains(store.pointer)) {
                                        continue;
                                    }
                                    record_store_count++;
                                    all_record_stores_guarded &=
                                        store.block == branch.true_target;
                                }
                                facts.record_stores_are_control_dependent =
                                    all_record_stores_guarded &&
                                    record_store_count >=
                                        lc::IndirectDispatchLayout::record_word_count;
                                if (facts.record_stores_are_control_dependent) {
                                    return facts;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return facts;
}

[[nodiscard]] auto find_spirv_dumps() {
    std::vector<std::filesystem::path> dumps;
    std::error_code ec;
    for (auto iter = std::filesystem::directory_iterator{".", ec};
         !ec && iter != std::filesystem::directory_iterator{}; iter.increment(ec)) {
        if (!iter->is_regular_file(ec)) { continue; }
        auto filename = iter->path().filename().string();
        if (filename.starts_with("spv_code_") &&
            filename.ends_with(".spvasm") &&
            !filename.starts_with("spv_code_hlsl_") &&
            !filename.starts_with("spv_code_llvm_")) {
            dumps.emplace_back(iter->path());
        }
    }
    std::sort(dumps.begin(), dumps.end());
    return dumps;
}

[[nodiscard]] auto any_hlsl_dump_exists() {
    std::error_code ec;
    for (auto iter = std::filesystem::directory_iterator{".", ec};
         !ec && iter != std::filesystem::directory_iterator{}; iter.increment(ec)) {
        if (!iter->is_regular_file(ec)) { continue; }
        auto filename = iter->path().filename().string();
        if (filename.rfind("hlsl_output_", 0u) == 0u ||
            filename.rfind("spv_code_hlsl_", 0u) == 0u) {
            return true;
        }
    }
    return false;
}

void remove_hlsl_dumps() noexcept {
    std::error_code ec;
    for (auto iter = std::filesystem::directory_iterator{".", ec};
         !ec && iter != std::filesystem::directory_iterator{}; iter.increment(ec)) {
        if (!iter->is_regular_file(ec)) { continue; }
        auto filename = iter->path().filename().string();
        if (filename.rfind("hlsl_output_", 0u) == 0u ||
            filename.rfind("spv_code_hlsl_", 0u) == 0u) {
            std::filesystem::remove(iter->path(), ec);
        }
    }
}

void remove_dump(std::string_view name) noexcept {
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path{name}, ec);
}

struct ScopedCurrentPath {
    std::filesystem::path previous;
    explicit ScopedCurrentPath(const std::filesystem::path &path)
        : previous{std::filesystem::current_path()} {
        std::filesystem::current_path(path);
    }
    ~ScopedCurrentPath() noexcept {
        std::error_code ec;
        std::filesystem::current_path(previous, ec);
    }
};

struct ScopedDirectoryCleanup {
    std::filesystem::path path;
    ~ScopedDirectoryCleanup() noexcept {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct ScopedTemporaryCurrentPath {
    std::filesystem::path path;
    std::filesystem::path previous;

    explicit ScopedTemporaryCurrentPath(std::string_view prefix)
        : previous{std::filesystem::current_path()} {
        auto nonce = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        nonce ^= static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
        path = std::filesystem::temp_directory_path() /
               luisa::format("{}_{}", prefix, nonce);
        std::filesystem::create_directories(path);
        std::filesystem::current_path(path);
    }

    ~ScopedTemporaryCurrentPath() noexcept {
        std::error_code ec;
        std::filesystem::current_path(previous, ec);
        std::filesystem::remove_all(path, ec);
    }

    ScopedTemporaryCurrentPath(const ScopedTemporaryCurrentPath &) = delete;
    ScopedTemporaryCurrentPath &operator=(const ScopedTemporaryCurrentPath &) = delete;
};

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;
    explicit ScopedEnvironmentVariable(const char *env_name,
                                       const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(), previous ? previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;
};

struct ScopedSourceDump {
    ScopedEnvironmentVariable environment;
    explicit ScopedSourceDump(const char *value = "1")
        : environment{"LUISA_DUMP_SOURCE", value} {}
};

[[nodiscard]] int probe_failure(std::string_view message) noexcept {
    LUISA_WARNING("Vulkan SPIR-V child probe failed: {}", message);
    return 1;
}

[[nodiscard]] int run_xir_disable_optimization_probe(
    int argc, char *argv[]) {
    ScopedEnvironmentVariable disable_optimization{
        "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    ScopedSourceDump source_dump;
    ScopedTemporaryCurrentPath work_dir{
        "luisa_vk_spirv_opt_disabled_probe"};

    auto dc = luisa::test::create_device(argc, argv);
    auto input = dc.device.create_buffer<float>(2u);
    auto output = dc.device.create_buffer<float>(2u);
    auto stream = dc.device.create_stream();

    Callable differentiated = [](Float x) noexcept {
        Float y;
        $if (x > 0.0f) {
            y = x * x;
        }
        $else {
            y = x * x * x;
        };
        return y;
    };
    Kernel1D kernel = [&](BufferFloat in, BufferFloat out) noexcept {
        auto i = dispatch_x();
        auto x = in.read(i);
        $autodiff {
            requires_grad(x);
            auto y = differentiated(x);
            backward(y);
            out.write(i, grad(x));
        };
    };
    auto kernel_hash = kernel.function()->function().hash();
    auto shader = dc.device.compile(
        kernel, ShaderOption{.enable_cache = false,
                             .enable_fast_math = false});

    constexpr std::array source{-2.0f, 2.0f};
    std::array<float, 2u> result{};
    stream << input.copy_from(luisa::span{source})
           << shader(input, output).dispatch(2u)
           << output.copy_to(luisa::span{result})
           << synchronize();
    if (result != std::array{12.0f, 4.0f}) {
        return probe_failure("mandatory autodiff legalization changed results");
    }

    auto raw_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.xir", kernel_hash)};
    auto structured_opt_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.structured_opt.xir", kernel_hash)};
    auto pre_ad_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.pre_ad.xir", kernel_hash)};
    auto ad_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.ad.xir", kernel_hash)};
    auto norm_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.norm.xir", kernel_hash)};
    if (!std::filesystem::exists(raw_path) ||
        !std::filesystem::exists(pre_ad_path) ||
        !std::filesystem::exists(ad_path) ||
        !std::filesystem::exists(norm_path)) {
        return probe_failure(
            "mandatory raw/pre-AD/AD/final XIR dump stages were not all emitted");
    }
    if (std::filesystem::exists(structured_opt_path)) {
        return probe_failure(
            "optional structured optimization ran while explicitly disabled");
    }
    auto raw_xir = read_text_file(raw_path);
    auto ad_xir = read_text_file(ad_path);
    auto normalized_xir = read_text_file(norm_path);
    if (raw_xir.find("autodiff_scope") == std::string::npos) {
        return probe_failure("raw XIR did not preserve the source autodiff scope");
    }
    if (ad_xir.find("autodiff_scope") != std::string::npos ||
        normalized_xir.find("autodiff_scope") != std::string::npos) {
        return probe_failure(
            "mandatory autodiff lowering left an autodiff scope at codegen handoff");
    }
    if (normalized_xir.find("reg2mem_spill") != std::string::npos) {
        return probe_failure(
            "final autodiff XIR retained typed reg2mem spill provenance");
    }
    if (normalized_xir.find("alloca to lower phi node") !=
            std::string::npos ||
        normalized_xir.find("load from phi alloca") !=
            std::string::npos ||
        normalized_xir.find("alloca to lower cross-block value") !=
            std::string::npos ||
        normalized_xir.find("load from cross-block alloca") !=
            std::string::npos) {
        return probe_failure(
            "final autodiff XIR retained a reg2mem spill artifact");
    }
    return 0;
}

[[nodiscard]] int run_callable_builtin_interface_probe(
    int argc, char *argv[]) {
    ScopedEnvironmentVariable disable_optimization{
        "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    ScopedSourceDump source_dump;
    ScopedTemporaryCurrentPath work_dir{
        "luisa_vk_spirv_callable_builtin_probe"};

    auto dc = luisa::test::create_device(argc, argv);
    constexpr auto invocation_count = 64u;
    auto output = dc.device.create_buffer<uint32_t>(invocation_count);
    auto stream = dc.device.create_stream();

    Callable callable_global_lane = []() noexcept {
        return block_x() * 32u + thread_x();
    };
    callable_global_lane.set_name("callable_only_builtin_lane");
    Kernel1D kernel = [&](BufferUInt out) noexcept {
        set_block_size(32u, 1u, 1u);
        auto lane = callable_global_lane();
        out.write(lane, lane * 9u + 2u);
    };
    auto normalized_xir_path = std::filesystem::path{luisa::format(
        "kernel.{:016x}.norm.xir",
        kernel.function()->function().hash())};
    auto shader = dc.device.compile(
        kernel, ShaderOption{.enable_cache = false,
                             .enable_fast_math = false});

    std::array<uint32_t, invocation_count> result{};
    stream << shader(output).dispatch(invocation_count)
           << output.copy_to(luisa::span{result})
           << synchronize();
    for (auto i = 0u; i < result.size(); i++) {
        if (result[i] != i * 9u + 2u) {
            return probe_failure(luisa::format(
                "callable-only builtin mismatch at lane {}", i));
        }
    }
    if (!std::filesystem::exists(normalized_xir_path)) {
        return probe_failure(
            "callable-only builtin probe did not emit normalized XIR");
    }
    auto normalized_xir = read_text_file(normalized_xir_path);
    if (count_substring(normalized_xir, "callable ") != 1u) {
        return probe_failure(
            "opt-disabled legalization did not retain exactly one non-resource callable");
    }

    auto dumps = find_spirv_dumps();
    if (dumps.size() != 1u) {
        return probe_failure(luisa::format(
            "expected one native SPIR-V dump, found {}", dumps.size()));
    }
    auto disassembly = read_text_file(dumps.front());
    auto function_call_count =
        count_spirv_opcode(disassembly, "FunctionCall");
    if (function_call_count != 1u) {
        return probe_failure(luisa::format(
            "opt0 SPIR-V expected one retained OpFunctionCall, found {}",
            function_call_count));
    }
    if (!spirv_entry_point_lists_callable_builtins(disassembly)) {
        return probe_failure(
            "callable-only LocalInvocationId/WorkgroupId globals were not both listed by OpEntryPoint");
    }
    if (disassembly.find("VariablePointersStorageBuffer") !=
            std::string::npos ||
        disassembly.find("SPV_KHR_variable_pointers") !=
            std::string::npos) {
        return probe_failure(
            "non-resource callable unexpectedly requested variable-pointer support");
    }
    return 0;
}

[[nodiscard]] int run_nested_callable_buffer_subview_probe(
    int argc, char *argv[]) {
    ScopedEnvironmentVariable disable_optimization{
        "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    ScopedSourceDump source_dump;
    ScopedTemporaryCurrentPath work_dir{
        "luisa_vk_spirv_nested_callable_subview_probe"};

    constexpr auto total_count = 17u;
    constexpr auto view_offset = 3u;
    constexpr auto view_count = 7u;
    constexpr auto replacement = 0xdecafbadu;
    auto dc = luisa::test::create_device(argc, argv);
    auto values = dc.device.create_buffer<uint32_t>(total_count);
    auto observed = dc.device.create_buffer<uint32_t>(4u);
    auto stream = dc.device.create_stream();

    Callable inspect = [](BufferUInt slice) noexcept {
        auto logical_size = def<uint32_t>(
            luisa::compute::detail::FunctionBuilder::current()->call(
                Type::of<uint32_t>(), CallOp::BUFFER_SIZE,
                {slice.expression()}));
        auto first = slice.read(0u);
        auto last = slice.read(logical_size - 1u);
        slice.write(2u, 0xdecafbadu);
        return make_uint4(
            first, last, logical_size, slice.read(2u));
    };
    inspect.set_name("nested_subview_inspect");
    Callable relay = [&](BufferUInt slice) noexcept {
        return inspect(slice);
    };
    relay.set_name("nested_subview_relay");
    Kernel1D kernel = [&](BufferUInt slice, BufferUInt out) noexcept {
        auto result = relay(slice);
        out.write(0u, result.x);
        out.write(1u, result.y);
        out.write(2u, result.z);
        out.write(3u, result.w);
    };
    auto kernel_hash = kernel.function()->function().hash();
    auto raw_xir_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.xir", kernel_hash)};
    auto structured_opt_xir_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.structured_opt.xir", kernel_hash)};
    auto normalized_xir_path = std::filesystem::path{
        luisa::format("kernel.{:016x}.norm.xir", kernel_hash)};
    auto shader = dc.device.compile(
        kernel, ShaderOption{.enable_cache = false,
                             .enable_fast_math = false});

    std::array<uint32_t, total_count> source{};
    for (auto i = 0u; i < source.size(); ++i) {
        source[i] = 0x1000u + i * 17u;
    }
    auto expected_values = source;
    expected_values[view_offset + 2u] = replacement;
    const std::array expected_observed{
        source[view_offset],
        source[view_offset + view_count - 1u],
        view_count,
        replacement};
    std::array<uint32_t, 4u> result{};
    std::array<uint32_t, total_count> result_values{};
    stream << values.copy_from(luisa::span{source})
           << shader(values.view(view_offset, view_count), observed).dispatch(1u)
           << observed.copy_to(luisa::span{result})
           << values.copy_to(luisa::span{result_values})
           << synchronize();

    if (result != expected_observed) {
        return probe_failure(
            "nested callable changed the typed subview read/write/size contract");
    }
    if (result_values != expected_values) {
        return probe_failure(
            "nested callable write escaped the nonzero typed subview");
    }
    if (!std::filesystem::exists(raw_xir_path) ||
        !std::filesystem::exists(normalized_xir_path)) {
        return probe_failure(
            "nested callable subview probe did not emit raw and normalized XIR");
    }
    if (std::filesystem::exists(structured_opt_xir_path)) {
        return probe_failure(
            "optional structured optimization ran in the disabled probe");
    }
    auto raw_xir = read_text_file(raw_xir_path);
    auto normalized_xir = read_text_file(normalized_xir_path);
    if (count_substring(raw_xir, "callable ") != 2u) {
        return probe_failure(
            "nested callable subview fixture did not start with two callables");
    }
    if (normalized_xir.find("callable ") != std::string::npos) {
        return probe_failure(
            "mandatory legalization retained a resource callable");
    }
    if (any_hlsl_dump_exists()) {
        return probe_failure(
            "nested callable subview probe left the native SPIR-V path");
    }
    auto dumps = find_spirv_dumps();
    if (dumps.size() != 1u) {
        return probe_failure(luisa::format(
            "expected one nested callable SPIR-V dump, found {}",
            dumps.size()));
    }
    auto disassembly = read_text_file(dumps.front());
    if (count_spirv_opcode(disassembly, "FunctionCall") != 0u) {
        return probe_failure(
            "resource callable specialization left an OpFunctionCall");
    }
    return 0;
}

[[nodiscard]] int run_indirect_alias_rejection_probe(
    int argc, char *argv[], bool bindless_alias) {
    auto dc = luisa::test::create_device(argc, argv);
    auto &device = dc.device;
    auto stream = device.create_stream();
    auto commands = device.create_indirect_dispatch_buffer(1u);

    Kernel1D author = [](Var<IndirectDispatchBuffer> target) noexcept {
        target.set_dispatch_count(1u);
        target.set_kernel(
            0u, make_uint3(1u), make_uint3(1u), 0u);
    };
    auto author_shader = device.compile(
        author, ShaderOption{.enable_cache = false});

    if (bindless_alias) {
        auto alias = device.import_external_buffer<uint32_t>(
            commands.native_handle(),
            commands.size_bytes() / sizeof(uint32_t));
        auto heap = device.create_bindless_array(1u);
        heap.emplace_on_update(0u, alias);
        Kernel1D overwrite = [](BindlessVar bindless) noexcept {
            bindless.buffer<uint32_t>(0u).write(0u, 0xdeadbeefu);
        };
        auto overwrite_shader = device.compile(
            overwrite, ShaderOption{.enable_cache = false});
        stream << heap.update() << synchronize();
        // A correct backend rejects this command before recording target work.
        stream << author_shader(commands).dispatch(1u)
               << overwrite_shader(heap).dispatch(commands)
               << synchronize();
    } else {
        auto alias = device.import_external_buffer<uint32_t>(
            commands.native_handle(),
            commands.size_bytes() / sizeof(uint32_t));
        Kernel1D overwrite = [](BufferUInt target) noexcept {
            target.write(0u, 0xdeadbeefu);
        };
        auto overwrite_shader = device.compile(
            overwrite, ShaderOption{.enable_cache = false});
        // A distinct Luisa wrapper around the same VkBuffer is still an alias.
        stream << author_shader(commands).dispatch(1u)
               << overwrite_shader(alias).dispatch(commands)
               << synchronize();
    }
    return 0;
}

template<typename T, size_t N>
void expect_vector_equal(const Vector<T, N> &actual,
                         const Vector<T, N> &expected) noexcept {
    for (size_t i = 0u; i < N; i++) {
        expect(actual[i] == expected[i])
            << luisa::format("vector component {} mismatch", i);
    }
}

template<typename Scalar, typename Vector, bool test_log_exp = true>
void run_typed_float_constant_case(Device &device, double epsilon) {
    auto stream = device.create_stream();
    auto scalar_input = device.create_buffer<Scalar>(2u);
    auto vector_input = device.create_buffer<Vector>(2u);
    auto scalar_saturate_output = device.create_buffer<Scalar>(2u);
    auto vector_saturate_output = device.create_buffer<Vector>(2u);
    auto scalar_log_exp_output = device.create_buffer<Scalar>(2u);
    auto vector_log_exp_output = device.create_buffer<Vector>(2u);

    Kernel1D kernel = [](BufferVar<Scalar> scalar_in,
                         BufferVar<Vector> vector_in,
                         BufferVar<Scalar> scalar_saturate_out,
                         BufferVar<Vector> vector_saturate_out,
                         BufferVar<Scalar> scalar_log_exp_out,
                         BufferVar<Vector> vector_log_exp_out) noexcept {
        auto i = dispatch_x();
        auto scalar = scalar_in.read(i);
        auto vector = vector_in.read(i);
        scalar_saturate_out.write(i, saturate(scalar));
        vector_saturate_out.write(i, saturate(vector));
        if constexpr (test_log_exp) {
            auto quarter = cast<Scalar>(0.25f);
            scalar_log_exp_out.write(i, exp10(log10(abs(scalar) + quarter)));
            vector_log_exp_out.write(i, exp10(log10(abs(vector) + quarter)));
        } else {
            // Keep one kernel signature for f64 SATURATE coverage. Native SPIR-V
            // cannot legally emit GLSL.std.450 transcendental operations on f64.
            scalar_log_exp_out.write(i, scalar);
            vector_log_exp_out.write(i, vector);
        }
    };
    ShaderOption option{.enable_fast_math = false};
    auto shader = device.compile(kernel, option);

    std::array scalar_source{
        static_cast<Scalar>(-0.5),
        static_cast<Scalar>(1.5)};
    std::array vector_source{
        Vector{static_cast<Scalar>(-0.5), static_cast<Scalar>(0.25)},
        Vector{static_cast<Scalar>(1.5), static_cast<Scalar>(-2.0)}};
    std::array<Scalar, 2u> scalar_saturate_result{};
    std::array<Vector, 2u> vector_saturate_result{};
    std::array<Scalar, 2u> scalar_log_exp_result{};
    std::array<Vector, 2u> vector_log_exp_result{};
    stream << scalar_input.copy_from(luisa::span{scalar_source})
           << vector_input.copy_from(luisa::span{vector_source})
           << shader(scalar_input, vector_input,
                     scalar_saturate_output, vector_saturate_output,
                     scalar_log_exp_output, vector_log_exp_output)
                  .dispatch(2u)
           << scalar_saturate_output.copy_to(luisa::span{scalar_saturate_result})
           << vector_saturate_output.copy_to(luisa::span{vector_saturate_result})
           << scalar_log_exp_output.copy_to(luisa::span{scalar_log_exp_result})
           << vector_log_exp_output.copy_to(luisa::span{vector_log_exp_result})
           << synchronize();

    auto close = [epsilon](auto actual, double expected) noexcept {
        return std::abs(static_cast<double>(actual) - expected) <= epsilon;
    };
    for (auto i = 0u; i < scalar_source.size(); i++) {
        auto scalar_value = static_cast<double>(scalar_source[i]);
        auto scalar_saturate_expected = scalar_value < 0.0 ? 0.0 : std::min(scalar_value, 1.0);
        expect(close(scalar_saturate_result[i], scalar_saturate_expected));
        if constexpr (test_log_exp) {
            auto scalar_log_exp_expected = std::abs(scalar_value) + 0.25;
            expect(close(scalar_log_exp_result[i], scalar_log_exp_expected));
        }
        for (auto j = 0u; j < 2u; j++) {
            auto vector_value = static_cast<double>(vector_source[i][j]);
            auto vector_saturate_expected = vector_value < 0.0 ? 0.0 : std::min(vector_value, 1.0);
            expect(close(vector_saturate_result[i][j], vector_saturate_expected));
            if constexpr (test_log_exp) {
                auto vector_log_exp_expected = std::abs(vector_value) + 0.25;
                expect(close(vector_log_exp_result[i][j], vector_log_exp_expected));
            }
        }
    }
}

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LUISA_INFO("Usage: {} vk", argc > 0 ? argv[0] : "test_vk_spirv_codegen_path");
        return 2;
    }
    if (std::string_view{argv[1]} != "vk") {
        LUISA_INFO("Usage: {} vk", argc > 0 ? argv[0] : "test_vk_spirv_codegen_path");
        return 2;
    }
    if (argc >= 3) {
        auto probe = std::string_view{argv[2]};
        if (probe == "--xir-disable-optimization-probe") {
            return run_xir_disable_optimization_probe(argc, argv);
        }
        if (probe == "--callable-builtin-interface-probe") {
            return run_callable_builtin_interface_probe(argc, argv);
        }
        if (probe == "--nested-callable-buffer-subview-probe") {
            return run_nested_callable_buffer_subview_probe(argc, argv);
        }
        if (probe == "--indirect-native-alias-probe") {
            return run_indirect_alias_rejection_probe(argc, argv, false);
        }
        if (probe == "--indirect-bindless-alias-probe") {
            return run_indirect_alias_rejection_probe(argc, argv, true);
        }
    }
    std::vector<const char *> ut_argv;
    ut_argv.reserve(static_cast<size_t>(argc));
    ut_argv.emplace_back(argv[0]);
    for (auto i = 2; i < argc; i++) { ut_argv.emplace_back(argv[i]); }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        static_cast<int>(ut_argv.size()), ut_argv.data());

    "vk_spirv_opcode_token_boundary_guards"_test = [] {
        expect(!is_spirv_opcode_token("", "Phi"));
        expect(!is_spirv_opcode_token("O", "Phi"));
        expect(!is_spirv_opcode_token("Op", "Phi"));
        expect(is_spirv_opcode_token("Phi", "Phi"));
        expect(is_spirv_opcode_token("OpPhi", "Phi"));
        expect(!is_spirv_opcode_token("XpPhi", "Phi"));
        constexpr auto disassembly = R"(
OpCapability GroupNonUniformShuffle
%1 = OpGroupNonUniformShuffle %uint %scope %value %lane
2(result): 3(type) GroupNonUniformShuffle 4 5 6
)";
        expect(count_spirv_opcode(
                   disassembly, "GroupNonUniformShuffle") == 2u)
            << "capability operands must not be counted as instruction opcodes";
        constexpr auto extended_disassembly = R"(
%1 = OpExtInstImport "GLSL.std.450"
%2 = OpExtInst %float %1 Fma %a %b %c
3: 4(float) ExtInst 1(GLSL.std.450) 50(Fma) 5 6 7
OpName %8 "Fma"
)";
        expect(count_spirv_extended_instruction(
                   extended_disassembly, "Fma") == 2u)
            << "extended instructions must be counted only on OpExtInst lines";
    };

    auto executable_path = std::filesystem::absolute(argv[0]).string();
    argv[0] = executable_path.data();
    auto process_work_dir = std::filesystem::temp_directory_path() /
                            luisa::format("luisa_vk_spirv_codegen_path_process_{}",
                                          std::filesystem::path{argv[0]}.filename().string());
    std::error_code process_work_dir_ec;
    std::filesystem::remove_all(process_work_dir, process_work_dir_ec);
    std::filesystem::create_directories(process_work_dir);
    ScopedDirectoryCleanup process_work_dir_cleanup{process_work_dir};
    ScopedCurrentPath process_work_path{process_work_dir};

    "vk_user_compute_dumps_spirv_not_hlsl"_test = [&] {
        constexpr std::string_view hlsl_dump = "hlsl_output_vk_spirv_codegen_path.hlsl";
        constexpr std::string_view spv_dump = "spv_code_vk_spirv_codegen_path.spvasm";

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_codegen_path_{}", std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedEnvironmentVariable require_native{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);

        auto buffer = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, 42u);
        };
        ShaderOption option{.name = "vk_spirv_codegen_path"};
        uint32_t value = 0u;
        {
            ScopedEnvironmentVariable disable_source_dump{
                "LUISA_DUMP_SOURCE", nullptr};
            static_cast<void>(dc.device.compile(kernel, option));
            auto cached_shader =
                dc.device.load_shader<1, Buffer<uint32_t>>(option.name);
            stream << cached_shader(buffer).dispatch(1u)
                   << buffer.copy_to(luisa::span{&value, 1u})
                   << synchronize();
            expect(value == 42u)
                << "the first named native SPIR-V artifact must be loadable";
        }
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);
        ScopedSourceDump scoped_source_dump;
        auto shader = dc.device.compile(kernel, option);

        value = 0u;
        stream << shader(buffer).dispatch(1u)
               << buffer.copy_to(luisa::span{&value, 1u})
               << synchronize();
        expect(value == 42u);

        expect(!dump_exists(hlsl_dump)) << "Vulkan user compute must not dump HLSL";
        expect(!any_hlsl_dump_exists()) << "Vulkan user compute must not emit any HLSL-derived dumps";
        expect(dump_exists(spv_dump)) << "Vulkan user compute should dump native SPIR-V when LUISA_DUMP_SOURCE=1";
        if (dump_exists(spv_dump)) {
            auto disassembly = read_text_file(spv_dump);
            expect(count_spirv_opcode(disassembly, "EntryPoint") == 1u)
                << "the forced native source dump must be a complete SPIR-V module";
        }
    };

    "vk_user_compute_outlines_unique_readonly_resources_after_cfg_destructure"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_structured_callable_{}",
                                      std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;
        remove_hlsl_dumps();

        auto input = dc.device.create_buffer<uint32_t>(4u);
        auto bindless_input = dc.device.create_buffer<uint32_t>(4u);
        auto heap = dc.device.create_bindless_array(1u);
        auto output = dc.device.create_buffer<uint32_t>(4u);
        auto stream = dc.device.create_stream();
        Callable classify_buffer = [](BufferUInt source,
                                      UInt index) noexcept {
            auto value = source.read(index);
            UInt result;
            $if ((value & 1u) == 0u) {
                result = value * 3u + 1u;
            }
            $else {
                result = value + 7u;
            };
            return result;
        };
        Callable classify_bindless = [](BindlessVar bindless,
                                        UInt index) noexcept {
            auto value = bindless.buffer<uint32_t>(0u).read(index);
            UInt result;
            $if ((value & 1u) == 0u) {
                result = value * 3u + 1u;
            }
            $else {
                result = value + 7u;
            };
            return result;
        };
        Kernel1D kernel = [&](BufferUInt source,
                              BindlessVar bindless,
                              BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, classify_buffer(source, i) +
                             classify_bindless(bindless, i));
        };
        auto normalized_xir_dump = luisa::format(
            "kernel.{:016x}.norm.xir", kernel.function()->function().hash());
        auto structured_opt_xir_dump = luisa::format(
            "kernel.{:016x}.structured_opt.xir",
            kernel.function()->function().hash());
        ShaderOption option{.enable_cache = false};
        auto shader = dc.device.compile(kernel, option);

        constexpr std::array source{0u, 1u, 2u, 3u};
        constexpr std::array bindless_source{4u, 5u, 6u, 7u};
        std::array<uint32_t, 4u> result{};
        stream << input.copy_from(luisa::span{source})
               << bindless_input.copy_from(luisa::span{bindless_source})
               << heap.emplace_on_update(0u, bindless_input).update()
               << shader(input, heap, output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{14u, 20u, 26u, 24u};
        expect(result == expected)
            << "both branches of the structured callable should execute deterministically";

        // A bindless update may compile a backend-owned HLSL helper while
        // source dumping is enabled. The normalized XIR and native SPIR-V
        // artifacts below are the precise oracle for this user kernel.
        expect(dump_exists(normalized_xir_dump)) << "Vulkan structured callable should dump normalized XIR";
        expect(!dump_exists(structured_opt_xir_dump))
            << "resource specialization must not rely on optional XIR optimization";
        std::ifstream xir_stream{normalized_xir_dump.c_str()};
        auto normalized_xir = std::string{
            std::istreambuf_iterator<char>{xir_stream},
            std::istreambuf_iterator<char>{}};
        expect(count_substring(normalized_xir, "callable ") == 2u)
            << "uniquely rooted read-only buffer and bindless callables "
               "should remain outlined after CFG destructuring";
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "Vulkan structured callable should dump exactly one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "FunctionCall") == 2u)
                << "each uniquely rooted read-only resource callable must "
                   "survive as one OpFunctionCall";
            expect(disassembly.find("VariablePointersStorageBuffer") ==
                   std::string::npos)
                << "buffer/bindless callable specialization must not request VariablePointersStorageBuffer";
            expect(disassembly.find("SPV_KHR_variable_pointers") ==
                   std::string::npos)
                << "buffer/bindless callable specialization must not request SPV_KHR_variable_pointers";
        }
    };

    "vk_user_compute_outlined_readonly_buffer_uses_frozen_kernel_argument_layout"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_outlined_buffer_metadata_layout"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto backing = device.create_buffer<uint32_t>(8u);
        auto output = device.create_buffer<uint32_t>(4u);

        Callable inspect = [](BufferUInt source,
                              UInt index) noexcept {
            return source.read(index);
        };
        Kernel1D kernel = [&](BufferUInt source,
                              BufferUInt destination,
                              UInt salt) noexcept {
            auto i = dispatch_x();
            destination.write(i, inspect(source, i) + salt);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            1000u, 2000u, 11u, 22u, 33u, 44u, 3000u, 4000u};
        std::array<uint32_t, 4u> result{};
        stream << backing.copy_from(luisa::span{source})
               << shader(backing.view(2u, 4u), output, 7u)
                      .dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        constexpr std::array expected{18u, 29u, 40u, 51u};
        expect(result == expected)
            << "an outlined read-only callable must use the kernel ABI's "
               "nonzero metadata offset for the direct-buffer subview bias";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "outlined direct-buffer metadata regression should emit one "
               "native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "FunctionCall") == 1u)
                << "the regression must cross a real outlined callable boundary";
        }
    };

    "vk_user_compute_autodiff_inlines_multiblock_callable_after_cfg_destructure"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_autodiff_callable_{}",
                                      std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;

        auto input = dc.device.create_buffer<float>(6u);
        auto selector = dc.device.create_buffer<uint32_t>(6u);
        auto output = dc.device.create_buffer<float>(6u);
        auto stream = dc.device.create_stream();
        Callable differentiated = [](Float x, UInt branch) noexcept {
            auto y = def(0.0f);
            $if (x > 0.0f) {
                y = x * x;
            }
            $else {
                y = x * x * x;
            };
            $switch (branch) {
                $case (0u) {
                    y = y + x;
                };
                $case (1u) {
                    y = y * 2.0f;
                };
                $default {
                    y = y - 3.0f * x;
                };
            };
            return y;
        };
        Kernel1D kernel = [&](BufferFloat in, BufferUInt branches, BufferFloat out) noexcept {
            auto i = dispatch_x();
            auto x = in.read(i);
            $autodiff {
                requires_grad(x);
                auto y = differentiated(x, branches.read(i));
                backward(y);
                out.write(i, grad(x));
            };
        };
        auto normalized_xir_dump = luisa::format(
            "kernel.{:016x}.norm.xir", kernel.function()->function().hash());
        ShaderOption option{.enable_fast_math = false,
                            .name = "vk_spirv_autodiff_callable"};
        auto shader = dc.device.compile(kernel, option);

        constexpr std::array input_values{-2.0f, -1.5f, -1.0f, 0.5f, 1.0f, 2.0f};
        constexpr std::array selector_values{0u, 1u, 2u, 0u, 1u, 2u};
        constexpr std::array expected{13.0f, 13.5f, 0.0f, 2.0f, 4.0f, 1.0f};
        std::array<float, 6u> result{};
        stream << input.copy_from(luisa::span{input_values})
               << selector.copy_from(luisa::span{selector_values})
               << shader(input, selector, output).dispatch(6u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        auto gradients_match = true;
        for (auto i = 0u; i < result.size(); i++) {
            gradients_match &= std::isfinite(result[i]) &&
                               std::abs(result[i] - expected[i]) < 1e-4f;
        }
        expect(gradients_match)
            << "autodiff should preserve the selected if/switch derivative after callable inlining";
        expect(dump_exists(normalized_xir_dump))
            << "Vulkan autodiff callable should dump normalized XIR";
        std::ifstream xir_stream{normalized_xir_dump.c_str()};
        auto normalized_xir = std::string{
            std::istreambuf_iterator<char>{xir_stream},
            std::istreambuf_iterator<char>{}};
        expect(normalized_xir.find("callable ") == std::string::npos)
            << "autodiff callable should be inlined after CFG destructuring";
        expect(normalized_xir.find("autodiff_scope") == std::string::npos)
            << "autodiff scope should be lowered before SPIR-V emission";
    };

    "vk_user_compute_autodiff_array_store_uses_logical_type"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<std::array<float, 1u>>(1u);
        auto output = dc.device.create_buffer<float>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferVar<std::array<float, 1u>> in,
                             BufferFloat out) noexcept {
            auto i = dispatch_x();
            auto p = in.read(i);
            $autodiff {
                requires_grad(p);
                ArrayFloat<1> scratch;
                scratch = p;
                auto used = scratch[0];
                auto loss = used * used;
                scratch[0] = p[0] * 7.0f;
                backward(loss);
                out.write(i, grad(p)[0]);
            };
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});
        constexpr std::array source{std::array<float, 1u>{3.0f}};
        std::array<float, 1u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 6.0f)
            << "aggregate autodiff stores should preserve the logical array type";
    };

    "vk_user_compute_aot_uses_spirv_not_hlsl"_test = [&] {
        constexpr std::string_view hlsl_dump = "hlsl_output_vk_spirv_codegen_path_aot.hlsl";
        constexpr std::string_view spv_dump = "spv_code_vk_spirv_codegen_path_aot.spvasm";

        auto dc = luisa::test::create_device(argc, argv);
        auto dump_dir = std::filesystem::temp_directory_path() /
                        luisa::format("luisa_vk_spirv_codegen_path_aot_{}", std::filesystem::path{argv[0]}.filename().string());
        std::error_code ec;
        std::filesystem::remove_all(dump_dir, ec);
        std::filesystem::create_directories(dump_dir);
        ScopedCurrentPath scoped_path{dump_dir};
        ScopedSourceDump scoped_source_dump;
        remove_hlsl_dumps();
        remove_dump(hlsl_dump);
        remove_dump(spv_dump);

        constexpr std::string_view shader_path = "vk_spirv_codegen_path_aot";
        Kernel1D kernel = [](BufferUInt output) noexcept {
            output.write(0u, 7u);
        };
        dc.device.compile_to(kernel, shader_path);

        auto buffer = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();
        ScopedEnvironmentVariable require_native{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
        auto shader = dc.device.load_shader<1, Buffer<uint32_t>>(shader_path);

        uint32_t value = 0u;
        stream << shader(buffer).dispatch(1u)
               << buffer.copy_to(luisa::span{&value, 1u})
               << synchronize();
        expect(value == 7u);

        expect(!dump_exists(hlsl_dump)) << "Vulkan AOT user compute must not dump HLSL";
        expect(!any_hlsl_dump_exists()) << "Vulkan AOT user compute must not emit any HLSL-derived dumps";
        expect(dump_exists(spv_dump)) << "Vulkan compile_to should dump native SPIR-V when LUISA_DUMP_SOURCE=1";
    };

    "vk_user_compute_same_shape_jit_shaders_do_not_alias"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto buffer = device.create_buffer<uint32_t>(512u);

        Kernel1D first = [](BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, i + 1u);
        };
        Kernel1D second = [](BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, (i + 1u) * 3u);
        };

        auto shader_a = device.compile(first);
        stream << shader_a(buffer).dispatch(512u) << synchronize();

        auto shader_b = device.compile(second);
        stream << shader_b(buffer).dispatch(512u) << synchronize();

        luisa::vector<uint32_t> host(512u);
        stream << buffer.copy_to(luisa::span{host}) << synchronize();
        auto ok = true;
        for (auto i = 0u; i < host.size(); i++) {
            auto expected = static_cast<uint32_t>((i + 1u) * 3u);
            if (host[i] != expected) {
                LUISA_WARNING("same-shape JIT shader alias mismatch at {}: got {}, expected {}",
                              i, host[i], expected);
                ok = false;
                break;
            }
        }
        expect(ok) << "Vulkan JIT compute shaders with the same default identity must not reuse stale pipelines";
    };

    "vk_user_compute_opt_disabled_still_runs_mandatory_legalization"_test = [&] {
        ScopedEnvironmentVariable disable_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto command = luisa::format(
            "\"{}\" vk --xir-disable-optimization-probe",
            executable_path);
        auto status = std::system(command.c_str());
        expect(status == 0)
            << "opt-disabled fresh process must lower autodiff and structured callable CFG into valid executable SPIR-V";
    };

    "vk_user_compute_opt_disabled_specializes_nested_sliced_buffer_callables"_test = [&] {
        auto command = luisa::format(
            "\"{}\" vk --nested-callable-buffer-subview-probe",
            executable_path);
        auto status = std::system(command.c_str());
        expect(status == 0)
            << "opt-disabled nested resource callables must preserve typed subview bias and size metadata";
    };

    "vk_user_compute_spirv_optimizer_levels_validate_and_execute"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<uint32_t>(4u);
        auto output = dc.device.create_buffer<uint32_t>(4u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt in, BufferUInt out) noexcept {
            auto i = dispatch_x();
            auto x = in.read(i);
            UInt value = x * 3u + 1u;
            $if ((x & 1u) == 0u) {
                value ^= 0x55u;
            }
            $else {
                value += 7u;
            };
            out.write(i, value);
        };
        constexpr std::array source{0u, 1u, 2u, 17u};
        constexpr std::array expected{84u, 11u, 82u, 59u};
        stream << input.copy_from(luisa::span{source}) << synchronize();

        for (auto level : {"0", "1", "2"}) {
            ScopedEnvironmentVariable optimization_level{
                "LUISA_SPIRV_OPT_LEVEL", level};
            auto shader = dc.device.compile(
                kernel, ShaderOption{.enable_cache = false});
            std::array<uint32_t, 4u> result{};
            stream << shader(input, output).dispatch(4u)
                   << output.copy_to(luisa::span{result})
                   << synchronize();
            expect(result == expected)
                << luisa::format(
                       "SPIR-V optimizer level {} changed exact kernel semantics",
                       level);
        }
    };

    "vk_user_compute_spirv_level_zero_preserves_loop_phi"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        Kernel1D kernel = [](BufferUInt in,
                             BufferUInt count,
                             BufferUInt out) noexcept {
            auto i = dispatch_x();
            UInt step = 0u;
            UInt value = in.read(i);
            $while (step < count.read(i)) {
                value = value * 3u + step + i;
                step += 1u;
            };
            out.write(i, value);
        };
        auto run_case = [&](const char *disable_xir_optimization,
                            luisa::string_view label) {
            ScopedEnvironmentVariable xir_optimization{
                "LUISA_XIR_DISABLE_OPTIMIZATION",
                disable_xir_optimization};
            ScopedTemporaryCurrentPath work_dir{luisa::format(
                "luisa_vk_spirv_loop_phi_{}", label)};
            ScopedSourceDump source_dump;

            auto dc = luisa::test::create_device(argc, argv);
            auto input = dc.device.create_buffer<uint32_t>(4u);
            auto trip_count = dc.device.create_buffer<uint32_t>(4u);
            auto output = dc.device.create_buffer<uint32_t>(4u);
            auto stream = dc.device.create_stream();
            auto normalized_xir_path = std::filesystem::path{luisa::format(
                "kernel.{:016x}.norm.xir",
                kernel.function()->function().hash())};
            auto shader = dc.device.compile(
                kernel, ShaderOption{.enable_cache = false,
                                     .enable_fast_math = false});

            constexpr std::array source{1u, 2u, 3u, 4u};
            constexpr std::array counts{0u, 1u, 3u, 5u};
            constexpr std::array expected{1u, 7u, 112u, 1393u};
            std::array<uint32_t, 4u> result{};
            stream << input.copy_from(luisa::span{source})
                   << trip_count.copy_from(luisa::span{counts})
                   << shader(input, trip_count, output).dispatch(4u)
                   << output.copy_to(luisa::span{result})
                   << synchronize();
            expect(result == expected)
                << luisa::format(
                       "level-zero SPIR-V must execute loop-carried SSA "
                       "exactly with XIR optimization {}",
                       label);
            auto dumps = find_spirv_dumps();
            expect(dumps.size() == 1u)
                << luisa::format(
                       "level-zero Phi regression should emit exactly one "
                       "native SPIR-V dump with XIR optimization {}",
                       label);
            expect(std::filesystem::exists(normalized_xir_path))
                << luisa::format(
                       "level-zero Phi regression should emit final "
                       "normalized XIR with optimization {}",
                       label);
            if (dumps.size() == 1u &&
                std::filesystem::exists(normalized_xir_path)) {
                auto disassembly = read_text_file(dumps.front());
                auto normalized_xir = read_text_file(normalized_xir_path);
                auto spirv_phi_count =
                    count_spirv_opcode(disassembly, "Phi");
                auto xir_phi_count =
                    count_substring(normalized_xir, " = phi");
                expect(spirv_phi_count >= 2u)
                    << luisa::format(
                           "opt0 must preserve at least the two source "
                           "loop-carried Phis; found {} (backend-owned "
                           "control metadata may add more)",
                           spirv_phi_count);
                expect(xir_phi_count == 2u)
                    << luisa::format(
                           "final XIR should contain exactly the two "
                           "loop-carried SSA Phi nodes; found {}",
                           xir_phi_count);
                expect(normalized_xir.find("reg2mem_spill") ==
                       std::string::npos)
                    << "final XIR must not retain typed reg2mem spill "
                       "provenance";
                expect(normalized_xir.find("alloca to lower phi node") ==
                       std::string::npos)
                    << "final XIR must not retain reg2mem Phi spill artifacts";
                expect(normalized_xir.find("load from phi alloca") ==
                       std::string::npos)
                    << "final XIR must not reload values from reg2mem Phi "
                       "spill slots";
                expect(normalized_xir.find(
                           "alloca to lower cross-block value") ==
                       std::string::npos)
                    << "final XIR must not retain cross-block spill artifacts";
                expect(normalized_xir.find("load from cross-block alloca") ==
                       std::string::npos)
                    << "final XIR must not reload values from cross-block "
                       "spill slots";
                expect(normalized_xir.find(" = alloca local") ==
                       std::string::npos)
                    << "this SSA-only fixture should not carry local allocas "
                       "into SPIR-V planning";
            }
        };
        run_case(nullptr, "enabled");
        run_case("1", "disabled");
    };

    "vk_user_compute_source_dump_is_dynamic"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_dynamic_source_dump"};
        ScopedSourceDump source_dump_disabled{nullptr};

        auto dc = luisa::test::create_device(argc, argv);
        auto output = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt out) noexcept {
            out.write(0u, 0x1234abcdu);
        };
        ShaderOption option{.enable_cache = false};
        auto compile_run_and_check = [&] {
            auto shader = dc.device.compile(kernel, option);
            uint32_t result = 0u;
            stream << shader(output).dispatch(1u)
                   << output.copy_to(luisa::span{&result, 1u})
                   << synchronize();
            expect(result == 0x1234abcdu);
        };

        compile_run_and_check();
        expect(find_spirv_dumps().empty())
            << "source dumping must stay disabled when the environment is absent";
        {
            ScopedSourceDump source_dump_enabled;
            compile_run_and_check();
            auto dumps = find_spirv_dumps();
            expect(dumps.size() == 1u)
                << "enabling source dumping must take effect on the next uncached compilation";
            for (auto &&dump : dumps) {
                std::error_code ec;
                std::filesystem::remove(dump, ec);
            }
        }
        compile_run_and_check();
        expect(find_spirv_dumps().empty())
            << "source dumping must stop immediately after the scoped environment is restored";
    };

    "vk_user_compute_repeated_compilation_and_device_destruction"_test = [&] {
        for (auto iteration = 0u; iteration < 4u; iteration++) {
            auto dc = luisa::test::create_device(argc, argv);
            auto output = dc.device.create_buffer<uint32_t>(8u);
            auto stream = dc.device.create_stream();
            auto bias = 101u + iteration * 37u;
            Kernel1D kernel = [bias](BufferUInt out) noexcept {
                auto i = dispatch_x();
                out.write(i, bias + i * 5u);
            };
            auto shader = dc.device.compile(
                kernel, ShaderOption{.enable_cache = false});
            std::array<uint32_t, 8u> result{};
            stream << shader(output).dispatch(8u)
                   << output.copy_to(luisa::span{result})
                   << synchronize();
            for (auto i = 0u; i < result.size(); i++) {
                expect(result[i] == bias + i * 5u)
                    << luisa::format(
                           "lifecycle iteration {}, lane {} returned stale code",
                           iteration, i);
            }
        }
    };

    "vk_user_compute_callable_only_builtins_enter_kernel_interface"_test = [&] {
        auto command = luisa::format(
            "\"{}\" vk --callable-builtin-interface-probe",
            executable_path);
        auto status = std::system(command.c_str());
        expect(status == 0)
            << "fresh opt0 child must retain OpFunctionCall and expose callable-only builtins through the entry-point interface";
    };

    "vk_user_compute_nested_constant_dynamic_index_uses_non_ubo_fallback"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_nested_constant"};
        ScopedSourceDump source_dump;

        constexpr std::array constant_source{
            NestedConstantRecord{
                {NestedConstantLeaf{101u, {-11, 12}},
                 NestedConstantLeaf{102u, {-13, 14}}},
                {1001u, 1002u, 1003u},
                0.25f},
            NestedConstantRecord{
                {NestedConstantLeaf{201u, {-21, 22}},
                 NestedConstantLeaf{202u, {-23, 24}}},
                {2001u, 2002u, 2003u},
                -0.5f},
            NestedConstantRecord{
                {NestedConstantLeaf{301u, {-31, 32}},
                 NestedConstantLeaf{302u, {-33, 34}}},
                {3001u, 3002u, 3003u},
                1.5f},
            NestedConstantRecord{
                {NestedConstantLeaf{401u, {-41, 42}},
                 NestedConstantLeaf{402u, {-43, 44}}},
                {4001u, 4002u, 4003u},
                -2.25f},
            NestedConstantRecord{
                {NestedConstantLeaf{501u, {-51, 52}},
                 NestedConstantLeaf{502u, {-53, 54}}},
                {5001u, 5002u, 5003u},
                3.75f},
            NestedConstantRecord{
                {NestedConstantLeaf{601u, {-61, 62}},
                 NestedConstantLeaf{602u, {-63, 64}}},
                {6001u, 6002u, 6003u},
                -4.5f}};
        constexpr std::array index_source{
            5u, 0u, 3u, 1u, 4u, 2u, 5u, 1u, 0u, 4u, 2u};
        constexpr auto dispatch_count = index_source.size();
        constexpr auto uint_fields_per_record = 5u;
        constexpr auto int_fields_per_record = 4u;

        auto dc = luisa::test::create_device(argc, argv);
        auto indices = dc.device.create_buffer<uint32_t>(dispatch_count);
        auto uint_output = dc.device.create_buffer<uint32_t>(
            dispatch_count * uint_fields_per_record);
        auto int_output = dc.device.create_buffer<int32_t>(
            dispatch_count * int_fields_per_record);
        auto scale_output = dc.device.create_buffer<float>(dispatch_count);
        auto selected_output = dc.device.create_buffer<int32_t>(dispatch_count);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [&](BufferUInt lookup,
                              BufferUInt uint_out,
                              BufferInt int_out,
                              BufferFloat scale_out,
                              BufferInt selected_out) noexcept {
            Constant<NestedConstantRecord> constants{constant_source};
            auto lane = dispatch_x();
            auto table_index = lookup.read(lane);
            auto record = constants[table_index];
            auto uint_base = lane * uint_fields_per_record;
            uint_out.write(uint_base + 0u, record.leaves[0u].code);
            uint_out.write(uint_base + 1u, record.leaves[1u].code);
            uint_out.write(uint_base + 2u, record.order[0u]);
            uint_out.write(uint_base + 3u, record.order[1u]);
            uint_out.write(uint_base + 4u, record.order[2u]);
            auto int_base = lane * int_fields_per_record;
            int_out.write(int_base + 0u, record.leaves[0u].offsets[0u]);
            int_out.write(int_base + 1u, record.leaves[0u].offsets[1u]);
            int_out.write(int_base + 2u, record.leaves[1u].offsets[0u]);
            int_out.write(int_base + 3u, record.leaves[1u].offsets[1u]);
            scale_out.write(lane, record.scale);
            auto leaf_index = (table_index + lane) & 1u;
            auto offset_index = (table_index * 3u + lane) & 1u;
            auto selected_leaf = record.leaves[leaf_index];
            selected_out.write(
                lane, selected_leaf.offsets[offset_index]);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t,
                   dispatch_count * uint_fields_per_record>
            uint_result{};
        std::array<int32_t,
                   dispatch_count * int_fields_per_record>
            int_result{};
        std::array<float, dispatch_count> scale_result{};
        std::array<int32_t, dispatch_count> selected_result{};
        stream << indices.copy_from(luisa::span{index_source})
               << shader(indices, uint_output, int_output,
                         scale_output, selected_output)
                      .dispatch(dispatch_count)
               << uint_output.copy_to(luisa::span{uint_result})
               << int_output.copy_to(luisa::span{int_result})
               << scale_output.copy_to(luisa::span{scale_result})
               << selected_output.copy_to(luisa::span{selected_result})
               << synchronize();

        for (auto lane = 0u; lane < dispatch_count; lane++) {
            auto table_index = index_source[lane];
            auto &&expected = constant_source[table_index];
            auto uint_base = lane * uint_fields_per_record;
            expect(uint_result[uint_base + 0u] == expected.leaves[0u].code);
            expect(uint_result[uint_base + 1u] == expected.leaves[1u].code);
            for (auto i = 0u; i < 3u; i++) {
                expect(uint_result[uint_base + 2u + i] == expected.order[i])
                    << luisa::format(
                           "nested constant order mismatch at lane {}, field {}",
                           lane, i);
            }
            auto int_base = lane * int_fields_per_record;
            for (auto leaf = 0u; leaf < 2u; leaf++) {
                for (auto offset = 0u; offset < 2u; offset++) {
                    auto flat = leaf * 2u + offset;
                    expect(int_result[int_base + flat] ==
                           expected.leaves[leaf].offsets[offset])
                        << luisa::format(
                               "nested constant offset mismatch at lane {}, leaf {}, field {}",
                               lane, leaf, offset);
                }
            }
            expect(scale_result[lane] == expected.scale)
                << luisa::format(
                       "nested constant scale mismatch at lane {}", lane);
            auto leaf_index = (table_index + lane) & 1u;
            auto offset_index = (table_index * 3u + lane) & 1u;
            expect(selected_result[lane] ==
                   expected.leaves[leaf_index].offsets[offset_index])
                << luisa::format(
                       "nested dynamic constant selection mismatch at lane {}",
                       lane);
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "nested constant fallback should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(!spirv_opcode_has_operand(
                disassembly, "Variable", "Uniform"))
                << "nested struct/array constants must not use the memcpy-backed std140 UBO path";
        }
    };

    "vk_user_compute_mat2_constant_array_uses_std140_matrix_stride"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_mat2_constant"};
        ScopedSourceDump source_dump;

        constexpr std::array constant_source{
            make_float2x2(1.0f, 2.0f, 3.0f, 4.0f),
            make_float2x2(-5.0f, 6.0f, 7.0f, -8.0f),
            make_float2x2(9.5f, -10.5f, 11.5f, -12.5f),
            make_float2x2(13.25f, 14.25f, -15.25f, -16.25f)};
        constexpr std::array index_source{
            3u, 0u, 2u, 1u, 3u, 1u, 0u, 2u};
        constexpr auto dispatch_count = index_source.size();

        auto dc = luisa::test::create_device(argc, argv);
        auto indices = dc.device.create_buffer<uint32_t>(dispatch_count);
        auto output = dc.device.create_buffer<float4>(dispatch_count);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [&](BufferUInt lookup,
                              BufferFloat4 result) noexcept {
            Constant<float2x2> constants{constant_source};
            auto lane = dispatch_x();
            auto matrix = constants[lookup.read(lane)];
            result.write(lane, make_float4(
                                   matrix[0u].x, matrix[0u].y,
                                   matrix[1u].x, matrix[1u].y));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<float4, dispatch_count> result{};
        stream << indices.copy_from(luisa::span{index_source})
               << shader(indices, output).dispatch(dispatch_count)
               << output.copy_to(luisa::span{result})
               << synchronize();

        for (auto lane = 0u; lane < dispatch_count; lane++) {
            auto &&expected = constant_source[index_source[lane]];
            for (auto column = 0u; column < 2u; column++) {
                for (auto row = 0u; row < 2u; row++) {
                    auto flat = column * 2u + row;
                    expect(result[lane][flat] == expected[column][row])
                        << luisa::format(
                               "mat2 constant mismatch at lane {}, column {}, row {}",
                               lane, column, row);
                }
            }
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "mat2 constant test should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "Decorate", "ArrayStride", "32"))
                << "std140 arrays of float2x2 must have a 32-byte array stride";
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "MemberDecorate", "MatrixStride", "16"))
                << "std140 float2x2 columns must have a 16-byte matrix stride";
        }
    };

    "vk_user_compute_bool_equality_uses_logical_spirv_ops"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_bool_equality"};
        ScopedSourceDump source_dump;

        constexpr std::array source{
            0u, 1u, 2u, 3u, 5u, 10u, 12u, 15u};
        constexpr auto dispatch_count = source.size();

        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<uint32_t>(dispatch_count);
        auto output = dc.device.create_buffer<uint32_t>(dispatch_count);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt in, BufferUInt out) noexcept {
            auto lane = dispatch_x();
            auto bits = in.read(lane);
            auto a = (bits & 1u) != 0u;
            auto b = (bits & 2u) != 0u;
            auto c = (bits & 4u) != 0u;
            auto d = (bits & 8u) != 0u;
            auto lhs2 = make_bool2(a, b);
            auto rhs2 = make_bool2(b, a);
            auto lhs3 = make_bool3(a, b, c);
            auto rhs3 = make_bool3(c, a, b);
            auto lhs4 = make_bool4(a, b, c, d);
            auto rhs4 = make_bool4(d, c, b, a);
            auto equal2 = lhs2 == rhs2;
            auto not_equal2 = lhs2 != rhs2;
            auto equal3 = lhs3 == rhs3;
            auto not_equal3 = lhs3 != rhs3;
            auto equal4 = lhs4 == rhs4;
            auto not_equal4 = lhs4 != rhs4;
            UInt packed = ite(a == b, 1u << 0u, 0u) |
                          ite(a != b, 1u << 1u, 0u) |
                          ite(equal2.x, 1u << 2u, 0u) |
                          ite(equal2.y, 1u << 3u, 0u) |
                          ite(not_equal2.x, 1u << 4u, 0u) |
                          ite(not_equal2.y, 1u << 5u, 0u) |
                          ite(equal3.x, 1u << 6u, 0u) |
                          ite(equal3.y, 1u << 7u, 0u) |
                          ite(equal3.z, 1u << 8u, 0u) |
                          ite(not_equal3.x, 1u << 9u, 0u) |
                          ite(not_equal3.y, 1u << 10u, 0u) |
                          ite(not_equal3.z, 1u << 11u, 0u) |
                          ite(equal4.x, 1u << 12u, 0u) |
                          ite(equal4.y, 1u << 13u, 0u) |
                          ite(equal4.z, 1u << 14u, 0u) |
                          ite(equal4.w, 1u << 15u, 0u) |
                          ite(not_equal4.x, 1u << 16u, 0u) |
                          ite(not_equal4.y, 1u << 17u, 0u) |
                          ite(not_equal4.z, 1u << 18u, 0u) |
                          ite(not_equal4.w, 1u << 19u, 0u);
            out.write(lane, packed);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, dispatch_count> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(dispatch_count)
               << output.copy_to(luisa::span{result})
               << synchronize();

        auto append_expected = [](uint32_t &packed, bool value,
                                  uint32_t bit) noexcept {
            packed |= static_cast<uint32_t>(value) << bit;
        };
        for (auto lane = 0u; lane < dispatch_count; lane++) {
            std::array values{
                (source[lane] & 1u) != 0u,
                (source[lane] & 2u) != 0u,
                (source[lane] & 4u) != 0u,
                (source[lane] & 8u) != 0u};
            constexpr std::array lhs2_indices{0u, 1u};
            constexpr std::array rhs2_indices{1u, 0u};
            constexpr std::array lhs3_indices{0u, 1u, 2u};
            constexpr std::array rhs3_indices{2u, 0u, 1u};
            constexpr std::array lhs4_indices{0u, 1u, 2u, 3u};
            constexpr std::array rhs4_indices{3u, 2u, 1u, 0u};
            auto expected = uint32_t{0u};
            auto bit = uint32_t{0u};
            append_expected(expected, values[0u] == values[1u], bit++);
            append_expected(expected, values[0u] != values[1u], bit++);
            auto append_vector = [&](auto lhs_indices,
                                     auto rhs_indices) noexcept {
                for (auto i = 0u; i < lhs_indices.size(); i++) {
                    append_expected(expected,
                                    values[lhs_indices[i]] ==
                                        values[rhs_indices[i]],
                                    bit++);
                }
                for (auto i = 0u; i < lhs_indices.size(); i++) {
                    append_expected(expected,
                                    values[lhs_indices[i]] !=
                                        values[rhs_indices[i]],
                                    bit++);
                }
            };
            append_vector(lhs2_indices, rhs2_indices);
            append_vector(lhs3_indices, rhs3_indices);
            append_vector(lhs4_indices, rhs4_indices);
            expect(result[lane] == expected)
                << luisa::format(
                       "boolean equality mismatch at lane {}: "
                       "expected 0x{:x}, got 0x{:x}",
                       lane, expected, result[lane]);
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "boolean equality test should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "LogicalEqual") > 0u)
                << "boolean equality must lower to OpLogicalEqual";
            expect(count_spirv_opcode(disassembly, "LogicalNotEqual") > 0u)
                << "boolean inequality must lower to OpLogicalNotEqual";
        }
    };

    "vk_clear_render_target_respects_nonzero_mip_level"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto *raster = dc.device.extension<RasterExt>();
        expect(raster != nullptr);
        if (raster == nullptr) { return; }

        constexpr auto base_size = make_uint2(8u, 4u);
        constexpr auto mip_levels = 3u;
        auto image = dc.device.create_image<float>(
            PixelStorage::BYTE4, base_size, mip_levels, false, true);
        auto stream = dc.device.create_stream(StreamTag::GRAPHICS);
        std::array<luisa::vector<std::byte>, mip_levels> initial;
        std::array<luisa::vector<std::byte>, mip_levels> actual;
        for (auto level = 0u; level < mip_levels; level++) {
            auto byte_count = image.view(level).size_bytes();
            initial[level].resize(byte_count);
            actual[level].resize(byte_count);
            std::fill(initial[level].begin(), initial[level].end(),
                      std::byte{static_cast<uint8_t>(0x11u * (level + 1u))});
            stream << image.view(level).copy_from(
                luisa::span{initial[level]});
        }
        constexpr auto target_level = 1u;
        stream << raster->clear_render_target(
            image.view(target_level),
            make_float4(1.0f, 0.0f, 1.0f, 0.0f));
        for (auto level = 0u; level < mip_levels; level++) {
            stream << image.view(level).copy_to(luisa::span{actual[level]});
        }
        stream << synchronize();

        for (auto level : {0u, 2u}) {
            expect(static_cast<bool>(actual[level] == initial[level]))
                << luisa::format(
                       "clearing mip {} modified neighboring mip {}",
                       target_level, level);
        }
        constexpr std::array expected_pixel{
            std::byte{0xff}, std::byte{0x00},
            std::byte{0xff}, std::byte{0x00}};
        expect(actual[target_level].size() % expected_pixel.size() == 0u);
        for (auto offset = size_t{0u};
             offset + expected_pixel.size() <= actual[target_level].size();
             offset += expected_pixel.size()) {
            for (auto channel = 0u; channel < expected_pixel.size(); channel++) {
                expect(actual[target_level][offset + channel] ==
                       expected_pixel[channel])
                    << luisa::format(
                           "mip clear mismatch at byte {}",
                           offset + channel);
            }
        }
    };

    "vk_user_compute_word_backed_four_bools_dynamic_indices"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<FourBools>(4u);
        auto output = dc.device.create_buffer<FourBools>(4u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferVar<FourBools> in,
                             BufferVar<FourBools> out) noexcept {
            auto lane = dispatch_x();
            auto source_index = lane + 1u;
            auto destination_index = 3u - lane;
            auto component_index = lane + 1u;
            auto source = in.read(source_index);
            Var<FourBools> result{source};
            result.values[component_index] =
                !source.values[component_index];
            out.write(destination_index, result);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            FourBools{{false, false, false, false}},
            FourBools{{true, false, true, false}},
            FourBools{{false, true, false, true}},
            FourBools{{true, true, false, false}}};
        constexpr std::array initial{
            FourBools{{true, true, true, true}},
            FourBools{{false, false, false, false}},
            FourBools{{true, false, false, true}},
            FourBools{{false, true, true, false}}};
        auto expected = initial;
        for (auto lane = 0u; lane < 3u; lane++) {
            auto source_index = lane + 1u;
            auto destination_index = 3u - lane;
            auto component_index = lane + 1u;
            expected[destination_index] = source[source_index];
            expected[destination_index].values[component_index] =
                !expected[destination_index].values[component_index];
        }
        std::array<FourBools, 4u> result{};
        stream << input.copy_from(luisa::span{source})
               << output.copy_from(luisa::span{initial})
               << shader(input, output).dispatch(3u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        for (auto element = 0u; element < result.size(); element++) {
            for (auto component = 0u; component < 4u; component++) {
                expect(result[element].values[component] ==
                       expected[element].values[component])
                    << luisa::format(
                           "FourBools mismatch at element {}, component {}",
                           element, component);
            }
        }
    };

    "vk_user_compute_word_backed_mixed_composite_round_trip"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto input =
            dc.device.create_buffer<WordBackedMixedComposite>(3u);
        auto output =
            dc.device.create_buffer<WordBackedMixedComposite>(3u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](
                              BufferVar<WordBackedMixedComposite> in,
                              BufferVar<WordBackedMixedComposite> out) noexcept {
            auto i = dispatch_x();
            auto source = in.read(i);
            Var<WordBackedMixedComposite> result{source};
            result.tag = source.tag + 10u + i;
            result.flags[0] = !source.flags[3];
            result.flags[1] = source.flags[0];
            result.flags[2] = !source.flags[1];
            result.flags[3] = source.flags[2];
            result.payload = source.payload ^ 0x55aa55aau;
            out.write(2u - i, result);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            WordBackedMixedComposite{
                3u, {true, false, true, false}, 0x12345678u},
            WordBackedMixedComposite{
                7u, {false, true, false, true}, 29u},
            WordBackedMixedComposite{
                13u, {true, true, false, false}, 0xffffffffu}};
        std::array<WordBackedMixedComposite, 3u> expected{};
        for (auto i = 0u; i < source.size(); i++) {
            auto value = source[i];
            value.tag += 10u + i;
            value.flags[0] = !source[i].flags[3];
            value.flags[1] = source[i].flags[0];
            value.flags[2] = !source[i].flags[1];
            value.flags[3] = source[i].flags[2];
            value.payload = source[i].payload ^ 0x55aa55aau;
            expected[2u - i] = value;
        }
        std::array<WordBackedMixedComposite, 3u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(3u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        for (auto element = 0u; element < result.size(); element++) {
            expect(result[element].tag == expected[element].tag);
            for (auto component = 0u; component < 4u; component++) {
                expect(result[element].flags[component] ==
                       expected[element].flags[component])
                    << luisa::format(
                           "mixed-composite bool mismatch at element {}, component {}",
                           element, component);
            }
            expect(result[element].payload == expected[element].payload);
        }
    };

    "vk_user_compute_word_backed_wide_vector_layout_round_trip"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_wide_vector_storage_layout"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto input =
            dc.device.create_buffer<WideVectorStorageRecord>(3u);
        auto output =
            dc.device.create_buffer<WideVectorStorageRecord>(3u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](
                              BufferVar<WideVectorStorageRecord> in,
                              BufferVar<WideVectorStorageRecord> out) noexcept {
            auto i = dispatch_x();
            auto source = in.read(i);
            Var<WideVectorStorageRecord> result{source};
            result.prefix = source.prefix + make_float4(
                                                cast<float>(i), 1.0f,
                                                2.0f, 3.0f);
            result.suffix = source.suffix ^ (0x10203040u + i);
            out.write(2u - i, result);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            WideVectorStorageRecord{
                {1.0f, 2.0f, 3.0f, 4.0f},
                {10.0, 20.0, 30.0, 40.0},
                0x11223344u},
            WideVectorStorageRecord{
                {-1.0f, -2.0f, -3.0f, -4.0f},
                {-10.0, -20.0, -30.0, -40.0},
                0x55667788u},
            WideVectorStorageRecord{
                {0.25f, 0.5f, 0.75f, 1.0f},
                {0.125, 0.25, 0.5, 1.0},
                0xaabbccddu}};
        auto expected = source;
        for (auto i = 0u; i < source.size(); ++i) {
            auto value = source[i];
            value.prefix += float4{
                static_cast<float>(i), 1.0f, 2.0f, 3.0f};
            value.suffix ^= 0x10203040u + i;
            expected[2u - i] = value;
        }

        std::array<WideVectorStorageRecord, 3u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(3u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        for (auto element = 0u; element < result.size(); ++element) {
            for (auto component = 0u; component < 4u; ++component) {
                expect(result[element].prefix[component] ==
                       expected[element].prefix[component]);
                expect(result[element].payload[component] ==
                       expected[element].payload[component]);
            }
            expect(result[element].suffix == expected[element].suffix);
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "wide-vector layout fallback should emit one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "Decorate", "ArrayStride", "4"))
                << "the incompatible host aggregate must use uint32 word storage";
            expect(!spirv_opcode_has_adjacent_operands(
                disassembly, "Decorate", "ArrayStride", "64"))
                << "the incompatible host aggregate must not be emitted as a typed runtime array";
            expect(count_spirv_opcode(
                       disassembly, "AtomicCompareExchange") == 0u)
                << "proven-aligned non-atomic word stores must not expand into masked CAS loops";
        }
    };

    "vk_user_compute_typed_nested_matrix_layout_round_trip"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_nested_matrix_storage_layout"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto input =
            dc.device.create_buffer<NestedMatrixStorageRecord>(2u);
        auto output =
            dc.device.create_buffer<NestedMatrixStorageRecord>(2u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](
                              BufferVar<NestedMatrixStorageRecord> in,
                              BufferVar<NestedMatrixStorageRecord> out) noexcept {
            auto i = dispatch_x();
            auto source = in.read(i);
            Var<NestedMatrixStorageRecord> result{source};
            result.prefix = source.prefix + make_float4(
                                                cast<float>(i), 1.0f,
                                                2.0f, 3.0f);
            result.transforms[0] = source.transforms[1];
            result.transforms[1] = source.transforms[0];
            result.suffix = source.suffix ^ (0x01020304u + i);
            out.write(1u - i, result);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            NestedMatrixStorageRecord{
                {1.0f, 2.0f, 3.0f, 4.0f},
                {make_float2x2(5.0f, 6.0f, 7.0f, 8.0f),
                 make_float2x2(9.0f, 10.0f, 11.0f, 12.0f)},
                0x11223344u},
            NestedMatrixStorageRecord{
                {-1.0f, -2.0f, -3.0f, -4.0f},
                {make_float2x2(-5.0f, -6.0f, -7.0f, -8.0f),
                 make_float2x2(-9.0f, -10.0f, -11.0f, -12.0f)},
                0xaabbccddu}};
        auto expected = source;
        for (auto i = 0u; i < source.size(); ++i) {
            auto value = source[i];
            value.prefix += float4{
                static_cast<float>(i), 1.0f, 2.0f, 3.0f};
            std::swap(value.transforms[0], value.transforms[1]);
            value.suffix ^= 0x01020304u + i;
            expected[1u - i] = value;
        }

        std::array<NestedMatrixStorageRecord, 2u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(2u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        for (auto element = 0u; element < result.size(); ++element) {
            for (auto component = 0u; component < 4u; ++component) {
                expect(result[element].prefix[component] ==
                       expected[element].prefix[component]);
            }
            for (auto matrix = 0u; matrix < 2u; ++matrix) {
                for (auto column = 0u; column < 2u; ++column) {
                    for (auto row = 0u; row < 2u; ++row) {
                        expect(result[element].transforms[matrix][column][row] ==
                               expected[element].transforms[matrix][column][row]);
                    }
                }
            }
            expect(result[element].suffix == expected[element].suffix);
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "nested-matrix layout regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "Decorate", "ArrayStride", "64"))
                << "the outer typed runtime array must retain the 64-byte host stride";
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "Decorate", "ArrayStride", "16"))
                << "the nested float2x2 array must retain its 16-byte element stride";
            expect(spirv_opcode_has_adjacent_operands(
                disassembly, "MemberDecorate", "MatrixStride", "8"))
                << "the array-bearing struct member must carry the float2 column stride";
        }
    };

    "vk_user_compute_unaligned_byte_buffer_cross_word_io"_test = [&] {
        constexpr auto byte_count = 24u;
        auto dc = luisa::test::create_device(argc, argv);
        auto bytes = dc.device.create_byte_buffer(byte_count);
        auto observed = dc.device.create_buffer<uint32_t>(5u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](ByteBufferVar buffer,
                             BufferUInt out) noexcept {
            out.write(0u, buffer.read<uint32_t>(1u));
            out.write(1u, buffer.read<uint32_t>(3u));
            buffer.write(5u, 0xd4c3b2a1u);
            buffer.write(11u, make_uint2(0x44332211u, 0x88776655u));
            out.write(2u, buffer.read<uint32_t>(5u));
            auto pair = buffer.read<uint2>(11u);
            out.write(3u, pair.x);
            out.write(4u, pair.y);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint8_t, byte_count> source{};
        for (auto i = 0u; i < source.size(); i++) {
            source[i] = static_cast<uint8_t>(0x20u + i);
        }
        auto expected_bytes = source;
        uint32_t expected_read_1{};
        uint32_t expected_read_3{};
        std::memcpy(&expected_read_1, source.data() + 1u,
                    sizeof(expected_read_1));
        std::memcpy(&expected_read_3, source.data() + 3u,
                    sizeof(expected_read_3));
        constexpr auto scalar_write = 0xd4c3b2a1u;
        constexpr std::array vector_write{0x44332211u, 0x88776655u};
        std::memcpy(expected_bytes.data() + 5u, &scalar_write,
                    sizeof(scalar_write));
        std::memcpy(expected_bytes.data() + 11u, vector_write.data(),
                    sizeof(vector_write));

        std::array<uint32_t, 5u> observed_values{};
        std::array<uint8_t, byte_count> result_bytes{};
        stream << bytes.copy_from(source.data())
               << shader(bytes, observed).dispatch(1u)
               << observed.copy_to(luisa::span{observed_values})
               << bytes.copy_to(result_bytes.data())
               << synchronize();

        constexpr std::array expected_observed{
            uint32_t{0u}, uint32_t{0u},
            scalar_write, vector_write[0], vector_write[1]};
        auto expected_values = expected_observed;
        expected_values[0] = expected_read_1;
        expected_values[1] = expected_read_3;
        expect(observed_values == expected_values)
            << "unaligned scalar/vector reads must reconstruct every crossed word exactly";
        expect(result_bytes == expected_bytes)
            << "unaligned scalar/vector writes must preserve every surrounding canary byte";
    };

    "vk_user_compute_volatile_buffer_is_coherent"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_volatile_buffer"};
        ScopedSourceDump source_dump;
        auto dc = luisa::test::create_device(argc, argv);
        auto volatile_buffer = dc.device.create_buffer<uint32_t>(2u);
        auto ordinary_buffer = dc.device.create_buffer<uint32_t>(2u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt volatile_values,
                             BufferUInt ordinary_values) noexcept {
            auto volatile_value = volatile_values.volatile_read(0u);
            volatile_values.volatile_write(
                1u, volatile_value + 1u);
            auto ordinary_value = ordinary_values.read(0u);
            ordinary_values.write(1u, ordinary_value + 1u);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array<uint32_t, 2u> volatile_source{
            0x12345678u, 0u};
        constexpr std::array<uint32_t, 2u> ordinary_source{
            0x87654321u, 0u};
        std::array<uint32_t, 2u> volatile_result{};
        std::array<uint32_t, 2u> ordinary_result{};
        stream << volatile_buffer.copy_from(
                      luisa::span{volatile_source})
               << ordinary_buffer.copy_from(
                      luisa::span{ordinary_source})
               << shader(volatile_buffer, ordinary_buffer).dispatch(1u)
               << volatile_buffer.copy_to(
                      luisa::span{volatile_result})
               << ordinary_buffer.copy_to(
                      luisa::span{ordinary_result})
               << synchronize();

        expect(volatile_result[0] == volatile_source[0]);
        expect(volatile_result[1] == volatile_source[0] + 1u);
        expect(ordinary_result[0] == ordinary_source[0]);
        expect(ordinary_result[1] == ordinary_source[0] + 1u);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "volatile-buffer regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto volatile_id = spirv_id_named(
                disassembly, "_buf_0");
            auto ordinary_id = spirv_id_named(
                disassembly, "_buf_1");
            expect(volatile_id.has_value());
            expect(ordinary_id.has_value());
            if (volatile_id) {
                expect(spirv_id_has_decoration(
                    disassembly, *volatile_id, "Coherent"));
            }
            if (ordinary_id) {
                expect(!spirv_id_has_decoration(
                    disassembly, *ordinary_id, "Coherent"));
            }
            expect(count_spirv_opcode(
                       disassembly, "MemoryBarrier") == 2u)
                << "volatile read/write should retain their matching device fences";
            expect(disassembly.find("Volatile") !=
                   std::string::npos)
                << "volatile loads/stores must retain their memory-access operand";
        }
    };

    "vk_user_compute_sliced_buffer_descriptor_bias_is_exact"_test = [&] {
        constexpr auto byte_count = 37u;
        constexpr auto view_offset = 5u;
        constexpr auto view_size = 13u;
        auto dc = luisa::test::create_device(argc, argv);
        auto bytes = dc.device.create_byte_buffer(byte_count);
        auto observed = dc.device.create_buffer<uint32_t>(3u);
        auto stream = dc.device.create_stream();

        Callable inspect_slice = [](ByteBufferVar slice) noexcept {
            auto first = slice.read<uint32_t>(0u);
            auto last = slice.read<uint32_t>(view_size - sizeof(uint32_t));
            slice.write(4u, 0xa1b2c3d4u);
            return make_uint2(first, last);
        };
        Kernel1D kernel = [&](ByteBufferVar slice,
                              BufferUInt out) noexcept {
            auto ends = inspect_slice(slice);
            out.write(0u, ends.x);
            out.write(1u, ends.y);
            out.write(2u, slice.read<uint32_t>(4u));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint8_t, byte_count> source{};
        for (auto i = 0u; i < source.size(); i++) {
            source[i] = static_cast<uint8_t>(0x40u + i);
        }
        auto expected_bytes = source;
        uint32_t expected_first{};
        uint32_t expected_last{};
        std::memcpy(&expected_first, source.data() + view_offset,
                    sizeof(expected_first));
        std::memcpy(&expected_last,
                    source.data() + view_offset + view_size - sizeof(uint32_t),
                    sizeof(expected_last));
        constexpr auto replacement = 0xa1b2c3d4u;
        std::memcpy(expected_bytes.data() + view_offset + 4u,
                    &replacement, sizeof(replacement));

        std::array<uint32_t, 3u> result{};
        std::array<uint8_t, byte_count> result_bytes{};
        stream << bytes.copy_from(source.data())
               << shader(bytes.view(view_offset, view_size), observed).dispatch(1u)
               << observed.copy_to(luisa::span{result})
               << bytes.copy_to(result_bytes.data())
               << synchronize();

        expect(result[0] == expected_first);
        expect(result[1] == expected_last);
        expect(result[2] == replacement);
        expect(result_bytes == expected_bytes)
            << "descriptor-relative byte-buffer bias must preserve bytes outside the slice";
    };

    "vk_user_compute_typed_sliced_buffer_uses_element_bias"_test = [&] {
        constexpr auto total_count = 19u;
        constexpr auto view_offset = 3u;
        constexpr auto view_count = 7u;
        auto dc = luisa::test::create_device(argc, argv);
        auto values = dc.device.create_buffer<uint32_t>(total_count);
        auto observed = dc.device.create_buffer<uint32_t>(3u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt slice,
                             BufferUInt out) noexcept {
            out.write(0u, slice.read(0u));
            out.write(1u, slice.read(view_count - 1u));
            slice.write(2u, 0xfedcba98u);
            out.write(2u, slice.read(2u));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, total_count> source{};
        for (auto i = 0u; i < source.size(); i++) {
            source[i] = 1000u + i * 17u;
        }
        auto expected = source;
        expected[view_offset + 2u] = 0xfedcba98u;
        std::array<uint32_t, 3u> result{};
        std::array<uint32_t, total_count> result_values{};
        stream << values.copy_from(luisa::span{source})
               << shader(values.view(view_offset, view_count), observed).dispatch(1u)
               << observed.copy_to(luisa::span{result})
               << values.copy_to(luisa::span{result_values})
               << synchronize();

        expect(result[0] == source[view_offset]);
        expect(result[1] == source[view_offset + view_count - 1u]);
        expect(result[2] == 0xfedcba98u);
        expect(result_values == expected)
            << "typed descriptor-relative element bias must not escape the bound slice";
    };

    "vk_user_compute_bindless_sliced_buffer_preserves_view_contract"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_bindless_buffer_view"};
        ScopedSourceDump source_dump;

        constexpr auto byte_count = 37u;
        constexpr auto view_offset = 5u;
        constexpr auto view_size = 13u;
        constexpr auto wide_index = luisa::ulong{2u};
        constexpr auto replacement = 0xa1b2c3d4u;
        auto dc = luisa::test::create_device(argc, argv);
        auto bytes = dc.device.create_byte_buffer(byte_count);
        auto indices = dc.device.create_buffer<luisa::ulong>(1u);
        auto observed = dc.device.create_buffer<uint32_t>(6u);
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferULong dynamic_indices,
                             BufferUInt out) noexcept {
            auto byte_view = bindless.byte_buffer(0u);
            auto word_view = bindless.buffer<uint32_t>(0u);
            auto wide = dynamic_indices.read(0u);
            out.write(0u, byte_view.size());
            out.write(1u, word_view.size());
            out.write(2u, byte_view.read<uint32_t>(0u));
            out.write(3u, word_view.read(wide));
            word_view.write(wide - 1ull, 0xa1b2c3d4u);
            out.write(4u, word_view.read(wide - 1ull));
            out.write(
                5u, byte_view.read<uint32_t>(view_size - sizeof(uint32_t)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint8_t, byte_count> source{};
        for (auto i = 0u; i < source.size(); ++i) {
            source[i] = static_cast<uint8_t>(0x30u + i);
        }
        auto expected_bytes = source;
        uint32_t expected_first{};
        uint32_t expected_wide{};
        uint32_t expected_last{};
        std::memcpy(&expected_first, source.data() + view_offset,
                    sizeof(expected_first));
        std::memcpy(
            &expected_wide,
            source.data() + view_offset + wide_index * sizeof(uint32_t),
            sizeof(expected_wide));
        std::memcpy(
            &expected_last,
            source.data() + view_offset + view_size - sizeof(uint32_t),
            sizeof(expected_last));
        std::memcpy(
            expected_bytes.data() + view_offset + sizeof(uint32_t),
            &replacement, sizeof(replacement));

        constexpr std::array<luisa::ulong, 1u> index_source{wide_index};
        std::array<uint32_t, 6u> result{};
        std::array<uint8_t, byte_count> result_bytes{};
        stream << bytes.copy_from(source.data())
               << indices.copy_from(luisa::span{index_source})
               << heap.emplace_on_update(
                          0u, bytes.view(view_offset, view_size))
                      .update()
               << shader(heap, indices, observed).dispatch(1u)
               << observed.copy_to(luisa::span{result})
               << bytes.copy_to(result_bytes.data())
               << synchronize();

        const std::array expected{
            view_size,
            view_size / static_cast<uint32_t>(sizeof(uint32_t)),
            expected_first,
            expected_wide,
            replacement,
            expected_last};
        expect(result == expected)
            << "bindless byte/typed reads and exact size queries must be relative to the logical view";
        expect(result_bytes == expected_bytes)
            << "bindless writes must preserve canary bytes outside the logical view";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "bindless view regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto uint64_type = spirv_unsigned_64_type_token(disassembly);
            expect(uint64_type.has_value())
                << "the dynamic uint64 bindless index must retain its SPIR-V type";
            if (uint64_type) {
                expect(spirv_u64_scaled_index_reaches_buffer_load(
                    disassembly, *uint64_type))
                    << "the uint64 element-to-byte IMul must feed the "
                       "descriptor-relative bias, word-index division, final "
                       "buffer access chain, and OpLoad without narrowing";
            }
        }
    };

    "vk_user_compute_bindless_updates_observe_command_order"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto first = dc.device.create_buffer<uint32_t>(1u);
        auto second = dc.device.create_buffer<uint32_t>(1u);
        auto output = dc.device.create_buffer<uint32_t>(2u);
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferUInt out) noexcept {
            out.write(0u, bindless.buffer<uint32_t>(0u).read(0u));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        heap.emplace_on_update(0u, first);
        auto bind_first = heap.update();
        heap.emplace_on_update(0u, second);
        auto bind_second = heap.update();

        constexpr std::array first_source{0x11223344u};
        constexpr std::array second_source{0xa1b2c3d4u};
        std::array<uint32_t, 2u> result{};
        stream << first.copy_from(luisa::span{first_source})
               << second.copy_from(luisa::span{second_source})
               << std::move(bind_first)
               << shader(heap, output.view(0u, 1u)).dispatch(1u)
               << std::move(bind_second)
               << shader(heap, output.view(1u, 1u)).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(result ==
               std::array{first_source[0], second_source[0]})
            << "each dispatch must observe the bindless update preceding it, "
               "even when all host descriptor writes occur before submission";
    };

    "vk_bindless_write_snapshot_survives_later_heap_update"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto first = dc.device.create_buffer<uint32_t>(1u);
        auto second = dc.device.create_buffer<uint32_t>(1u);
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless, UInt value) noexcept {
            bindless.buffer<uint32_t>(0u).write(0u, value);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        heap.emplace_on_update(0u, first);
        auto bind_first = heap.update();
        heap.emplace_on_update(0u, second);
        auto bind_second = heap.update();

        constexpr std::array first_source{0x11111111u};
        constexpr std::array second_source{0x22222222u};
        constexpr std::array first_sentinel{0xfeed0001u};
        constexpr std::array second_sentinel{0xfeed0002u};
        constexpr auto replacement = 0xa1b2c3d4u;
        std::array<uint32_t, 1u> first_result{};
        std::array<uint32_t, 1u> second_result{};
        stream << first.copy_from(luisa::span{first_sentinel})
               << second.copy_from(luisa::span{second_sentinel})
               << synchronize();
        stream << first.copy_from(luisa::span{first_source})
               << second.copy_from(luisa::span{second_source})
               << std::move(bind_first)
               << shader(heap, replacement).dispatch(1u)
               // Replacing the slot after recording the write must not erase
               // the reorder dependency snapshotted for `first`.
               << std::move(bind_second)
               << first.copy_to(luisa::span{first_result})
               << second.copy_to(luisa::span{second_result})
               << synchronize();

        expect(first_result[0] == replacement);
        expect(second_result == second_source)
            << "the later heap update must not retarget the earlier dispatch";
    };

    "vk_bindless_write_orders_prior_and_later_direct_reads"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto buffer = dc.device.create_buffer<uint32_t>(1u);
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless, UInt value) noexcept {
            bindless.buffer<uint32_t>(0u).write(0u, value);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});
        heap.emplace_on_update(0u, buffer);

        constexpr std::array initial{0x10203040u};
        constexpr auto replacement = 0xa1b2c3d4u;
        stream << buffer.copy_from(luisa::span{initial})
               << heap.update()
               << synchronize();

        std::array<uint32_t, 1u> before{};
        std::array<uint32_t, 1u> after{};
        stream << buffer.copy_to(luisa::span{before})
               << shader(heap, replacement).dispatch(1u)
               << buffer.copy_to(luisa::span{after})
               << synchronize();

        expect(before == initial)
            << "the bindless write must wait for the preceding direct read";
        expect(after[0] == replacement)
            << "the later direct read must wait for the bindless write";
    };

    "vk_typed_buffer_only_updates_preserve_order_bias_and_size"_test = [&] {
        constexpr auto first_offset = 1u;
        constexpr auto first_count = 3u;
        constexpr auto second_offset = 2u;
        constexpr auto second_count = 2u;
        auto dc = luisa::test::create_device(argc, argv);
        ScopedEnvironmentVariable require_native{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
        auto first = dc.device.create_buffer<uint32_t>(6u);
        auto second = dc.device.create_buffer<uint32_t>(6u);
        auto output = dc.device.create_buffer<uint32_t>(8u);
        auto heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::BUFFER_ONLY);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferUInt out) noexcept {
            auto view = bindless.buffer<uint32_t>(0u, true);
            auto count = view.size();
            out.write(0u, view.read(0u));
            out.write(1u, view.read(count - 1u));
            out.write(2u, count);
            out.write(
                3u,
                bindless.byte_buffer(0u, true, true)
                    .read<uint32_t>((count - 1u) * 4u));
            view.write(count - 1u, view.read(0u) + 100u);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        heap.emplace_on_update(
            0u, first.view(first_offset, first_count));
        auto bind_first = heap.update();
        heap.emplace_on_update(
            0u, second.view(second_offset, second_count));
        auto bind_second = heap.update();

        constexpr std::array first_source{
            0xdead0000u, 11u, 12u, 13u, 0xdead0004u, 0xdead0005u};
        constexpr std::array second_source{
            0xbeef0000u, 0xbeef0001u, 21u, 22u, 0xbeef0004u, 0xbeef0005u};
        std::array<uint32_t, 8u> result{};
        std::array<uint32_t, 6u> first_after{};
        std::array<uint32_t, 6u> second_after{};
        stream << first.copy_from(luisa::span{first_source})
               << second.copy_from(luisa::span{second_source})
               << std::move(bind_first)
               << shader(heap, output.view(0u, 4u)).dispatch(1u)
               << std::move(bind_second)
               << shader(heap, output.view(4u, 4u)).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << first.copy_to(luisa::span{first_after})
               << second.copy_to(luisa::span{second_after})
               << synchronize();

        constexpr std::array expected{
            first_source[first_offset],
            first_source[first_offset + first_count - 1u],
            first_count,
            first_source[first_offset + first_count - 1u],
            second_source[second_offset],
            second_source[second_offset + second_count - 1u],
            second_count,
            second_source[second_offset + second_count - 1u]};
        expect(result == expected)
            << "typed BUFFER_ONLY records must be versioned in command order "
               "and retain each sliced view's bias and exact logical size";
        auto expected_first_after = first_source;
        expected_first_after[first_offset + first_count - 1u] =
            first_source[first_offset] + 100u;
        auto expected_second_after = second_source;
        expected_second_after[second_offset + second_count - 1u] =
            second_source[second_offset] + 100u;
        expect(first_after == expected_first_after);
        expect(second_after == expected_second_after);
    };

    "vk_typed_texture_only_uses_native_slot_layout_and_sampler"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        ScopedEnvironmentVariable require_native{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
        constexpr auto extent = make_uint2(2u, 1u);
        auto first = dc.device.create_image<float>(
            PixelStorage::FLOAT4, extent);
        auto second = dc.device.create_image<float>(
            PixelStorage::FLOAT4, extent);
        auto heap = dc.device.create_bindless_array(
            2u, BindlessSlotType::TEXTURE2D_ONLY);
        auto sizes = dc.device.create_buffer<uint2>(2u);
        auto colors = dc.device.create_buffer<float4>(7u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferUInt2 size_output,
                             BufferFloat4 color_output) noexcept {
            auto lane = dispatch_x();
            auto texture = bindless.tex2d(lane, true, false);
            Float default_u = 0.5f;
            $if (lane == 0u) { default_u = 0.75f; };
            size_output.write(lane, texture.size());
            color_output.write(
                lane * 3u,
                texture.read(make_uint2(1u, 0u)));
            color_output.write(
                lane * 3u + 1u,
                texture.sample(make_float2(default_u, 0.5f)));
            color_output.write(
                lane * 3u + 2u,
                texture.sample(
                    make_float2(0.75f, 0.5f),
                    SamplerFilter::POINT, SamplerAddress::EDGE));
            $if (lane == 0u) {
                color_output.write(
                    6u,
                    bindless.tex2d(0u, true, true)
                        .read(make_uint2(0u)));
            };
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array first_source{
            float4{1.0f, 0.0f, 0.0f, 1.0f},
            float4{0.0f, 1.0f, 0.0f, 1.0f}};
        constexpr std::array second_source{
            float4{0.0f, 0.0f, 1.0f, 1.0f},
            float4{1.0f, 1.0f, 0.0f, 1.0f}};
        heap.emplace_on_update(
            0u, first, Sampler::point_edge());
        heap.emplace_on_update(
            1u, second, Sampler::linear_linear_edge());
        std::array<uint2, 2u> size_result{};
        std::array<float4, 7u> color_result{};
        stream << first.copy_from(luisa::span{first_source})
               << second.copy_from(luisa::span{second_source})
               << heap.update()
               << shader(heap, sizes, colors).dispatch(2u)
               << sizes.copy_to(luisa::span{size_result})
               << colors.copy_to(luisa::span{color_result})
               << synchronize();

        expect(size_result[0].x == extent.x &&
               size_result[0].y == extent.y &&
               size_result[1].x == extent.x &&
               size_result[1].y == extent.y);
        expect_vector_equal(color_result[0], first_source[1]);
        expect_vector_equal(color_result[1], first_source[1]);
        expect_vector_equal(color_result[2], first_source[1]);
        expect_vector_equal(color_result[3], second_source[1]);
        constexpr auto linear_midpoint =
            (second_source[0] + second_source[1]) * 0.5f;
        for (auto component = 0u; component < 4u; ++component) {
            expect(std::abs(color_result[4][component] -
                            linear_midpoint[component]) < 1e-6f)
                << luisa::format(
                       "typed default sampler component {} mismatch",
                       component);
        }
        expect_vector_equal(color_result[5], second_source[1]);
        expect_vector_equal(color_result[6], first_source[0]);
    };

    "vk_typed_volume_only_uses_native_slot_layout_and_sampler"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        ScopedEnvironmentVariable require_native{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1"};
        constexpr auto extent = make_uint3(2u, 1u, 1u);
        auto volume = dc.device.create_volume<float>(
            PixelStorage::FLOAT4, extent);
        auto heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::TEXTURE3D_ONLY);
        auto size_output = dc.device.create_buffer<uint3>(1u);
        auto color_output = dc.device.create_buffer<float4>(4u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferUInt3 sizes,
                             BufferFloat4 colors) noexcept {
            auto texture = bindless.tex3d(
                dispatch_x(), true, false);
            sizes.write(0u, texture.size());
            colors.write(
                0u, texture.read(make_uint3(1u, 0u, 0u)));
            colors.write(
                1u, texture.sample(make_float3(0.5f)));
            colors.write(
                2u, texture.sample(
                        make_float3(0.75f, 0.5f, 0.5f),
                        SamplerFilter::POINT, SamplerAddress::EDGE));
            colors.write(
                3u, bindless.tex3d(0u, true, true)
                        .read(make_uint3(0u)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            float4{1.0f, 0.0f, 0.0f, 1.0f},
            float4{0.0f, 0.0f, 1.0f, 1.0f}};
        heap.emplace_on_update(
            0u, volume, Sampler::linear_linear_edge());
        std::array<uint3, 1u> size_result{};
        std::array<float4, 4u> color_result{};
        stream << volume.copy_from(luisa::span{source})
               << heap.update()
               << shader(heap, size_output, color_output).dispatch(1u)
               << size_output.copy_to(luisa::span{size_result})
               << color_output.copy_to(luisa::span{color_result})
               << synchronize();

        expect(size_result[0].x == extent.x &&
               size_result[0].y == extent.y &&
               size_result[0].z == extent.z);
        expect_vector_equal(color_result[0], source[1]);
        constexpr auto midpoint = (source[0] + source[1]) * 0.5f;
        for (auto component = 0u; component < 4u; ++component) {
            expect(std::abs(color_result[1][component] -
                            midpoint[component]) < 1e-6f);
        }
        expect_vector_equal(color_result[2], source[1]);
        expect_vector_equal(color_result[3], source[0]);
    };

#if LUISA_TEST_VK_HAS_DXC_COMPATIBILITY
    "vk_dxc_fallback_retains_typed_bindless_slot_abis"_test = [&] {
        ScopedEnvironmentVariable allow_hlsl_fallback{
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", nullptr};
        auto dc = luisa::test::create_device(argc, argv);
        auto source_buffer = dc.device.create_buffer<uint32_t>(4u);
        auto source_image = dc.device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(2u, 1u));
        auto buffer_heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::BUFFER_ONLY);
        auto texture_heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::TEXTURE2D_ONLY);
        auto integers = dc.device.create_buffer<uint32_t>(5u);
        auto colors = dc.device.create_buffer<float4>(2u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar buffers,
                             BindlessVar textures,
                             BufferUInt integer_output,
                             BufferFloat4 color_output) noexcept {
            auto buffer = buffers.buffer<uint32_t>(0u, true);
            auto texture = textures.tex2d(0u, true, false);
            auto extent = texture.size();
            integer_output.write(0u, buffer.size());
            integer_output.write(1u, buffer.read(0u));
            integer_output.write(2u, buffer.read(buffer.size() - 1u));
            integer_output.write(3u, extent.x);
            integer_output.write(4u, extent.y);
            color_output.write(
                0u, texture.read(make_uint2(1u, 0u)));
            color_output.write(
                1u, texture.sample(
                        make_float2(0.75f, 0.5f),
                        SamplerFilter::POINT,
                        SamplerAddress::EDGE));
        };
        ShaderOption fallback_option{.enable_cache = false,
                                     .enable_fast_math = false};
        fallback_option.native_include = R"(
uint lc_typed_bindless_dxc_compatibility_marker(uint value) { return value; }
)";
        auto shader = dc.device.compile(kernel, fallback_option);

        constexpr std::array buffer_source{
            0xdead0000u, 41u, 42u, 0xdead0003u};
        constexpr std::array image_source{
            float4{1.0f, 0.0f, 0.0f, 1.0f},
            float4{0.0f, 1.0f, 0.0f, 1.0f}};
        buffer_heap.emplace_on_update(
            0u, source_buffer.view(1u, 2u));
        texture_heap.emplace_on_update(
            0u, source_image, Sampler::linear_linear_edge());
        std::array<uint32_t, 5u> integer_result{};
        std::array<float4, 2u> color_result{};
        stream << source_buffer.copy_from(luisa::span{buffer_source})
               << source_image.copy_from(luisa::span{image_source})
               << buffer_heap.update()
               << texture_heap.update()
               << shader(buffer_heap, texture_heap, integers, colors)
                      .dispatch(1u)
               << integers.copy_to(luisa::span{integer_result})
               << colors.copy_to(luisa::span{color_result})
               << synchronize();

        constexpr std::array expected_integers{2u, 41u, 42u, 2u, 1u};
        expect(integer_result == expected_integers)
            << "DXC fallback must retain typed buffer bias/size and typed "
               "texture descriptor layout";
        expect_vector_equal(color_result[0], image_source[1]);
        expect_vector_equal(color_result[1], image_source[1]);
    };
#endif

    "vk_bindless_texture_sampler_bits_do_not_escape_descriptor_recycling"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto first = dc.device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(1u));
        auto second = dc.device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(1u));
        auto output = dc.device.create_buffer<float4>(4u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferFloat4 out) noexcept {
            out.write(0u, bindless.tex2d(0u).read(make_uint2(0u)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});
        constexpr std::array first_source{
            float4{1.0f, 2.0f, 3.0f, 4.0f}};
        constexpr std::array second_source{
            float4{5.0f, 6.0f, 7.0f, 8.0f}};
        constexpr auto sampler = Sampler::linear_linear_repeat();

        // Keep upload and sampling in separate submissions. Upload completion
        // restores the texture to GENERAL, so this also exercises agreement
        // between the persistent bindless descriptor layout and tracked state.
        stream << first.copy_from(luisa::span{first_source})
               << second.copy_from(luisa::span{second_source})
               << synchronize();
        {
            auto heap = dc.device.create_bindless_array(1u);
            stream << heap.emplace_on_update(0u, first, sampler).update()
                   << shader(heap, output.view(0u, 1u)).dispatch(1u)
                   << synchronize();
            stream << heap.emplace_on_update(0u, second, sampler).update()
                   << shader(heap, output.view(1u, 1u)).dispatch(1u)
                   << synchronize();
            stream << heap.emplace_on_update(0u, first, sampler).update()
                   << shader(heap, output.view(2u, 1u)).dispatch(1u)
                   << synchronize();
        }
        // Destruction above also returns a packed texture slot. Allocate once
        // more so both replacement and destructor recycling are exercised.
        {
            auto heap = dc.device.create_bindless_array(1u);
            stream << heap.emplace_on_update(0u, second, sampler).update()
                   << shader(heap, output.view(3u, 1u)).dispatch(1u)
                   << synchronize();
        }

        std::array<float4, 4u> result{};
        stream << output.copy_to(luisa::span{result}) << synchronize();
        constexpr std::array expected{
            first_source[0], second_source[0],
            first_source[0], second_source[0]};
        for (auto i = 0u; i < result.size(); ++i) {
            expect_vector_equal(result[i], expected[i]);
        }
    };

    "vk_bindless_texture_waits_for_same_submission_upload"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto texture = dc.device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(1u));
        auto heap = dc.device.create_bindless_array(1u);
        auto output = dc.device.create_buffer<float4>(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BindlessVar bindless,
                             BufferFloat4 result) noexcept {
            result.write(
                0u, bindless.tex2d(0u).read(make_uint2(0u)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});
        constexpr std::array sentinel{
            float4{0.875f, 0.75f, 0.625f, 0.5f}};
        constexpr std::array source{
            float4{0.125f, 0.25f, 0.5f, 1.0f}};
        std::array<float4, 1u> result{};

        stream << texture.copy_from(luisa::span{sentinel})
               << output.copy_from(luisa::span{sentinel})
               << synchronize();
        stream << texture.copy_from(luisa::span{source})
               << heap.emplace_on_update(
                          0u, texture,
                          Sampler::point_zero())
                      .update()
               << shader(heap, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect_vector_equal(result[0], source[0]);
    };

    "vk_user_compute_texture_queries_reads_and_samples_preserve_mips"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_texture_mip_queries"};
        ScopedSourceDump source_dump;

        constexpr auto base_size = make_uint2(4u);
        constexpr auto mip_levels = 3u;
        constexpr auto selected_level = 2u;
        constexpr auto base_pixel = float4{0.125f, 0.25f, 0.5f, 1.0f};
        constexpr auto middle_pixel = float4{0.75f, 0.5f, 0.25f, 1.0f};
        constexpr auto final_pixel = float4{0.875f, 0.625f, 0.375f, 1.0f};
        auto dc = luisa::test::create_device(argc, argv);
        auto texture = dc.device.create_image<float>(
            PixelStorage::FLOAT4, base_size, mip_levels);
        auto heap = dc.device.create_bindless_array(1u);
        auto controls = dc.device.create_buffer<uint32_t>(2u);
        auto sizes = dc.device.create_buffer<uint2>(3u);
        auto pixels = dc.device.create_buffer<float4>(3u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](ImageFloat direct_view,
                             BindlessVar bindless,
                             BufferUInt control,
                             BufferUInt2 size_output,
                             BufferFloat4 pixel_output) noexcept {
            auto slot = control.read(0u);
            auto level = control.read(1u);
            auto sampled = bindless.tex2d(slot);
            size_output.write(0u, direct_view.size());
            size_output.write(1u, sampled.size());
            size_output.write(2u, sampled.size(level));
            pixel_output.write(
                0u, sampled.read(make_uint2(0u), level));
            pixel_output.write(
                1u, sampled.sample(make_float2(0.375f, 0.625f)));
            pixel_output.write(
                2u, sampled.sample(
                        make_float2(0.375f, 0.625f),
                        cast<float>(level)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<float4, 16u> mip0;
        std::array<float4, 4u> mip1;
        constexpr std::array mip2{final_pixel};
        mip0.fill(base_pixel);
        mip1.fill(middle_pixel);
        constexpr std::array<uint32_t, 2u> control_source{
            0u, selected_level};
        std::array<uint2, 3u> size_result{};
        std::array<float4, 3u> pixel_result{};
        stream << texture.view(0u).copy_from(luisa::span{mip0})
               << texture.view(1u).copy_from(luisa::span{mip1})
               << texture.view(2u).copy_from(luisa::span{mip2})
               << controls.copy_from(luisa::span{control_source})
               << heap.emplace_on_update(
                          0u, texture, Sampler::point_edge())
                      .update()
               << shader(texture.view(1u), heap, controls,
                         sizes, pixels)
                      .dispatch(1u)
               << sizes.copy_to(luisa::span{size_result})
               << pixels.copy_to(luisa::span{pixel_result})
               << synchronize();

        constexpr std::array expected_sizes{
            make_uint2(2u), base_size, make_uint2(1u)};
        for (auto i = 0u; i < size_result.size(); ++i) {
            expect(size_result[i].x == expected_sizes[i].x &&
                   size_result[i].y == expected_sizes[i].y)
                << luisa::format(
                       "texture size query {} mismatch: expected {}x{}, got {}x{}",
                       i, expected_sizes[i].x, expected_sizes[i].y,
                       size_result[i].x, size_result[i].y);
        }
        expect_vector_equal(pixel_result[0], final_pixel);
        expect_vector_equal(pixel_result[1], base_pixel);
        expect_vector_equal(pixel_result[2], final_pixel);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "texture mip regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto size_query_count =
                count_spirv_opcode(disassembly, "ImageQuerySize") +
                count_spirv_opcode(disassembly, "ImageQuerySizeLod");
            expect(size_query_count == 3u)
                << "each direct/bindless size query must remain an image query at SPIR-V opt0";
            expect(count_spirv_opcode(disassembly, "ImageFetch") == 1u)
                << "the explicit-level bindless read must lower to OpImageFetch";
            expect(count_spirv_opcode(
                       disassembly, "ImageSampleExplicitLod") == 2u)
                << "compute sampling must use explicit LOD operands, including the plain sample form";
            expect(count_spirv_opcode(
                       disassembly, "ImageSampleImplicitLod") == 0u)
                << "compute shaders must not depend on implicit derivatives";
        }
    };

    "vk_user_compute_volume_queries_reads_and_samples_preserve_mips"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_volume_mip_queries"};
        ScopedSourceDump source_dump;

        constexpr auto base_size = make_uint3(4u);
        constexpr auto mip_levels = 3u;
        constexpr auto selected_level = 2u;
        constexpr auto base_voxel =
            float4{0.125f, 0.25f, 0.5f, 1.0f};
        constexpr auto middle_voxel =
            float4{0.75f, 0.5f, 0.25f, 1.0f};
        constexpr auto final_voxel =
            float4{0.875f, 0.625f, 0.375f, 1.0f};
        auto dc = luisa::test::create_device(argc, argv);
        auto volume = dc.device.create_volume<float>(
            PixelStorage::FLOAT4, base_size, mip_levels);
        auto heap = dc.device.create_bindless_array(1u);
        auto controls = dc.device.create_buffer<uint32_t>(2u);
        auto sizes = dc.device.create_buffer<uint3>(3u);
        auto voxels = dc.device.create_buffer<float4>(6u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](VolumeFloat direct_view,
                             BindlessVar bindless,
                             BufferUInt control,
                             BufferUInt3 size_output,
                             BufferFloat4 voxel_output) noexcept {
            auto slot = control.read(0u);
            auto level = control.read(1u);
            auto sampled = bindless.tex3d(slot);
            size_output.write(0u, direct_view.size());
            size_output.write(1u, sampled.size());
            size_output.write(2u, sampled.size(level));
            voxel_output.write(
                0u, sampled.read(make_uint3(0u), level));
            voxel_output.write(
                1u, sampled.sample(make_float3(0.375f, 0.625f,
                                               0.375f)));
            voxel_output.write(
                2u, sampled.sample(
                        make_float3(0.375f, 0.625f, 0.375f),
                        cast<float>(level)));
            voxel_output.write(
                3u, direct_view.read(make_uint3(0u)));
            auto mip_one_ddx = make_float3(0.5f, 0.0f, 0.0f);
            auto mip_one_ddy = make_float3(0.0f, 0.5f, 0.0f);
            voxel_output.write(
                4u, sampled.sample(
                        make_float3(0.375f, 0.625f, 0.375f),
                        mip_one_ddx, mip_one_ddy));
            auto base_ddx = make_float3(0.25f, 0.0f, 0.0f);
            auto base_ddy = make_float3(0.0f, 0.25f, 0.0f);
            voxel_output.write(
                5u, sampled.sample(
                        make_float3(0.375f, 0.625f, 0.375f),
                        base_ddx, base_ddy, 2.0f));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<float4, 64u> mip0;
        std::array<float4, 8u> mip1;
        constexpr std::array mip2{final_voxel};
        mip0.fill(base_voxel);
        mip1.fill(middle_voxel);
        constexpr std::array<uint32_t, 2u> control_source{
            0u, selected_level};
        std::array<uint3, 3u> size_result{};
        std::array<float4, 6u> voxel_result{};
        stream << volume.view(0u).copy_from(luisa::span{mip0})
               << volume.view(1u).copy_from(luisa::span{mip1})
               << volume.view(2u).copy_from(luisa::span{mip2})
               << controls.copy_from(luisa::span{control_source})
               << heap.emplace_on_update(
                          0u, volume, Sampler::linear_linear_edge())
                      .update()
               << shader(volume.view(1u), heap, controls,
                         sizes, voxels)
                      .dispatch(1u)
               << sizes.copy_to(luisa::span{size_result})
               << voxels.copy_to(luisa::span{voxel_result})
               << synchronize();

        constexpr std::array expected_sizes{
            make_uint3(2u), base_size, make_uint3(1u)};
        for (auto i = 0u; i < size_result.size(); ++i) {
            expect(size_result[i].x == expected_sizes[i].x &&
                   size_result[i].y == expected_sizes[i].y &&
                   size_result[i].z == expected_sizes[i].z)
                << luisa::format(
                       "volume size query {} mismatch: expected {}x{}x{}, got {}x{}x{}",
                       i, expected_sizes[i].x, expected_sizes[i].y,
                       expected_sizes[i].z, size_result[i].x,
                       size_result[i].y, size_result[i].z);
        }
        expect_vector_equal(voxel_result[0], final_voxel);
        expect_vector_equal(voxel_result[1], base_voxel);
        expect_vector_equal(voxel_result[2], final_voxel);
        expect_vector_equal(voxel_result[3], middle_voxel);
        expect_vector_equal(voxel_result[4], middle_voxel);
        expect_vector_equal(voxel_result[5], final_voxel);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "volume mip regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto size_query_count =
                count_spirv_opcode(disassembly, "ImageQuerySize") +
                count_spirv_opcode(disassembly, "ImageQuerySizeLod");
            expect(size_query_count == 3u)
                << "each direct/bindless volume size query must remain an "
                   "image query at SPIR-V opt0";
            expect(count_spirv_opcode(disassembly, "ImageFetch") == 2u)
                << "direct-view and explicit-level bindless volume reads "
                   "must lower to OpImageFetch";
            expect(count_spirv_opcode(
                       disassembly, "ImageSampleExplicitLod") == 4u)
                << "compute volume sampling must use explicit LOD operands, "
                   "including the plain sample form";
            expect(count_spirv_opcode(
                       disassembly, "ImageSampleImplicitLod") == 0u)
                << "compute shaders must not derive implicit volume LOD";
            expect(count_substring(disassembly, " Grad ") == 2u)
                << "both volume gradient samples must carry the Grad image "
                   "operand exactly once";
            expect(count_substring(disassembly, " Grad MinLod ") == 1u)
                << "minimum-LOD volume gradient sampling must carry combined "
                   "Grad and MinLod image operands exactly once";
        }
    };

    "vk_custom_bindless_buffer_write_has_matching_native_barrier"_test = [&] {
        auto dc = create_native_command_device(argc, argv);
        auto target = dc.device.create_buffer<uint32_t>(4u);
        auto heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::BUFFER_ONLY);
        auto stream = dc.device.create_stream();

        heap.emplace_on_update(0u, target);
        constexpr auto fill_value = 0x6a09e667u;
        constexpr std::array<uint32_t, 4u> zero{};
        std::array<uint32_t, 4u> result{};
        // Prime in a completed submission so the tested submission has no
        // stale direct-resource state that could mask the custom write scope.
        stream << target.copy_from(luisa::span{zero})
               << synchronize();
        stream << heap.update()
               << luisa::make_unique<BindlessBufferFillCommand>(
                      heap, target, fill_value)
               << target.copy_to(luisa::span{result})
               << synchronize();

        expect(result == std::array<uint32_t, 4u>{
                             fill_value, fill_value,
                             fill_value, fill_value})
            << "a custom bindless-member transfer write must be visible to "
               "the following direct buffer read";
    };

    "vk_custom_bindless_texture_read_waits_for_upload"_test = [&] {
        auto dc = create_native_command_device(argc, argv);
        constexpr auto size = make_uint2(1u);
        auto source = dc.device.create_image<float>(
            PixelStorage::FLOAT4, size);
        auto heap = dc.device.create_bindless_array(
            1u, BindlessSlotType::TEXTURE2D_ONLY);
        auto target = dc.device.create_buffer<float4>(1u);
        auto stream = dc.device.create_stream();

        heap.emplace_on_update(
            0u, source, Sampler::point_zero());
        constexpr std::array sentinel{
            float4{0.875f, 0.75f, 0.625f, 0.5f}};
        constexpr std::array pixels{
            float4{0.125f, 0.25f, 0.5f, 1.0f}};
        std::array<float4, 1u> result{};
        // A sentinel distinct from the expected pixels makes both an omitted
        // copy and a stale image read deterministic failures.
        stream << source.copy_from(luisa::span{sentinel})
               << target.copy_from(luisa::span{sentinel})
               << synchronize();
        stream << source.copy_from(luisa::span{pixels})
               << heap.update()
               << luisa::make_unique<BindlessTextureCopyCommand>(
                      heap, source, target)
               << target.copy_to(luisa::span{result})
               << synchronize();

        expect_vector_equal(result[0], pixels[0]);
    };

    "vk_config_bindless_states_cover_encoded_texture_mips"_test = [&] {
        Context context{argv[0]};
        DeviceConfig config{};
        auto config_ext = luisa::make_unique<BindlessConfigStateExt>();
        auto config_ext_ptr = config_ext.get();
        config.extension = std::move(config_ext);
        auto device = context.create_device("vk", &config);
        auto stream = device.create_stream(StreamTag::COMPUTE);

        constexpr auto mip_levels = 2u;
        auto texture = device.create_image<float>(
            PixelStorage::FLOAT4, make_uint2(2u), mip_levels);
        auto heap = device.create_bindless_array(1u);
        auto shader_output = device.create_buffer<float4>(1u);
        auto native_output = device.create_buffer<float4>(1u);
        auto native_upload = device.create_buffer<float4>(1u);

        Kernel1D read_mip = [](BindlessVar bindless,
                               BufferFloat4 output) noexcept {
            output.write(
                0u,
                bindless.tex2d(0u).read(make_uint2(0u), 1u));
        };
        auto shader = device.compile(
            read_mip,
            ShaderOption{.enable_cache = false,
                         .enable_fast_math = false});

        constexpr std::array mip0{
            float4{0.1f, 0.2f, 0.3f, 1.0f},
            float4{0.2f, 0.3f, 0.4f, 1.0f},
            float4{0.3f, 0.4f, 0.5f, 1.0f},
            float4{0.4f, 0.5f, 0.6f, 1.0f}};
        constexpr std::array mip1_before{
            float4{0.125f, 0.25f, 0.5f, 1.0f}};
        constexpr std::array mip1_after{
            float4{0.875f, 0.75f, 0.625f, 0.5f}};
        std::array<float4, 1u> observed{};

        heap.emplace_on_update(
            0u, texture, Sampler::point_zero());
        stream << texture.view(0u).copy_from(luisa::span{mip0})
               << texture.view(1u).copy_from(luisa::span{mip1_before})
               << heap.update()
               << synchronize();

        // A bindless after-state must transition every encoded image mip,
        // even though ResourceUsage names only the array handle. The raw copy
        // deliberately relies on that published native layout contract.
        config_ext_ptr->set_after_state(
            heap, VKCustomCmd::ResourceUsageType::CopySource);
        stream << shader(heap, shader_output).dispatch(1u)
               << synchronize();
        config_ext_ptr->clear_states();
        stream << luisa::make_unique<ConfiguredTextureMipCopyCommand>(
                      texture, native_output, 1u)
               << native_output.copy_to(luisa::span{observed})
               << synchronize();
        expect_vector_equal(observed[0], mip1_before[0]);

        // Simulate native work outside Luisa's tracker. It transitions the
        // complete image to TRANSFER_DST and overwrites mip 1. The next
        // submission publishes that state through the bindless before hook;
        // the regular bindless shader must then transition and observe mip 1.
        stream << native_upload.copy_from(luisa::span{mip1_after})
               << luisa::make_unique<ConfiguredTextureMipOverwriteCommand>(
                      native_upload, texture, 1u)
               << synchronize();
        config_ext_ptr->set_before_state(
            heap,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT |
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        stream << shader(heap, shader_output).dispatch(1u)
               << shader_output.copy_to(luisa::span{observed})
               << synchronize();
        config_ext_ptr->clear_states();
        expect_vector_equal(observed[0], mip1_after[0]);
    };

    "vk_user_compute_overlapping_buffer_arguments_preserve_aliasing"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_overlapping_buffer_arguments"};
        ScopedSourceDump source_dump;
        constexpr auto total_count = 16u;
        auto dc = luisa::test::create_device(argc, argv);
        auto values = dc.device.create_buffer<uint32_t>(total_count);
        auto observed = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt read_view,
                             BufferUInt write_view,
                             BufferUInt out) noexcept {
            // read_view[2] and write_view[0] deliberately name the same word.
            auto before = read_view.read(2u);
            write_view.write(0u, before + 0x1234u);
            out.write(0u, before);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, total_count> source{};
        for (auto i = 0u; i < source.size(); ++i) {
            source[i] = 0x10000u + i * 13u;
        }
        constexpr auto read_base = 2u;
        constexpr auto write_base = 4u;
        auto before = source[write_base];
        auto after = before + 0x1234u;
        auto expected_values = source;
        expected_values[write_base] = after;
        std::array<uint32_t, 1u> result{};
        std::array<uint32_t, total_count> updated{};
        stream << values.copy_from(luisa::span{source})
               << shader(values.view(read_base, 8u),
                         values.view(write_base, 8u),
                         observed)
                      .dispatch(1u)
               << observed.copy_to(luisa::span{result})
               << values.copy_to(luisa::span{updated})
               << synchronize();

        expect(result[0] == before);
        expect(updated == expected_values)
            << "the writable overlapping view must update only the shared target word";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "overlapping buffer regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto read_buffer_id = spirv_id_named(disassembly, "_buf_0");
            expect(read_buffer_id.has_value())
                << "overlapping buffer regression should retain the named read descriptor";
            if (read_buffer_id) {
                expect(spirv_id_has_decoration(
                    disassembly, *read_buffer_id, "Aliased"));
                expect(!spirv_id_has_decoration(
                    disassembly, *read_buffer_id, "NonWritable"))
                    << "a read descriptor that may alias a writable user buffer cannot claim its backing memory is NonWritable";
            }
            expect(count_substring(disassembly, "Coherent") == 0u)
                << "ordinary writable/aliased buffers must not make internal descriptor blocks coherent";
        }
    };

    "vk_user_compute_indirect_author_buffer_alias_contract"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_indirect_author_buffer_alias"};
        ScopedSourceDump source_dump;
        auto dc = luisa::test::create_device(argc, argv);
        auto commands = dc.device.create_indirect_dispatch_buffer(1u);
        auto alias = dc.device.import_external_buffer<uint32_t>(
            commands.native_handle(),
            commands.size_bytes() / sizeof(uint32_t));
        auto observed = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();

        Kernel1D kernel = [](BufferUInt read_alias,
                             Var<IndirectDispatchBuffer> target,
                             BufferUInt output) noexcept {
            UInt before = read_alias.read(0u);
            target.set_dispatch_count(1u);
            target.set_kernel(
                0u, make_uint3(1u), make_uint3(1u), 0u);
            output.write(0u, before);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, 1u> before{};
        stream << shader(alias, commands, observed).dispatch(1u)
               << observed.copy_to(luisa::span{before})
               << synchronize();
        expect(before[0] == 0u)
            << "the author must observe the initialized header before overwriting it";

        std::array<uint32_t, 1u> authored_header{};
        stream << alias.view(0u, 1u).copy_to(
                      luisa::span{authored_header})
               << synchronize();
        expect(authored_header[0] == 1u);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "indirect author alias regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            auto read_buffer_id = spirv_id_named(disassembly, "_buf_0");
            auto indirect_id = spirv_id_named(
                disassembly, "_indirect_dispatch");
            expect(read_buffer_id.has_value());
            expect(indirect_id.has_value());
            if (read_buffer_id) {
                expect(spirv_id_has_decoration(
                    disassembly, *read_buffer_id, "Aliased"));
                expect(!spirv_id_has_decoration(
                    disassembly, *read_buffer_id, "NonWritable"));
            }
            if (indirect_id) {
                expect(spirv_id_has_decoration(
                    disassembly, *indirect_id, "Aliased"));
            }
        }
    };

    "vk_native_buffer_aliases_share_direct_and_bindless_hazards"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto owner = dc.device.create_buffer<uint32_t>(1u);
        auto alias = dc.device.import_external_buffer<uint32_t>(
            owner.native_handle(), 1u);
        expect(owner.native_handle() != nullptr);
        expect(alias.native_handle() == owner.native_handle());
        expect(alias.handle() != owner.handle());
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();

        Kernel1D direct_kernel = [](BufferUInt target,
                                    UInt value) noexcept {
            target.write(0u, value);
        };
        Kernel1D bindless_kernel = [](BindlessVar bindless,
                                      UInt value) noexcept {
            bindless.buffer<uint32_t>(0u).write(0u, value);
        };
        auto direct_shader = dc.device.compile(
            direct_kernel, ShaderOption{.enable_cache = false});
        auto bindless_shader = dc.device.compile(
            bindless_kernel, ShaderOption{.enable_cache = false});

        constexpr std::array initial{0x01020304u};
        constexpr std::array sentinel{0xfeedfaceu};
        constexpr auto direct_value = 0x11223344u;
        std::array<uint32_t, 1u> direct_result{};
        stream << owner.copy_from(luisa::span{sentinel})
               << synchronize();
        stream << owner.copy_from(luisa::span{initial})
               << direct_shader(alias, direct_value).dispatch(1u)
               << owner.copy_to(luisa::span{direct_result})
               << synchronize();
        expect(direct_result[0] == direct_value)
            << "native VkBuffer identity must order hazards across distinct Luisa wrappers";

        heap.emplace_on_update(0u, alias);
        constexpr auto bindless_value = 0xa1b2c3d4u;
        std::array<uint32_t, 1u> bindless_result{};
        stream << owner.copy_from(luisa::span{sentinel})
               << synchronize();
        stream << owner.copy_from(luisa::span{initial})
               << heap.update()
               << bindless_shader(heap, bindless_value).dispatch(1u)
               << owner.copy_to(luisa::span{bindless_result})
               << synchronize();
        expect(bindless_result[0] == bindless_value)
            << "bindless snapshots and Vulkan barriers must use the same native buffer identity";
    };

    "vk_native_image_aliases_share_layout_and_access_hazards"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        constexpr auto size = make_uint2(2u);
        constexpr auto mip_levels = 2u;
        auto owner = dc.device.create_image<float>(
            PixelStorage::FLOAT4, size, mip_levels);
        auto observed = dc.device.create_buffer<float4>(1u);
        auto stream = dc.device.create_stream();

        constexpr auto delta = float4{0.125f, 0.25f, 0.5f, 1.0f};
        Kernel1D kernel = [](ImageFloat target,
                             BufferFloat4 output) noexcept {
            auto value = target.read(make_uint2(0u));
            output.write(0u, value);
            target.write(
                make_uint2(0u),
                value + make_float4(0.125f, 0.25f, 0.5f, 1.0f));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array mip0_source{
            float4{1.0f, 2.0f, 3.0f, 4.0f},
            float4{5.0f, 6.0f, 7.0f, 8.0f},
            float4{9.0f, 10.0f, 11.0f, 12.0f},
            float4{13.0f, 14.0f, 15.0f, 16.0f}};
        constexpr std::array mip1_source{
            float4{17.0f, 18.0f, 19.0f, 20.0f}};
        auto native_image = owner.native_handle();
        expect(native_image != nullptr);
        expect(owner.mip_levels() == mip_levels);

        // Establish the owner's per-mip state before the alias exists. The
        // imported wrapper must join this state in a later submission.
        stream << owner.view(0u).copy_from(luisa::span{mip0_source})
               << owner.view(1u).copy_from(luisa::span{mip1_source})
               << synchronize();

        std::array<float4, 1u> alias_before{};
        std::array<float4, 4u> mip0_after_alias{};
        std::array<float4, 1u> mip1_after_alias{};
        {
            auto alias = dc.device.import_external_image<float>(
                PixelStorage::FLOAT4, native_image, size, mip_levels);
            expect(alias.native_handle() == native_image);
            expect(alias.handle() != owner.handle());

            stream << shader(alias.view(1u), observed).dispatch(1u)
                   << observed.copy_to(luisa::span{alias_before})
                   << owner.view(0u).copy_to(luisa::span{mip0_after_alias})
                   << owner.view(1u).copy_to(luisa::span{mip1_after_alias})
                   << synchronize();
        }

        expect_vector_equal(alias_before[0], mip1_source[0]);
        expect_vector_equal(mip1_after_alias[0], mip1_source[0] + delta);
        for (auto i = 0u; i < mip0_source.size(); ++i) {
            expect_vector_equal(mip0_after_alias[i], mip0_source[i]);
        }

        // Destroying the imported wrapper must neither destroy the VkImage nor
        // discard the surviving owner's per-mip state.
        std::array<float4, 1u> owner_before{};
        std::array<float4, 4u> mip0_after_owner{};
        std::array<float4, 1u> mip1_after_owner{};
        expect(owner.native_handle() == native_image);
        stream << shader(owner.view(0u), observed).dispatch(1u)
               << observed.copy_to(luisa::span{owner_before})
               << owner.view(0u).copy_to(luisa::span{mip0_after_owner})
               << owner.view(1u).copy_to(luisa::span{mip1_after_owner})
               << synchronize();

        expect_vector_equal(owner_before[0], mip0_source[0]);
        expect_vector_equal(mip0_after_owner[0], mip0_source[0] + delta);
        for (auto i = 1u; i < mip0_source.size(); ++i) {
            expect_vector_equal(mip0_after_owner[i], mip0_source[i]);
        }
        expect_vector_equal(mip1_after_owner[0], mip1_source[0] + delta);
    };

    "vk_cross_queue_events_preserve_owned_resources_and_shared_mip_layouts"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto copy_stream = device.create_stream(StreamTag::COPY);
        auto compute_stream = device.create_stream(StreamTag::COMPUTE);
        auto graphics_stream = device.create_stream(StreamTag::GRAPHICS);
        auto handoff = device.create_timeline_event();

        // Vulkan creates one queue at index zero for every selected family, so
        // distinct native queue handles identify the cross-family case here.
        // The same-family configuration remains a useful event/layout test,
        // but cannot exercise the concurrent-sharing branch of resource creation.
        std::array native_queues{
            copy_stream.native_handle(),
            compute_stream.native_handle(),
            graphics_stream.native_handle()};
        auto distinct_native_queue_count = 0u;
        for (auto i = 0u; i < native_queues.size(); ++i) {
            auto first_occurrence = true;
            for (auto j = 0u; j < i; ++j) {
                first_occurrence &= native_queues[j] != native_queues[i];
            }
            distinct_native_queue_count += static_cast<uint>(first_occurrence);
        }
        if (distinct_native_queue_count == 1u) {
            LUISA_INFO(
                "Vulkan maps COPY, COMPUTE, and GRAPHICS to one native queue; "
                "running the ordered same-family variant of the cross-queue test.");
        }

        constexpr auto base_size = make_uint2(4u);
        constexpr auto mip_size = make_uint2(2u);
        constexpr auto mip_levels = 2u;
        constexpr auto element_count = 4u;
        auto values = device.create_buffer<uint32_t>(element_count);
        auto image = device.create_image<float>(
            PixelStorage::FLOAT4, base_size, mip_levels, false);

        constexpr std::array<uint32_t, element_count> buffer_source{
            3u, 5u, 7u, 11u};
        std::array<float4, 16u> mip0_source{};
        for (auto i = 0u; i < mip0_source.size(); ++i) {
            auto value = static_cast<float>(i + 1u);
            mip0_source[i] = float4{value, value + 16.0f,
                                    value + 32.0f, 1.0f};
        }
        constexpr std::array<float4, 4u> mip1_source{
            float4{-1.0f, -2.0f, -3.0f, -4.0f},
            float4{-5.0f, -6.0f, -7.0f, -8.0f},
            float4{-9.0f, -10.0f, -11.0f, -12.0f},
            float4{-13.0f, -14.0f, -15.0f, -16.0f}};

        Kernel2D compute_stage = [](BufferUInt buffer,
                                    ImageFloat source,
                                    ImageFloat target) noexcept {
            auto coord = dispatch_id().xy();
            auto lane = coord.y * 2u + coord.x;
            auto color = source.read(coord * 2u);
            target.write(
                coord, color + make_float4(0.25f, 0.5f, 0.75f, 1.0f));
            buffer.write(lane, buffer.read(lane) + 100u + lane);
        };
        Kernel2D graphics_stage = [](BufferUInt buffer,
                                     ImageFloat source,
                                     ImageFloat target) noexcept {
            auto coord = dispatch_id().xy();
            auto lane = coord.y * 2u + coord.x;
            target.write(coord, source.read(coord) * 2.0f);
            buffer.write(lane, buffer.read(lane) * 2u + 1u);
        };
        auto compute_shader = device.compile(
            compute_stage, ShaderOption{.enable_cache = false,
                                        .enable_fast_math = false});
        auto graphics_shader = device.compile(
            graphics_stage, ShaderOption{.enable_cache = false,
                                         .enable_fast_math = false});

        // COPY owns the first use and explicitly publishes it before a late
        // non-owning wrapper joins the same native image state.
        copy_stream << values.copy_from(luisa::span{buffer_source})
                    << image.view(0u).copy_from(luisa::span{mip0_source})
                    << image.view(1u).copy_from(luisa::span{mip1_source})
                    << handoff.signal(1u);
        auto image_alias = device.import_external_image<float>(
            PixelStorage::FLOAT4, image.native_handle(),
            base_size, mip_levels, false);
        expect(image_alias.native_handle() == image.native_handle());
        expect(image_alias.handle() != image.handle());

        // The false simultaneous-access policy intentionally keeps optimized
        // per-mip layouts. Events serialize every handoff; concurrent Vulkan
        // sharing removes only queue-family ownership transfers.
        compute_stream << handoff.wait(1u)
                       << compute_shader(values, image_alias.view(0u),
                                         image_alias.view(1u))
                              .dispatch(mip_size)
                       << handoff.signal(2u);
        graphics_stream << handoff.wait(2u)
                        << graphics_shader(values, image.view(1u),
                                           image.view(0u))
                               .dispatch(mip_size)
                        << handoff.signal(3u);

        std::array<uint32_t, element_count> buffer_result{};
        std::array<float4, 16u> mip0_result{};
        std::array<float4, 4u> mip1_result{};
        copy_stream << handoff.wait(3u)
                    << values.copy_to(luisa::span{buffer_result})
                    << image.view(0u).copy_to(luisa::span{mip0_result})
                    << image.view(1u).copy_to(luisa::span{mip1_result})
                    << synchronize();

        auto expected_mip0 = mip0_source;
        auto expected_mip1 = mip1_source;
        constexpr auto compute_delta =
            float4{0.25f, 0.5f, 0.75f, 1.0f};
        for (auto y = 0u; y < mip_size.y; ++y) {
            for (auto x = 0u; x < mip_size.x; ++x) {
                auto lane = y * mip_size.x + x;
                auto source_index = (y * 2u) * base_size.x + x * 2u;
                expected_mip1[lane] =
                    mip0_source[source_index] + compute_delta;
                expected_mip0[y * base_size.x + x] =
                    expected_mip1[lane] * 2.0f;
            }
        }
        for (auto i = 0u; i < element_count; ++i) {
            auto expected = (buffer_source[i] + 100u + i) * 2u + 1u;
            expect(buffer_result[i] == expected)
                << luisa::format(
                       "cross-queue buffer mismatch at {}: expected {}, got {}",
                       i, expected, buffer_result[i]);
        }
        for (auto i = 0u; i < mip0_result.size(); ++i) {
            expect_vector_equal(mip0_result[i], expected_mip0[i]);
        }
        for (auto i = 0u; i < mip1_result.size(); ++i) {
            expect_vector_equal(mip1_result[i], expected_mip1[i]);
        }
    };

    "vk_user_compute_argument_buffer_trailing_half_has_full_word"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto output = dc.device.create_buffer<float>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](Half value, BufferFloat out) noexcept {
            out.write(0u, value.cast<float>());
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        float result = 0.0f;
        stream << shader(half{1.5f}, output).dispatch(1u)
               << output.copy_to(luisa::span{&result, 1u})
               << synchronize();
        expect(result == 1.5f)
            << "a trailing 16-bit uniform must have a complete descriptor-visible word";
    };

    "vk_user_compute_parallel_subword_writes_preserve_shared_words"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_parallel_subword"};
        ScopedSourceDump source_dump;
        constexpr auto word_count = 256u;
        constexpr auto byte_count = word_count * sizeof(uint32_t);
        auto dc = luisa::test::create_device(argc, argv);
        auto bytes = dc.device.create_byte_buffer(byte_count);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](ByteBufferVar buffer) noexcept {
            set_block_size(64u, 1u, 1u);
            buffer.write(dispatch_x(), true);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, word_count> initial{};
        std::array<uint32_t, word_count> result{};
        stream << bytes.copy_from(initial.data())
               << shader(bytes).dispatch(byte_count)
               << bytes.copy_to(result.data())
               << synchronize();
        for (auto i = 0u; i < result.size(); i++) {
            expect(result[i] == 0x01010101u)
                << luisa::format(
                       "concurrent byte lanes lost an update in packed word {}: got 0x{:08x}",
                       i, result[i]);
        }
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "the packed-byte fixture should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(
                       disassembly, "AtomicCompareExchange") >= 1u)
                << "packed byte writes must select the masked uint32 CAS "
                   "fallback independently of scheduling luck";
            expect(count_spirv_opcode(disassembly, "LoopMerge") >= 1u)
                << "the masked write must retry through a structured CAS loop";
        }
    };

    "vk_user_compute_word_backed_nested_signed_atomics"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto values =
            dc.device.create_buffer<WordBackedSignedAtomicComposite>(1u);
        auto old_values = dc.device.create_buffer<int32_t>(4u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](
                              BufferVar<WordBackedSignedAtomicComposite> buffer,
                              BufferInt old) noexcept {
            auto atomic = buffer.atomic(0u);
            old.write(0u, atomic.values[0].fetch_add(-7));
            old.write(1u, atomic.values[1].fetch_min(-20));
            old.write(2u, atomic.values[0].compare_exchange(-17, 9));
            old.write(3u, atomic.values[1].compare_exchange(-19, 11));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{WordBackedSignedAtomicComposite{
            {true, false, true, false}, {-10, -5}}};
        std::array<WordBackedSignedAtomicComposite, 1u> result{};
        std::array<int32_t, 4u> observed_old{};
        stream << values.copy_from(luisa::span{source})
               << shader(values, old_values).dispatch(1u)
               << values.copy_to(luisa::span{result})
               << old_values.copy_to(luisa::span{observed_old})
               << synchronize();

        constexpr std::array expected_old{-10, -5, -17, -20};
        expect(observed_old == expected_old)
            << "signed word-backed atomics must return exact negative old values";
        expect(result[0].values[0] == 9);
        expect(result[0].values[1] == -20);
        constexpr std::array expected_flags{true, false, true, false};
        for (auto i = 0u; i < expected_flags.size(); i++) {
            expect(result[0].flags[i] == expected_flags[i])
                << luisa::format(
                       "signed nested atomic clobbered bool canary {}", i);
        }
    };

    "vk_user_compute_matrix_operation_shapes_are_exact"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_matrix_shapes"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto matrices = dc.device.create_buffer<float2x2>(2u);
        auto vectors = dc.device.create_buffer<float2>(2u);
        auto matrix_output = dc.device.create_buffer<float4>(4u);
        auto vector_output = dc.device.create_buffer<float2>(2u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat2x2 matrix_input,
                             BufferFloat2 vector_input,
                             BufferFloat4 matrix_out,
                             BufferFloat2 vector_out) noexcept {
            auto a = matrix_input.read(0u);
            auto b = matrix_input.read(1u);
            auto v = vector_input.read(0u);
            auto w = vector_input.read(1u);
            auto matrix_product = a * b;
            auto vector_outer_product = def<float2x2>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float2x2>(), CallOp::OUTER_PRODUCT,
                    {v.expression(), w.expression()}));
            auto matrix_outer_product = def<float2x2>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float2x2>(), CallOp::OUTER_PRODUCT,
                    {a.expression(), b.expression()}));
            auto scaled_matrix = a * 2.0f;
            auto vector_matrix_product = def<float2>(
                luisa::compute::detail::FunctionBuilder::current()->binary(
                    Type::of<float2>(), BinaryOp::MUL,
                    v.expression(), a.expression()));
            matrix_out.write(
                0u, make_float4(matrix_product[0u].x,
                                matrix_product[0u].y,
                                matrix_product[1u].x,
                                matrix_product[1u].y));
            matrix_out.write(
                1u, make_float4(vector_outer_product[0u].x,
                                vector_outer_product[0u].y,
                                vector_outer_product[1u].x,
                                vector_outer_product[1u].y));
            matrix_out.write(
                2u, make_float4(scaled_matrix[0u].x,
                                scaled_matrix[0u].y,
                                scaled_matrix[1u].x,
                                scaled_matrix[1u].y));
            matrix_out.write(
                3u, make_float4(matrix_outer_product[0u].x,
                                matrix_outer_product[0u].y,
                                matrix_outer_product[1u].x,
                                matrix_outer_product[1u].y));
            vector_out.write(0u, a * v);
            vector_out.write(1u, vector_matrix_product);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array matrix_source{
            make_float2x2(1.0f, 2.0f, 3.0f, 4.0f),
            make_float2x2(5.0f, 6.0f, 7.0f, 8.0f)};
        constexpr std::array vector_source{
            float2{2.0f, -1.0f}, float2{-3.0f, 5.0f}};
        std::array<float4, 4u> matrix_result{};
        std::array<float2, 2u> vector_result{};
        stream << matrices.copy_from(luisa::span{matrix_source})
               << vectors.copy_from(luisa::span{vector_source})
               << shader(matrices, vectors, matrix_output, vector_output)
                      .dispatch(1u)
               << matrix_output.copy_to(luisa::span{matrix_result})
               << vector_output.copy_to(luisa::span{vector_result})
               << synchronize();

        expect_vector_equal(
            matrix_result[0], float4{23.0f, 34.0f, 31.0f, 46.0f});
        expect_vector_equal(
            matrix_result[1], float4{-6.0f, 3.0f, 10.0f, -5.0f});
        expect_vector_equal(
            matrix_result[2], float4{2.0f, 4.0f, 6.0f, 8.0f});
        expect_vector_equal(
            matrix_result[3], float4{26.0f, 38.0f, 30.0f, 44.0f});
        expect_vector_equal(vector_result[0], float2{-1.0f, 0.0f});
        expect_vector_equal(vector_result[1], float2{0.0f, 2.0f});

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "matrix-shape regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(
                       disassembly, "MatrixTimesMatrix") == 2u)
                << "matrix multiplication and generalized matrix outer product must each emit one OpMatrixTimesMatrix";
            expect(count_spirv_opcode(
                       disassembly, "MatrixTimesVector") == 1u);
            expect(count_spirv_opcode(
                       disassembly, "VectorTimesMatrix") == 1u);
            expect(count_spirv_opcode(
                       disassembly, "OuterProduct") == 1u);
            expect(count_spirv_opcode(
                       disassembly, "MatrixTimesScalar") == 1u);
            expect(count_spirv_opcode(
                       disassembly, "Transpose") == 1u)
                << "generalized matrix outer product must transpose its right operand exactly once";
        }
    };

    "vk_user_compute_non_fast_math_does_not_contract_mul_add"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_no_contraction"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto lhs = dc.device.create_buffer<float>(1u);
        auto rhs = dc.device.create_buffer<float>(1u);
        auto addend = dc.device.create_buffer<float>(1u);
        auto output = dc.device.create_buffer<uint32_t>(1u);
        auto reduction_output = dc.device.create_buffer<uint32_t>(1u);
        auto dot_output = dc.device.create_buffer<uint2>(1u);
        auto matrix_lhs = dc.device.create_buffer<float2x2>(1u);
        auto matrix_addend = dc.device.create_buffer<float2x2>(1u);
        auto matrix_output = dc.device.create_buffer<uint4>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat a,
                             BufferFloat b,
                             BufferFloat c,
                             BufferUInt out,
                             BufferUInt reduction_out,
                             BufferUInt2 dot_out,
                             BufferFloat2x2 matrix_a,
                             BufferFloat2x2 matrix_c,
                             BufferUInt4 matrix_out) noexcept {
            auto product = a.read(0u) * b.read(0u);
            out.write(0u, as<uint>(product + c.read(0u)));
            auto reduction_product = reduce_prod(
                make_float2(a.read(0u), b.read(0u)));
            auto reduction_sum = reduce_sum(
                make_float2(reduction_product, c.read(0u)));
            reduction_out.write(0u, as<uint>(reduction_sum));
            auto lhs_value = a.read(0u);
            auto rhs_value = b.read(0u);
            auto addend_value = c.read(0u);
            auto dot_result = dot(
                make_float2(lhs_value, 1.0f),
                make_float2(rhs_value, addend_value));
            auto squared_result = length_squared(
                make_float2(lhs_value, rhs_value));
            dot_out.write(
                0u, make_uint2(
                        as<uint>(dot_result),
                        as<uint>(squared_result)));
            auto matrix_product = matrix_a.read(0u) * b.read(0u);
            auto matrix_sum = matrix_product + matrix_c.read(0u);
            matrix_out.write(
                0u, as<uint4>(make_float4(
                        matrix_sum[0u].x, matrix_sum[0u].y,
                        matrix_sum[1u].x, matrix_sum[1u].y)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr auto multiplicand_bits = 0x3f800001u;
        constexpr auto addend_bits = 0xbf800002u;
        std::array multiplicand{std::bit_cast<float>(multiplicand_bits)};
        std::array addend_value{std::bit_cast<float>(addend_bits)};
        uint32_t result = 0xffffffffu;
        uint32_t reduction_result = 0xffffffffu;
        uint2 dot_result{~0u};
        auto repeated_matrix = [](float value) noexcept {
            return make_float2x2(value, value, value, value);
        };
        std::array matrix_multiplicand{
            repeated_matrix(multiplicand[0])};
        std::array matrix_addend_value{
            repeated_matrix(addend_value[0])};
        uint4 matrix_result{~0u};
        stream << lhs.copy_from(luisa::span{multiplicand})
               << rhs.copy_from(luisa::span{multiplicand})
               << addend.copy_from(luisa::span{addend_value})
               << matrix_lhs.copy_from(luisa::span{matrix_multiplicand})
               << matrix_addend.copy_from(
                      luisa::span{matrix_addend_value})
               << shader(lhs, rhs, addend, output, reduction_output,
                         dot_output,
                         matrix_lhs, matrix_addend, matrix_output)
                      .dispatch(1u)
               << output.copy_to(luisa::span{&result, 1u})
               << reduction_output.copy_to(
                      luisa::span{&reduction_result, 1u})
               << dot_output.copy_to(
                      luisa::span{&dot_result, 1u})
               << matrix_output.copy_to(
                      luisa::span{&matrix_result, 1u})
               << synchronize();
        expect(result == 0u)
            << luisa::format(
                   "non-contracted (1+2^-23)^2-(1+2^-22) must round to +0, got bits 0x{:08x}",
                   result);
        expect(reduction_result == 0u)
            << luisa::format(
                   "non-contracted reduction product/sum must round to +0, got bits 0x{:08x}",
                   reduction_result);
        expect(dot_result.x == 0u)
            << luisa::format(
                   "non-contracted dot((1+2^-23, 1), (1+2^-23, -(1+2^-22))) must round to +0, got bits 0x{:08x}",
                   dot_result.x);
        expect(dot_result.y == 0x40000002u)
            << luisa::format(
                   "length_squared((1+2^-23, 1+2^-23)) produced unexpected bits 0x{:08x}",
                   dot_result.y);
        expect_vector_equal(matrix_result, uint4{0u});
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "non-contraction regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "FMul") == 2u)
                << "ordinary and reduction multiplications must remain OpFMul";
            expect(count_spirv_opcode(disassembly, "FAdd") == 4u)
                << "ordinary, reduction, and matrix additions must remain OpFAdd";
            expect(count_spirv_opcode(
                       disassembly, "MatrixTimesScalar") == 1u)
                << "matrix scaling must remain one OpMatrixTimesScalar";
            expect(count_spirv_opcode(disassembly, "Dot") == 2u)
                << "dot and length_squared must remain two OpDot instructions";
            expect(count_spirv_extended_instruction(disassembly, "Fma") == 0u)
                << "non-fast-math SPIR-V must not contain a fused Fma instruction";
            expect(count_substring(disassembly, "NoContraction") == 9u)
                << "all ordinary, reduction, dot, and matrix multiply/add results must carry NoContraction";
        }
    };

    "vk_user_compute_fast_math_fuses_single_use_mul_add_exactly"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_fast_math_fma"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto lhs = dc.device.create_buffer<float>(1u);
        auto rhs = dc.device.create_buffer<float>(1u);
        auto addend = dc.device.create_buffer<float>(1u);
        auto output = dc.device.create_buffer<uint32_t>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat a,
                             BufferFloat b,
                             BufferFloat c,
                             BufferUInt out) noexcept {
            auto product = a.read(0u) * b.read(0u);
            out.write(0u, as<uint>(product + c.read(0u)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = true});

        constexpr auto multiplicand_bits = 0x3f800001u;
        constexpr auto addend_bits = 0xbf800002u;
        std::array multiplicand{std::bit_cast<float>(multiplicand_bits)};
        std::array addend_value{std::bit_cast<float>(addend_bits)};
        uint32_t result = 0xffffffffu;
        stream << lhs.copy_from(luisa::span{multiplicand})
               << rhs.copy_from(luisa::span{multiplicand})
               << addend.copy_from(luisa::span{addend_value})
               << shader(lhs, rhs, addend, output).dispatch(1u)
               << output.copy_to(luisa::span{&result, 1u})
               << synchronize();
        auto expected = std::bit_cast<uint32_t>(std::fma(
            multiplicand[0], multiplicand[0], addend_value[0]));
        expect(result == expected)
            << luisa::format(
                   "fast-math FMA result mismatch: expected bits 0x{:08x}, got 0x{:08x}",
                   expected, result);
        expect(result != 0u)
            << "the fused fixture must distinguish FMA from separate multiply/add";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "fast-math FMA regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_extended_instruction(disassembly, "Fma") == 1u)
                << "single-use fast-math multiply/add must emit one fused Fma";
            expect(count_spirv_opcode(disassembly, "FMul") == 0u)
                << "single-use fused multiply must not leave a dead OpFMul";
            expect(count_spirv_opcode(disassembly, "FAdd") == 0u)
                << "single-use fused add must not leave a separate OpFAdd";
            expect(count_substring(disassembly, "NoContraction") == 0u)
                << "fast-math FMA must not carry NoContraction";
        }
    };

    "vk_user_compute_fast_math_fma_deferral_matches_selected_product"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_fast_math_fma_selection"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto lhs = dc.device.create_buffer<float>(2u);
        auto rhs = dc.device.create_buffer<float>(2u);
        auto other_lhs = dc.device.create_buffer<float>(2u);
        auto other_rhs = dc.device.create_buffer<float>(2u);
        auto output = dc.device.create_buffer<float>(3u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat a,
                             BufferFloat b,
                             BufferFloat c,
                             BufferFloat d,
                             BufferFloat out) noexcept {
            auto first_product = a.read(0u) * b.read(0u);
            auto second_product = c.read(0u) * d.read(0u);
            out.write(0u, first_product + second_product);

            auto reused_product = a.read(1u) * b.read(1u);
            auto other_product = c.read(1u) * d.read(1u);
            out.write(1u, reused_product + other_product);
            out.write(2u, reused_product);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = true});

        constexpr std::array lhs_values{2.0f, 3.0f};
        constexpr std::array rhs_values{5.0f, 7.0f};
        constexpr std::array other_lhs_values{11.0f, 13.0f};
        constexpr std::array other_rhs_values{17.0f, 19.0f};
        std::array<float, 3u> result{};
        stream << lhs.copy_from(luisa::span{lhs_values})
               << rhs.copy_from(luisa::span{rhs_values})
               << other_lhs.copy_from(luisa::span{other_lhs_values})
               << other_rhs.copy_from(luisa::span{other_rhs_values})
               << shader(lhs, rhs, other_lhs, other_rhs, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result == std::array{197.0f, 268.0f, 21.0f});

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "fast-math FMA selection regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_extended_instruction(disassembly, "Fma") == 2u)
                << "both product sums must emit a fused Fma";
            expect(count_spirv_opcode(disassembly, "FMul") == 3u)
                << "only the selected single-use product may be deferred";
            expect(count_spirv_opcode(disassembly, "FAdd") == 0u)
                << "both fast-math product sums must be fused";
        }
    };

    "vk_user_compute_integer_power_scalar_and_vector_semantics"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto scalar_base = dc.device.create_buffer<float>(5u);
        auto scalar_exponent = dc.device.create_buffer<int32_t>(5u);
        auto vector_base = dc.device.create_buffer<float2>(2u);
        auto vector_exponent = dc.device.create_buffer<int2>(1u);
        auto scalar_output = dc.device.create_buffer<float>(5u);
        auto vector_output = dc.device.create_buffer<float2>(2u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat scalar_bases,
                             BufferInt scalar_exponents,
                             BufferFloat2 vector_bases,
                             BufferInt2 vector_exponents,
                             BufferFloat scalar_out,
                             BufferFloat2 vector_out) noexcept {
            auto builder =
                luisa::compute::detail::FunctionBuilder::current();
            auto i = dispatch_x();
            auto base = scalar_bases.read(i);
            auto exponent = scalar_exponents.read(i);
            auto scalar_result = def<float>(builder->call(
                Type::of<float>(), CallOp::POW,
                {base.expression(), exponent.expression()}));
            scalar_out.write(i, scalar_result);
            $if (i == 0u) {
                auto broadcast_base = vector_bases.read(0u);
                auto broadcast_exponent = scalar_exponents.read(0u);
                auto lane_base = vector_bases.read(1u);
                auto lane_exponent = vector_exponents.read(0u);
                vector_out.write(0u, def<float2>(builder->call(
                                         Type::of<float2>(), CallOp::POW,
                                         {broadcast_base.expression(),
                                          broadcast_exponent.expression()})));
                vector_out.write(1u, def<float2>(builder->call(
                                         Type::of<float2>(), CallOp::POW,
                                         {lane_base.expression(),
                                          lane_exponent.expression()})));
            };
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array scalar_base_source{
            2.0f, -2.0f, -2.0f, 0.5f, -1.0f};
        constexpr std::array scalar_exponent_source{
            -3, -3, 3, 0, std::numeric_limits<int32_t>::min()};
        constexpr std::array vector_base_source{
            float2{2.0f, -2.0f}, float2{2.0f, 3.0f}};
        constexpr std::array vector_exponent_source{int2{-2, 3}};
        constexpr std::array scalar_expected{
            0.125f, -0.125f, -8.0f, 1.0f, 1.0f};
        constexpr std::array vector_expected{
            float2{0.125f, -0.125f}, float2{0.25f, 27.0f}};
        std::array<float, 5u> scalar_result{};
        std::array<float2, 2u> vector_result{};
        stream << scalar_base.copy_from(luisa::span{scalar_base_source})
               << scalar_exponent.copy_from(
                      luisa::span{scalar_exponent_source})
               << vector_base.copy_from(luisa::span{vector_base_source})
               << vector_exponent.copy_from(
                      luisa::span{vector_exponent_source})
               << shader(scalar_base, scalar_exponent,
                         vector_base, vector_exponent,
                         scalar_output, vector_output)
                      .dispatch(5u)
               << scalar_output.copy_to(luisa::span{scalar_result})
               << vector_output.copy_to(luisa::span{vector_result})
               << synchronize();
        expect(scalar_result == scalar_expected);
        for (auto i = 0u; i < vector_result.size(); i++) {
            expect_vector_equal(vector_result[i], vector_expected[i]);
        }
    };

    "vk_user_compute_componentwise_vector_static_casts"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto numeric_float = dc.device.create_buffer<float4>(1u);
        auto truth_float = dc.device.create_buffer<float4>(1u);
        auto numeric_int = dc.device.create_buffer<int4>(1u);
        auto signed_int = dc.device.create_buffer<int4>(1u);
        auto int_output = dc.device.create_buffer<int4>(1u);
        auto float_output = dc.device.create_buffer<float4>(1u);
        auto bool_output = dc.device.create_buffer<uint4>(1u);
        auto uint_output = dc.device.create_buffer<uint4>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat4 numeric_floats,
                             BufferFloat4 truth_floats,
                             BufferInt4 numeric_ints,
                             BufferInt4 signed_ints,
                             BufferInt4 int_out,
                             BufferFloat4 float_out,
                             BufferUInt4 bool_out,
                             BufferUInt4 uint_out) noexcept {
            int_out.write(0u, cast<int4>(numeric_floats.read(0u)));
            float_out.write(0u, cast<float4>(numeric_ints.read(0u)));
            bool_out.write(
                0u, select(make_uint4(0u), make_uint4(1u),
                           cast<bool4>(truth_floats.read(0u))));
            uint_out.write(0u, cast<uint4>(signed_ints.read(0u)));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        auto nan = std::numeric_limits<float>::quiet_NaN();
        std::array numeric_float_source{
            float4{-2.75f, -0.0f, 3.75f, 42.5f}};
        std::array truth_float_source{
            float4{nan, 0.0f, -0.0f, -2.0f}};
        std::array numeric_int_source{
            int4{-1024, -7, 0, 1048576}};
        std::array signed_int_source{
            int4{-1, 0, 1, std::numeric_limits<int32_t>::min()}};
        std::array<int4, 1u> int_result{};
        std::array<float4, 1u> float_result{};
        std::array<uint4, 1u> bool_result{};
        std::array<uint4, 1u> uint_result{};
        stream << numeric_float.copy_from(
                      luisa::span{numeric_float_source})
               << truth_float.copy_from(luisa::span{truth_float_source})
               << numeric_int.copy_from(luisa::span{numeric_int_source})
               << signed_int.copy_from(luisa::span{signed_int_source})
               << shader(numeric_float, truth_float,
                         numeric_int, signed_int,
                         int_output, float_output,
                         bool_output, uint_output)
                      .dispatch(1u)
               << int_output.copy_to(luisa::span{int_result})
               << float_output.copy_to(luisa::span{float_result})
               << bool_output.copy_to(luisa::span{bool_result})
               << uint_output.copy_to(luisa::span{uint_result})
               << synchronize();
        expect_vector_equal(int_result[0], int4{-2, 0, 3, 42});
        expect_vector_equal(
            float_result[0], float4{-1024.0f, -7.0f, 0.0f, 1048576.0f});
        expect_vector_equal(bool_result[0], uint4{1u, 0u, 0u, 1u});
        expect_vector_equal(
            uint_result[0], uint4{0xffffffffu, 0u, 1u, 0x80000000u});
    };

    "vk_user_compute_width_preserving_bitcasts_are_exact"_test = [&] {
        ScopedEnvironmentVariable use_default_scalarizer{
            "LUISA_XIR_ENABLE_SCALARIZER", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_width_preserving_bitcasts"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<uint4>(1u);
        auto float_output = dc.device.create_buffer<float4>(1u);
        auto wide_output = dc.device.create_buffer<ulong2>(1u);
        auto scalar_output = dc.device.create_buffer<luisa::ulong>(1u);
        auto pair_output = dc.device.create_buffer<uint2>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt4 in,
                             BufferFloat4 float_out,
                             BufferULong2 wide_out,
                             BufferULong scalar_out,
                             BufferUInt2 pair_out) noexcept {
            auto words = in.read(0u);
            float_out.write(0u, words.bitcast<float4>());
            wide_out.write(0u, words.bitcast<ulong2>());
            auto scalar = words.xy().bitcast<luisa::ulong>();
            scalar_out.write(0u, scalar);
            pair_out.write(0u, scalar.bitcast<uint2>());
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr uint4 source{
            0x7fc01234u, 0x80000000u,
            0x7f800000u, 0x01234567u};
        constexpr std::array source_data{source};
        std::array<float4, 1u> float_result{};
        std::array<ulong2, 1u> wide_result{};
        std::array<luisa::ulong, 1u> scalar_result{};
        std::array<uint2, 1u> pair_result{};
        stream << input.copy_from(luisa::span{source_data})
               << shader(input, float_output, wide_output,
                         scalar_output, pair_output)
                      .dispatch(1u)
               << float_output.copy_to(luisa::span{float_result})
               << wide_output.copy_to(luisa::span{wide_result})
               << scalar_output.copy_to(luisa::span{scalar_result})
               << pair_output.copy_to(luisa::span{pair_result})
               << synchronize();

        for (auto i = 0u; i < 4u; i++) {
            expect(std::bit_cast<uint32_t>(float_result[0][i]) == source[i])
                << "float bitcast changed lane " << i;
        }
        constexpr auto low =
            luisa::ulong{source.x} |
            (luisa::ulong{source.y} << 32u);
        constexpr auto high =
            luisa::ulong{source.z} |
            (luisa::ulong{source.w} << 32u);
        expect(wide_result[0].x == low && wide_result[0].y == high)
            << "uint4-to-ulong2 bitcast changed SPIR-V component packing";
        expect(scalar_result[0] == low)
            << "uint2-to-ulong bitcast changed SPIR-V component packing";
        expect(pair_result[0].x == source.x &&
               pair_result[0].y == source.y)
            << "ulong-to-uint2 bitcast did not invert the packed scalar";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "bitcast regression should emit exactly one native SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            // With the scalarizer disabled by default, the component-wise
            // uint4-to-float4 cast stays a single vector Bitcast; the three
            // scalar/vector shape-changing casts are also single
            // instructions. (With LUISA_XIR_ENABLE_SCALARIZER=1 the lane
            // cast scalarizes into four casts, i.e. seven Bitcasts total.)
            auto bitcast_count =
                count_spirv_opcode(disassembly, "Bitcast");
            expect(bitcast_count == 4u)
                << luisa::format(
                       "SPIR-V opt0 must preserve the vector lane cast "
                       "and three width-preserving shape casts; found {}",
                       bitcast_count);
        }
    };

    "vk_user_compute_scalarizer_option_and_environment_precedence"_test = [&] {
        auto run_case = [&](const char *environment,
                            bool option_enabled,
                            size_t expected_bitcasts,
                            luisa::string_view label) {
            ScopedEnvironmentVariable scalarizer_environment{
                "LUISA_XIR_ENABLE_SCALARIZER", environment};
            ScopedEnvironmentVariable disable_spirv_optimization{
                "LUISA_SPIRV_OPT_LEVEL", "0"};
            ScopedEnvironmentVariable clear_spirv_pass_override{
                "LUISA_SPIRV_OPT_PASSES", nullptr};
            ScopedTemporaryCurrentPath work_dir{luisa::format(
                "luisa_vk_spirv_scalarizer_{}", label)};
            ScopedSourceDump source_dump;

            auto dc = luisa::test::create_device(argc, argv);
            auto input = dc.device.create_buffer<uint4>(1u);
            auto output = dc.device.create_buffer<float4>(1u);
            auto stream = dc.device.create_stream();
            Kernel1D kernel = [](BufferUInt4 in,
                                 BufferFloat4 out) noexcept {
                out.write(
                    0u, in.read(0u).bitcast<float4>());
            };
            auto shader = dc.device.compile(
                kernel,
                ShaderOption{
                    .enable_cache = false,
                    .enable_fast_math = false,
                    .enable_scalarizer =
                        option_enabled});

            constexpr uint4 source{
                0x3f800000u, 0xc0000000u,
                0x7f800000u, 0x80000000u};
            constexpr std::array source_data{source};
            std::array<float4, 1u> result{};
            stream << input.copy_from(
                          luisa::span{source_data})
                   << shader(input, output).dispatch(1u)
                   << output.copy_to(luisa::span{result})
                   << synchronize();
            for (auto i = 0u; i < 4u; ++i) {
                expect(std::bit_cast<uint32_t>(
                           result[0][i]) == source[i])
                    << luisa::format(
                           "scalarizer configuration '{}' "
                           "changed lane {}",
                           label, i);
            }

            auto dumps = find_spirv_dumps();
            expect(dumps.size() == 1u)
                << luisa::format(
                       "scalarizer configuration '{}' "
                       "should emit one SPIR-V module",
                       label);
            if (dumps.size() == 1u) {
                auto disassembly =
                    read_text_file(dumps.front());
                auto bitcast_count =
                    count_spirv_opcode(
                        disassembly, "Bitcast");
                expect(bitcast_count ==
                       expected_bitcasts)
                    << luisa::format(
                           "scalarizer configuration '{}' "
                           "expected {} Bitcast(s), found {}",
                           label, expected_bitcasts,
                           bitcast_count);
            }
        };

        run_case(nullptr, false, 1u, "default");
        run_case(nullptr, true, 4u, "option");
        run_case("0", true, 1u, "environment_off");
        run_case("1", false, 4u, "environment_on");
    };

    "vk_user_compute_wide_integer_boolean_casts_are_exact"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<luisa::ulong>(2u);
        auto output = dc.device.create_buffer<luisa::ulong>(2u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferULong in,
                             BufferULong out) noexcept {
            auto i = dispatch_x();
            out.write(i, cast<luisa::ulong>(cast<bool>(in.read(i))));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{
            luisa::ulong{0u}, luisa::ulong{0xfedcba9876543210ull}};
        std::array<luisa::ulong, 2u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(2u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result == std::array{
                             luisa::ulong{0u}, luisa::ulong{1u}});
    };

    "vk_user_compute_vector_rounds_half_away_from_zero"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float4>(3u);
        auto output = device.create_buffer<float4>(3u);
        Kernel1D kernel = [](BufferFloat4 in, BufferFloat4 out) noexcept {
            auto i = dispatch_x();
            out.write(i, round(in.read(i)));
        };
        ShaderOption option{.enable_fast_math = false};
        auto shader = device.compile(kernel, option);
        std::array source{
            float4{-2.5f, -0.5f, 0.5f, 3.5f},
            float4{-0.0f, 0.0f, 1.25f, -1.25f},
            float4{
                std::nextafter(0.5f, 0.0f),
                std::nextafter(-0.5f, 0.0f),
                std::nextafter(0.5f, 1.0f),
                std::nextafter(-0.5f, -1.0f)}};
        std::array<float4, 3u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(3u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0][0] == -3.0f);
        expect(result[0][1] == -1.0f);
        expect(result[0][2] == 1.0f);
        expect(result[0][3] == 4.0f);
        expect(std::signbit(result[1][0]));
        expect(!std::signbit(result[1][1]));
        expect(result[1][2] == 1.0f);
        expect(result[1][3] == -1.0f);
        expect(result[2][0] == 0.0f &&
               !std::signbit(result[2][0]));
        expect(result[2][1] == 0.0f &&
               std::signbit(result[2][1]));
        expect(result[2][2] == 1.0f);
        expect(result[2][3] == -1.0f);
    };

    "vk_user_compute_float_to_bool_treats_nan_as_nonzero"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float>(4u);
        auto output = device.create_buffer<uint32_t>(4u);
        Kernel1D kernel = [](BufferFloat in, BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, ite(cast<bool>(in.read(i)), 1u, 0u));
        };
        ShaderOption option{.enable_fast_math = false};
        auto shader = device.compile(kernel, option);
        std::array source{
            std::numeric_limits<float>::quiet_NaN(),
            0.0f,
            -0.0f,
            -2.0f};
        std::array<uint32_t, 4u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 1u);
        expect(result[1] == 0u);
        expect(result[2] == 0u);
        expect(result[3] == 1u);
    };

    "vk_user_compute_signed_power_of_two_division_and_kernel_block_size"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<int>(8u);
        auto division_output = device.create_buffer<int4>(8u);
        auto block_size_output = device.create_buffer<uint3>(8u);

        Callable divide_by_powers_of_two = [](Int value) noexcept {
            return make_int4(value / 2, value / 4, value / 8, value / 16);
        };
        Kernel1D kernel = [&](BufferInt in,
                              BufferInt4 division_out,
                              BufferUInt3 block_size_out) noexcept {
            set_block_size(32u, 1u, 1u);
            auto i = dispatch_x();
            division_out.write(i, divide_by_powers_of_two(in.read(i)));
            block_size_out.write(i, block_size());
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false});

        constexpr std::array source{-17, -15, -9, -7, -3, -1, 1, 17};
        constexpr std::array expected{
            int4{-8, -4, -2, -1},
            int4{-7, -3, -1, 0},
            int4{-4, -2, -1, 0},
            int4{-3, -1, 0, 0},
            int4{-1, 0, 0, 0},
            int4{0, 0, 0, 0},
            int4{0, 0, 0, 0},
            int4{8, 4, 2, 1}};
        std::array<int4, 8u> division_result{};
        std::array<uint3, 8u> block_size_result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, division_output, block_size_output).dispatch(8u)
               << division_output.copy_to(luisa::span{division_result})
               << block_size_output.copy_to(luisa::span{block_size_result})
               << synchronize();

        for (auto i = 0u; i < source.size(); i++) {
            expect_vector_equal(division_result[i], expected[i]);
            expect_vector_equal(block_size_result[i], uint3{32u, 1u, 1u});
        }
    };

    "vk_user_compute_assume_is_an_explicit_semantic_no_op"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_assume_no_op"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        constexpr std::array source{3u, 11u, 97u, 251u};
        auto input = dc.device.create_buffer<uint32_t>(source.size());
        auto output = dc.device.create_buffer<uint32_t>(source.size());
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt in, BufferUInt out) noexcept {
            auto i = dispatch_x();
            auto value = in.read(i);
            assume(value < 1024u);
            out.write(i, value * 3u + 1u);
        };
        auto normalized_xir_path = std::filesystem::path{luisa::format(
            "kernel.{:016x}.norm.xir",
            kernel.function()->function().hash())};
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, source.size()> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(source.size())
               << output.copy_to(luisa::span{result})
               << synchronize();
        for (auto i = 0u; i < source.size(); ++i) {
            expect(result[i] == source[i] * 3u + 1u)
                << "ignoring a satisfied assumption changed result " << i;
        }

        expect(std::filesystem::exists(normalized_xir_path))
            << "assumption regression must retain the normalized XIR handoff";
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "assumption regression should emit one native SPIR-V module";
        if (std::filesystem::exists(normalized_xir_path) &&
            dumps.size() == 1u) {
            auto normalized_xir = read_text_file(normalized_xir_path);
            auto disassembly = read_text_file(dumps.front());
            expect(count_substring(normalized_xir, "assume ") == 1u)
                << "the disabled XIR optimizer must preserve the assumption "
                   "through the native codegen handoff";
            expect(disassembly.find("AssumeTrueKHR") == std::string::npos)
                << "semantic no-op lowering must not add an unsupported "
                   "SPV_KHR_expect_assume dependency";
        }
    };

    "vk_user_compute_release_device_assert_is_explicitly_disabled"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_release_assert"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        constexpr std::array source{5u, 17u, 101u, 509u};
        auto input = dc.device.create_buffer<uint32_t>(source.size());
        auto output = dc.device.create_buffer<uint32_t>(source.size());
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt in, BufferUInt out) noexcept {
            auto i = dispatch_x();
            auto value = in.read(i);
            device_assert(value < 1024u, "value must be in range");
            out.write(i, value * 5u + 3u);
        };
        auto normalized_xir_path = std::filesystem::path{luisa::format(
            "kernel.{:016x}.norm.xir",
            kernel.function()->function().hash())};
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false,
                                 .enable_debug_info = false});

        std::array<uint32_t, source.size()> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(source.size())
               << output.copy_to(luisa::span{result})
               << synchronize();
        for (auto i = 0u; i < source.size(); ++i) {
            expect(result[i] == source[i] * 5u + 3u)
                << "disabling a satisfied release assertion changed result "
                << i;
        }

        expect(std::filesystem::exists(normalized_xir_path))
            << "release-assert regression must retain the normalized XIR handoff";
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "release-assert regression should emit one native SPIR-V module";
        if (std::filesystem::exists(normalized_xir_path) &&
            dumps.size() == 1u) {
            auto normalized_xir = read_text_file(normalized_xir_path);
            auto disassembly = read_text_file(dumps.front());
            expect(count_substring(normalized_xir, "assert ") == 1u)
                << "the disabled XIR optimizer must preserve the assertion "
                   "through the option-aware codegen handoff";
            expect(disassembly.find("DebugPrintf") == std::string::npos)
                << "release assertion lowering must not claim a debug-reporting "
                   "side effect that it cannot provide";
        }
    };

    "vk_user_compute_kernel_id_matches_multi_dispatch_push_constant"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_dispatch_metadata"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto output = dc.device.create_buffer<uint2>(3u);
        auto stream = dc.device.create_stream();
        Callable read_dispatch_identity = []() noexcept {
            return make_uint2(kernel_id(), dispatch_size().x);
        };
        Kernel1D kernel = [&](BufferUInt2 out) noexcept {
            $if (all(dispatch_id() == make_uint3(0u))) {
                auto identity = read_dispatch_identity();
                out.write(identity.x, identity);
            };
        };
        auto normalized_xir_path = std::filesystem::path{luisa::format(
            "kernel.{:016x}.norm.xir",
            kernel.function()->function().hash())};
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false});

        constexpr std::array dispatches{
            uint3{1u, 1u, 1u},
            uint3{3u, 1u, 1u},
            uint3{7u, 1u, 1u}};
        std::array<uint2, dispatches.size()> result{};
        stream << shader(output).dispatch(luisa::span{dispatches})
               << output.copy_to(luisa::span{result})
               << synchronize();
        for (auto i = 0u; i < dispatches.size(); ++i) {
            expect_vector_equal(result[i], uint2{i, dispatches[i].x});
        }
        expect(std::filesystem::exists(normalized_xir_path))
            << "opt0 dispatch-metadata fixture should retain its normalized XIR";
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "opt0 dispatch-metadata fixture should emit one SPIR-V module";
        if (std::filesystem::exists(normalized_xir_path) &&
            dumps.size() == 1u) {
            auto normalized_xir = read_text_file(normalized_xir_path);
            auto disassembly = read_text_file(dumps.front());
            expect(count_substring(normalized_xir, "callable ") == 1u)
                << "the opt-disabled callable must survive to exercise the hidden metadata parameter";
            expect(count_spirv_opcode(disassembly, "FunctionCall") == 1u)
                << "dispatch metadata must cross one real SPIR-V callable boundary";
            expect(count_spirv_opcode(disassembly, "Phi") >= 1u)
                << "the straight-line source has no Phi, so at least one "
                   "backend-owned Phi must merge direct/indirect metadata";
            expect(count_spirv_opcode(disassembly, "Select") == 0u)
                << "direct/indirect metadata selection must use control flow, not eager OpSelect loads";
        }
    };

    "vk_user_compute_device_clock_uses_enabled_khr_feature"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_device_clock"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        if (dc.device.query("shader_device_clock") != "true") {
            expect(true)
                << "Vulkan device does not expose shaderDeviceClock; runtime "
                   "coverage is skipped while structural validation remains active";
            return;
        }
        constexpr auto invocation_count = 4u;
        auto begin_output =
            dc.device.create_buffer<luisa::ulong>(invocation_count);
        auto end_output =
            dc.device.create_buffer<luisa::ulong>(invocation_count);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferULong begin_out,
                             BufferULong end_out) noexcept {
            auto i = dispatch_x();
            auto begin = device_clock();
            begin_out.write(i, begin);
            auto end = device_clock();
            end_out.write(i, end);
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<luisa::ulong, invocation_count> begin_result{};
        std::array<luisa::ulong, invocation_count> end_result{};
        stream << shader(begin_output, end_output)
                      .dispatch(invocation_count)
               << begin_output.copy_to(luisa::span{begin_result})
               << end_output.copy_to(luisa::span{end_result})
               << synchronize();
        for (auto i = 0u; i < invocation_count; ++i) {
            expect(end_result[i] >= begin_result[i])
                << "device-scope shader clock regressed within invocation "
                << i;
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "device-clock regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "ReadClockKHR") == 2u)
                << "the two source clock reads must remain distinct at opt0";
            expect(count_substring(
                       disassembly,
                       "SPV_KHR_shader_clock") == 1u)
                << "device clock must declare SPV_KHR_shader_clock exactly once";
            expect(count_substring(
                       disassembly,
                       "Capability ShaderClockKHR") == 1u)
                << "device clock must declare ShaderClockKHR exactly once";
        }
    };

    "vk_user_compute_buffer_device_address_uses_exact_view_metadata"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_buffer_device_address"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        if (dc.device.query("buffer_device_address") != "true" ||
            dc.device.query("shader_int64") != "true") {
            expect(true)
                << "Vulkan device does not expose bufferDeviceAddress with "
                   "shaderInt64; "
                   "runtime coverage is skipped while structural validation "
                   "remains active";
            return;
        }
        constexpr auto element_count = 16u;
        constexpr auto view_offset = 5u;
        constexpr auto view_count = 7u;
        auto values = dc.device.create_buffer<uint32_t>(element_count);
        auto output = dc.device.create_buffer<uint64_t>(5u);
        auto heap = dc.device.create_bindless_array(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferUInt whole,
                             BufferUInt view,
                             BindlessVar bindless,
                             BufferVar<uint64_t> out) noexcept {
            out.write(0u, whole.device_address());
            out.write(1u, view.device_address());
            out.write(
                2u,
                bindless.buffer<uint32_t>(0u).device_address());
            out.write(
                3u,
                bindless.buffer<uint32_t>(0u, true).device_address());
            out.write(
                4u,
                bindless.byte_buffer(0u, true, true).device_address());
        };
        auto normalized_xir_path = std::filesystem::path{luisa::format(
            "kernel.{:016x}.norm.xir",
            kernel.function()->function().hash())};
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        heap.emplace_on_update(
            0u, values.view(view_offset, view_count));
        std::array<uint64_t, 5u> result{};
        stream << heap.update()
               << shader(values,
                         values.view(view_offset, view_count),
                         heap, output)
                      .dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(result[0] != 0u)
            << "an address-capable Vulkan buffer returned the reserved null address";
        expect(result[1] - result[0] ==
               static_cast<uint64_t>(
                   view_offset * sizeof(uint32_t)))
            << "direct buffer-view address did not include its logical byte offset";
        expect(result[2] == result[1])
            << "bindless and direct metadata disagreed on the same buffer view address";
        expect(result[3] == result[1])
            << "typed bindless metadata disagreed on the same buffer view address";
        expect(result[4] == result[1])
            << "typed uniform byte-buffer metadata disagreed on the same buffer view address";

        expect(std::filesystem::exists(normalized_xir_path))
            << "device-address regression must retain the normalized XIR handoff";
        if (std::filesystem::exists(normalized_xir_path)) {
            auto normalized_xir = read_text_file(normalized_xir_path);
            expect(normalized_xir.find("buffer_device_address") !=
                   std::string::npos)
                << "the opt-disabled handoff lost direct buffer-address queries";
            expect(normalized_xir.find("bindless_buffer_device_address") !=
                   std::string::npos)
                << "the opt-disabled handoff lost the bindless buffer-address query";
        }
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "device-address regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(disassembly.find("Capability PhysicalStorageBufferAddresses") ==
                   std::string::npos)
                << "returning an opaque address must not claim physical-pointer dereference";
            expect(disassembly.find("MemoryModel PhysicalStorageBuffer64") ==
                   std::string::npos)
                << "metadata-only address queries must retain Vulkan's logical addressing model";
        }
    };

    "vk_user_compute_nested_if_loop_exit_preserves_outer_merge"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto output = device.create_buffer<uint32_t>(5u);

        Kernel1D kernel = [](BufferUInt out) noexcept {
            auto lane = dispatch_x();
            UInt value = 1u;
            $if (lane < 4u) {
                UInt iteration = 0u;
                $while (iteration < 4u) {
                    iteration += 1u;
                    value += iteration;
                    $if ((lane < 3u) & (iteration == lane + 1u)) {
                        $break;
                    };
                    value += 10u;
                };
            };
            value = value * 2u + 1u;
            out.write(lane, value);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false});

        std::array<uint32_t, 5u> result{};
        stream << shader(output).dispatch(5u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{5u, 29u, 55u, 103u, 3u};
        expect(result == expected)
            << "the outer merge must execute exactly once after early and normal loop exits";
    };

    "vk_user_compute_sibling_one_sided_loop_exits_preserve_continuation"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto output = device.create_buffer<uint32_t>(4u);

        Kernel1D kernel = [](BufferUInt out) noexcept {
            auto lane = dispatch_x();
            UInt value = 0u;
            UInt iteration = 0u;
            $while (iteration < 1u) {
                iteration += 1u;
                $if ((lane & 1u) == 0u) {
                    $if (lane < 2u) {
                        value = 11u;
                        $break;
                    };
                    value = 12u;
                    $break;
                }
                $else {
                    $if (lane < 3u) {
                        value = 21u;
                        $break;
                    };
                    value = 22u;
                    $break;
                };
            };
            out.write(lane, value + 1000u);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, 4u> result{};
        stream << shader(output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{
            1011u, 1021u, 1012u, 1022u};
        expect(result == expected)
            << "sibling one-sided exits must preserve each path-local value "
               "and execute the loop continuation exactly once";
    };

    "vk_native_xir_spirv_recovers_cyclic_indexed_branch_before_emission"_test = [] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};

        // Build the exact raw-CFG category that caused the production volume
        // kernel to clone its complete loop body: the IndexedBranch itself has
        // a target that dominates it. A source-level switch/continue does not
        // necessarily retain this shape after destructuring, so this is an
        // explicit XIR-to-SPIR-V conformance test.
        Kernel1D ast_kernel = [](Int) noexcept {};
        auto ast_function = ast_kernel.function()->function();
        xir::Module module;
        auto *kernel = module.create_kernel();
        kernel->set_block_size(ast_function.block_size());
        auto *selector =
            kernel->create_value_argument(Type::of<int32_t>());
        auto *body = kernel->create_body_block();
        auto *header = kernel->create_basic_block();
        auto *payload = kernel->create_basic_block();
        auto *latch = kernel->create_basic_block();
        auto *case_exit = kernel->create_basic_block();
        auto *default_exit = kernel->create_basic_block();

        xir::XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.br(header);
        builder.set_insertion_point(header);
        builder.br(payload);
        builder.set_insertion_point(payload);
        builder.br(latch);
        builder.set_insertion_point(latch);
        auto *indexed = builder.indexed_branch(selector);
        indexed->add_case(0u, header);
        indexed->add_case(1u, case_exit);
        indexed->set_default_block(default_exit);
        builder.set_insertion_point(case_exit);
        builder.return_void();
        builder.set_insertion_point(default_exit);
        builder.return_void();

        expect(xir::xir_verify_module(&module).succeeded());
        auto restructure =
            xir::restructure_cfg_pass_run_on_function(kernel);
        expect(restructure.succeeded());
        expect(restructure.restructured_loop_count == 1u);
        expect(restructure.canonicalized_cfg_count != 0u);
        expect(xir::xir_verify_module(
                   &module,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());

        // compile_spirv_xir accepts the production legalization boundary,
        // not merely a structurally valid module. Preserve block identities
        // while clearing disconnected raw-CFG payloads exactly as the Vulkan
        // translation pipeline does before the direct handoff.
        auto post_restructure = luisa::compute::spirv::
            create_spirv_codegen_post_restructure_pipeline();
        [[maybe_unused]] auto cleanup_stats =
            post_restructure.run(&module);
        expect(xir::xir_verify_module(
                   &module,
                   {.require_no_unstructured_control_flow = true,
                    .require_unique_merge_blocks = true})
                   .succeeded());

        auto spirv = lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
            ast_function, &module,
            ShaderOption{.enable_cache = false,
                         .enable_fast_math = false});
        auto words = luisa::span<const uint32_t>{spirv.spv_bin};
        expect(count_spirv_binary_opcode(
                   words, spv::Op::OpLoopMerge) >= 1u)
            << "cyclic indexed control flow must reach SPIR-V as a loop";
        expect(count_spirv_binary_opcode(
                   words, spv::Op::OpSwitch) == 0u)
            << "the cyclic IndexedBranch must be lowered to conditional "
               "control flow before native SPIR-V emission";
    };

    "vk_user_compute_native_switch_nested_in_loop"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_native_switch"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto selectors = device.create_buffer<int32_t>(8u);
        auto output = device.create_buffer<int32_t>(8u);

        Kernel1D kernel = [](BufferInt selector_buffer,
                             BufferInt out) noexcept {
            auto lane = dispatch_x();
            auto selector = selector_buffer.read(lane);
            UInt iteration = 0u;
            Int value = 0;
            $while (iteration < 5u) {
                iteration += 1u;
                $switch (selector) {
                    // These semantically equivalent empty cases may be kept
                    // separate or merged by CFG simplification.
                    $case (-2) {};
                    $case (3) {};
                    $case (0) { value += 11; };
                    $default { value += 17; };
                };
                value += 5;
                $if ((iteration == 2u) & ((lane & 1u) == 0u)) {
                    $continue;
                };
                value += cast<int>(iteration * 10u);
                $if ((iteration == 4u) & (lane % 3u == 0u)) {
                    $break;
                };
            };
            out.write(lane, value);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array selector_source{-2, 0, 7, 3, 0, 7, 3, -2};
        std::array<int32_t, 8u> expected{};
        for (auto lane = 0u; lane < selector_source.size(); lane++) {
            auto iteration = 0u;
            auto value = int32_t{0};
            while (iteration < 5u) {
                iteration++;
                switch (selector_source[lane]) {
                    case -2:
                    case 3: break;
                    case 0: value += 11; break;
                    default: value += 17; break;
                }
                value += 5;
                if (iteration == 2u && (lane & 1u) == 0u) { continue; }
                value += static_cast<int32_t>(iteration * 10u);
                if (iteration == 4u && lane % 3u == 0u) { break; }
            }
            expected[lane] = value;
        }
        std::array<int32_t, 8u> result{};
        stream << selectors.copy_from(luisa::span{selector_source})
               << shader(selectors, output).dispatch(8u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result == expected)
            << "negative/default cases and enclosing loop break/continue paths must preserve exact switch semantics";
        expect(result[0] == result[6])
            << "semantically equivalent negative and positive cases must agree under identical loop control";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "native switch regression should emit exactly one SPIR-V dump";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "Switch") >= 1u)
                << "native XIR switch lowering must reach SPIR-V as OpSwitch";
        }
    };

    "vk_user_compute_native_u64_switch_literals"_test = [&] {
        ScopedEnvironmentVariable enable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", nullptr};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        // Keep the optional file dump disabled so the assertions below inspect
        // the raw binary independently. The codegen's mandatory in-memory
        // disassembly still runs and therefore also verifies that glslang
        // consumes both words of every 64-bit OpSwitch literal.
        ScopedEnvironmentVariable disable_source_dump{
            "LUISA_DUMP_SOURCE", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_native_u64_switch"};

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto selectors = device.create_buffer<luisa::ulong>(4u);
        auto output = device.create_buffer<uint32_t>(4u);

        Kernel1D kernel = [](BufferULong selector_buffer,
                             BufferUInt out) noexcept {
            auto lane = dispatch_x();
            auto selector = selector_buffer.read(lane);
            UInt value = 33u;
            $switch (selector) {
                $case (luisa::ulong{0x00000000ffffffffull}) { value = 11u; };
                $case (luisa::ulong{0xffffffffffffffffull}) { value = 22u; };
                $default {};
            };
            out.write(lane, value);
        };
        const ShaderOption compile_options{
            .enable_cache = false,
            .enable_fast_math = false};
        constexpr auto required_features =
            lc::spirv::target_feature::shader_int64;
        constexpr auto target_features =
            lc::spirv::SpirvTargetFeatures::from_enabled_mask(
                required_features);
        auto compiled = lc::spirv::SpirvCodegenEntry::compile_spirv(
            kernel.function()->function(), compile_options,
            target_features);
        expect(eq(compiled.required_target_features,
                  required_features));
        auto switches = inspect_spirv_u64_switches(compiled.spv_bin);
        expect(eq(switches.size(), 1u))
            << "the native opt0 module must contain exactly one uint64 "
               "OpSwitch";
        if (switches.size() == 1u) {
            auto &&shape = switches.front();
            expect(eq(shape.cases.size(), 2u))
                << "the uint64 OpSwitch must retain exactly two cases";
            expect(shape.targets_are_labels)
                << "the default and both case targets must name OpLabel "
                   "instructions";
            const auto low_word_collision =
                std::array<uint32_t, 2u>{0xffffffffu, 0x00000000u};
            const auto all_ones =
                std::array<uint32_t, 2u>{0xffffffffu, 0xffffffffu};
            auto first = std::find_if(
                shape.cases.begin(), shape.cases.end(),
                [&](auto &&candidate) noexcept {
                    return candidate.literal_words == low_word_collision;
                });
            auto second = std::find_if(
                shape.cases.begin(), shape.cases.end(),
                [&](auto &&candidate) noexcept {
                    return candidate.literal_words == all_ones;
                });
            expect(first != shape.cases.end())
                << "0x00000000ffffffff must be encoded as low=ffffffff, "
                   "high=00000000";
            expect(second != shape.cases.end())
                << "0xffffffffffffffff must be encoded as low=ffffffff, "
                   "high=ffffffff";
            if (first != shape.cases.end() &&
                second != shape.cases.end()) {
                expect(first->target != second->target)
                    << "the two complete 64-bit literals must retain "
                       "distinct target pairings";
                expect(first->target != shape.default_target);
                expect(second->target != shape.default_target);
            }
        }

        auto shader = device.compile(kernel, compile_options);

        constexpr std::array<luisa::ulong, 4u> selector_source{
            luisa::ulong{0x00000000ffffffffull},
            luisa::ulong{0xffffffffffffffffull},
            luisa::ulong{0xfffffffffffffffeull},
            luisa::ulong{7u}};
        std::array<uint32_t, 4u> result{};
        stream << selectors.copy_from(luisa::span{selector_source})
               << shader(selectors, output).dispatch(4u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{11u, 22u, 33u, 33u};
        expect(result == expected)
            << "64-bit switch literals with identical low words must remain distinct";
    };

    "vk_user_compute_typed_floating_constants"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        run_typed_float_constant_case<half, half2>(device, 5e-2);
        run_typed_float_constant_case<float, float2>(device, 2e-4);
        run_typed_float_constant_case<double, double2, false>(device, 1e-10);
    };

    "vk_user_compute_integer_vectors_and_wide_constant_indices"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto integer_input = device.create_buffer<uint2>(2u);
        auto clz_output = device.create_buffer<uint2>(2u);
        auto ctz_output = device.create_buffer<uint2>(2u);
        auto popcount_output = device.create_buffer<uint2>(2u);
        auto reverse_output = device.create_buffer<uint2>(2u);
        auto vector_input = device.create_buffer<float4>(1u);
        auto aggregate_output = device.create_buffer<float2>(1u);
        auto extract_output = device.create_buffer<float>(1u);
        auto shuffle_output = device.create_buffer<float2>(1u);
        auto matrix_input = device.create_buffer<float4x4>(1u);
        auto dynamic_index_input = device.create_buffer<short>(1u);
        auto dynamic_extract_output = device.create_buffer<float4>(1u);
        Kernel1D kernel = [](BufferUInt2 integers,
                             BufferUInt2 clz_out,
                             BufferUInt2 ctz_out,
                             BufferUInt2 popcount_out,
                             BufferUInt2 reverse_out,
                             BufferFloat4 vectors,
                             BufferFloat2 aggregate_out,
                             BufferFloat extract_out,
                             BufferFloat2 shuffle_out,
                             BufferFloat4x4 matrices,
                             BufferShort dynamic_indices,
                             BufferFloat4 dynamic_extract_out) noexcept {
            auto i = dispatch_x();
            $if (i < 2u) {
                auto value = integers.read(i);
                clz_out.write(i, clz(value));
                ctz_out.write(i, ctz(value));
                popcount_out.write(i, popcount(value));
                reverse_out.write(i, reverse(value));
            };
            $if (i == 0u) {
                auto value = vectors.read(0u);
                aggregate_out.write(
                    0u, make_float2(
                            value[static_cast<int16_t>(1)],
                            value[static_cast<uint64_t>(3)]));
                extract_out.write(0u, value[static_cast<int8_t>(2)]);
                shuffle_out.write(0u, value.wy());
                dynamic_extract_out.write(
                    0u, matrices.read(0u)[dynamic_indices.read(0u)]);
            };
        };
        auto shader = device.compile(kernel);
        std::array integer_source{
            uint2{0u, 1u},
            uint2{0x80000000u, 0x10u}};
        std::array vector_source{float4{10.0f, 20.0f, 30.0f, 40.0f}};
        std::array matrix_source{make_float4x4(
            float4{1.0f, 2.0f, 3.0f, 4.0f},
            float4{5.0f, 6.0f, 7.0f, 8.0f},
            float4{9.0f, 10.0f, 11.0f, 12.0f},
            float4{13.0f, 14.0f, 15.0f, 16.0f})};
        std::array dynamic_index_source{static_cast<short>(2)};
        std::array<uint2, 2u> clz_result{};
        std::array<uint2, 2u> ctz_result{};
        std::array<uint2, 2u> popcount_result{};
        std::array<uint2, 2u> reverse_result{};
        std::array<float2, 1u> aggregate_result{};
        std::array<float, 1u> extract_result{};
        std::array<float2, 1u> shuffle_result{};
        std::array<float4, 1u> dynamic_extract_result{};
        stream << integer_input.copy_from(luisa::span{integer_source})
               << vector_input.copy_from(luisa::span{vector_source})
               << matrix_input.copy_from(luisa::span{matrix_source})
               << dynamic_index_input.copy_from(luisa::span{dynamic_index_source})
               << shader(integer_input, clz_output, ctz_output,
                         popcount_output, reverse_output, vector_input,
                         aggregate_output, extract_output, shuffle_output,
                         matrix_input, dynamic_index_input,
                         dynamic_extract_output)
                      .dispatch(2u)
               << clz_output.copy_to(luisa::span{clz_result})
               << ctz_output.copy_to(luisa::span{ctz_result})
               << popcount_output.copy_to(luisa::span{popcount_result})
               << reverse_output.copy_to(luisa::span{reverse_result})
               << aggregate_output.copy_to(luisa::span{aggregate_result})
               << extract_output.copy_to(luisa::span{extract_result})
               << shuffle_output.copy_to(luisa::span{shuffle_result})
               << dynamic_extract_output.copy_to(luisa::span{dynamic_extract_result})
               << synchronize();
        expect_vector_equal(clz_result[0], uint2{32u, 31u});
        expect_vector_equal(clz_result[1], uint2{0u, 27u});
        expect_vector_equal(ctz_result[0], uint2{32u, 0u});
        expect_vector_equal(ctz_result[1], uint2{31u, 4u});
        expect_vector_equal(popcount_result[0], uint2{0u, 1u});
        expect_vector_equal(popcount_result[1], uint2{1u, 1u});
        expect_vector_equal(
            reverse_result[0], uint2{0u, 0x80000000u});
        expect_vector_equal(
            reverse_result[1], uint2{1u, 0x08000000u});
        expect_vector_equal(aggregate_result[0], float2{20.0f, 40.0f});
        expect(extract_result[0] == 30.0f);
        expect_vector_equal(shuffle_result[0], float2{40.0f, 20.0f});
        expect_vector_equal(dynamic_extract_result[0], float4{9.0f, 10.0f, 11.0f, 12.0f});
    };

    "vk_user_compute_dynamic_insert_preserves_source"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float4>(1u);
        auto dynamic_index = device.create_buffer<short>(1u);
        auto original_output = device.create_buffer<float4>(1u);
        auto inserted_output = device.create_buffer<float4>(1u);

        Callable make_inserted = [](Float4 value, Short index) noexcept {
            value[index] = 99.0f;
            return value;
        };
        Kernel1D kernel = [&](BufferFloat4 source,
                              BufferShort indices,
                              BufferFloat4 original_out,
                              BufferFloat4 inserted_out) noexcept {
            auto original = source.read(0u);
            auto inserted = make_inserted(original, indices.read(0u));
            original_out.write(0u, original);
            inserted_out.write(0u, inserted);
        };
        auto shader = device.compile(kernel);

        std::array source{float4{1.0f, 2.0f, 3.0f, 4.0f}};
        std::array index{static_cast<short>(2)};
        std::array<float4, 1u> original_result{};
        std::array<float4, 1u> inserted_result{};
        stream << input.copy_from(luisa::span{source})
               << dynamic_index.copy_from(luisa::span{index})
               << shader(input, dynamic_index, original_output, inserted_output).dispatch(1u)
               << original_output.copy_to(luisa::span{original_result})
               << inserted_output.copy_to(luisa::span{inserted_result})
               << synchronize();
        expect_vector_equal(original_result[0], float4{1.0f, 2.0f, 3.0f, 4.0f});
        expect_vector_equal(inserted_result[0], float4{1.0f, 2.0f, 99.0f, 4.0f});
    };

    "vk_user_compute_callable_aggregate_value_abi_is_exact"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_callable_aggregate_abi"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto input = device.create_buffer<float4>(1u);
        auto output = device.create_buffer<float4>(1u);

        Callable transform = [](Var<SpirvCallableAggregate> value) noexcept {
            Var<SpirvCallableAggregate> result;
            result.pair = make_float2(
                value.pair.y + value.weight,
                value.pair.x - value.weight);
            result.tag = value.tag * 3u + 1u;
            result.weight = value.pair.x * 2.0f + value.pair.y;
            return result;
        };
        Kernel1D kernel = [&](BufferFloat4 source,
                              BufferFloat4 destination) noexcept {
            auto packed = source.read(0u);
            Var<SpirvCallableAggregate> value;
            value.pair = packed.xy();
            value.tag = cast<uint32_t>(packed.z);
            value.weight = packed.w;
            auto result = transform(value);
            destination.write(
                0u, make_float4(
                        result.pair.x, result.pair.y,
                        cast<float>(result.tag), result.weight));
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        constexpr std::array source{float4{2.0f, 5.0f, 7.0f, 3.0f}};
        std::array<float4, 1u> result{};
        stream << input.copy_from(luisa::span{source})
               << shader(input, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect_vector_equal(result[0], float4{8.0f, -1.0f, 22.0f, 9.0f});

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "the aggregate callable fixture should emit one SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(disassembly, "FunctionCall") == 1u)
                << "the aggregate ABI must be exercised by a real OpFunctionCall";
        }
    };

    "vk_user_compute_subgroup_vector_and_matrix_shapes_are_exact"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_subgroup_composite_shapes"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto warp_size = device.compute_warp_size();
        expect(warp_size > 0u);
        if (warp_size == 0u) { return; }

        auto stream = device.create_stream();
        auto arithmetic_output = device.create_buffer<uint4>(warp_size);
        auto equality_output = device.create_buffer<uint2>(warp_size);
        auto matrix_output = device.create_buffer<float4>(warp_size);
        Kernel1D kernel = [=](BufferUInt4 arithmetic_out,
                              BufferUInt2 equality_out,
                              BufferFloat4 matrix_out) noexcept {
            set_block_size(warp_size, 1u, 1u);
            set_warp_size(warp_size);
            auto lane = warp_lane_id();
            auto lane_value = lane + 1u;
            auto vector_value = make_uint2(lane_value, lane_value * 3u);
            auto sum = warp_active_sum(vector_value);
            auto prefix = warp_prefix_sum(vector_value);
            arithmetic_out.write(
                lane, make_uint4(sum.x, sum.y, prefix.x, prefix.y));

            auto all_equal = warp_active_all_equal(make_uint2(17u, lane));
            equality_out.write(
                lane, select(make_uint2(0u), make_uint2(1u), all_equal));

            auto lane_float = cast<float>(lane);
            auto matrix = make_float2x2(
                make_float2(lane_float + 1.0f, lane_float + 2.0f),
                make_float2(lane_float + 3.0f, lane_float + 4.0f));
            auto first_matrix = warp_read_lane(matrix, 0u);
            matrix_out.write(
                lane, make_float4(
                          first_matrix[0u].x, first_matrix[0u].y,
                          first_matrix[1u].x, first_matrix[1u].y));
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        luisa::vector<uint4> arithmetic_result(warp_size);
        luisa::vector<uint2> equality_result(warp_size);
        luisa::vector<float4> matrix_result(warp_size);
        stream << shader(arithmetic_output, equality_output, matrix_output)
                      .dispatch(warp_size)
               << arithmetic_output.copy_to(luisa::span{arithmetic_result})
               << equality_output.copy_to(luisa::span{equality_result})
               << matrix_output.copy_to(luisa::span{matrix_result})
               << synchronize();

        auto sum = warp_size * (warp_size + 1u) / 2u;
        for (auto lane = 0u; lane < warp_size; ++lane) {
            auto prefix = lane * (lane + 1u) / 2u;
            expect_vector_equal(
                arithmetic_result[lane],
                uint4{sum, sum * 3u, prefix, prefix * 3u});
            expect_vector_equal(
                equality_result[lane],
                uint2{1u, warp_size == 1u ? 1u : 0u});
            expect_vector_equal(
                matrix_result[lane], float4{1.0f, 2.0f, 3.0f, 4.0f});
        }

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "the subgroup composite fixture should emit one SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(
                       disassembly, "GroupNonUniformIAdd") == 2u)
                << "vector reduce and prefix sum should each stay one subgroup op";
            expect(count_spirv_opcode(
                       disassembly, "GroupNonUniformAllEqual") == 2u)
                << "vector all-equal must scalarize to two subgroup votes";
            expect(count_spirv_opcode(
                       disassembly, "GroupNonUniformShuffle") == 4u)
                << "a 2x2 matrix shuffle must scalarize to four subgroup shuffles";
        }
    };

    "vk_user_compute_float_edge_semantics_and_scalar_broadcasts"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        auto scalar_lhs = device.create_buffer<float>(4u);
        auto scalar_rhs = device.create_buffer<float>(4u);
        auto scalar_copysign_output = device.create_buffer<float>(4u);
        auto scalar_not_equal_output = device.create_buffer<uint32_t>(4u);
        auto vector_lhs = device.create_buffer<float4>(1u);
        auto vector_rhs = device.create_buffer<float4>(1u);
        auto vector_copysign_output = device.create_buffer<float4>(1u);
        auto vector_not_equal_output = device.create_buffer<uint4>(1u);
        Kernel1D edge_kernel = [](BufferFloat lhs,
                                  BufferFloat rhs,
                                  BufferFloat copysign_out,
                                  BufferUInt not_equal_out,
                                  BufferFloat4 vector_lhs_buffer,
                                  BufferFloat4 vector_rhs_buffer,
                                  BufferFloat4 vector_copysign_out,
                                  BufferUInt4 vector_not_equal_out) noexcept {
            auto i = dispatch_x();
            auto a = lhs.read(i);
            auto b = rhs.read(i);
            copysign_out.write(i, copysign(a, b));
            not_equal_out.write(i, ite(a != b, 1u, 0u));
            $if (i == 0u) {
                auto va = vector_lhs_buffer.read(0u);
                auto vb = vector_rhs_buffer.read(0u);
                vector_copysign_out.write(0u, copysign(va, vb));
                vector_not_equal_out.write(
                    0u, select(make_uint4(0u), make_uint4(1u), va != vb));
            };
        };
        ShaderOption option{.enable_fast_math = false};
        auto edge_shader = device.compile(edge_kernel, option);
        auto nan = std::numeric_limits<float>::quiet_NaN();
        std::array scalar_lhs_source{2.0f, -3.0f, 0.0f, -0.0f};
        std::array scalar_rhs_source{0.0f, -0.0f, -0.0f, 0.0f};
        std::array vector_lhs_source{float4{nan, 0.0f, 1.0f, nan}};
        std::array vector_rhs_source{float4{nan, -0.0f, 2.0f, 4.0f}};
        std::array<float, 4u> scalar_copysign_result{};
        std::array<uint32_t, 4u> scalar_not_equal_result{};
        std::array<float4, 1u> vector_copysign_result{};
        std::array<uint4, 1u> vector_not_equal_result{};
        stream << scalar_lhs.copy_from(luisa::span{scalar_lhs_source})
               << scalar_rhs.copy_from(luisa::span{scalar_rhs_source})
               << vector_lhs.copy_from(luisa::span{vector_lhs_source})
               << vector_rhs.copy_from(luisa::span{vector_rhs_source})
               << edge_shader(
                      scalar_lhs, scalar_rhs,
                      scalar_copysign_output, scalar_not_equal_output,
                      vector_lhs, vector_rhs,
                      vector_copysign_output, vector_not_equal_output)
                      .dispatch(4u)
               << scalar_copysign_output.copy_to(luisa::span{scalar_copysign_result})
               << scalar_not_equal_output.copy_to(luisa::span{scalar_not_equal_result})
               << vector_copysign_output.copy_to(luisa::span{vector_copysign_result})
               << vector_not_equal_output.copy_to(luisa::span{vector_not_equal_result})
               << synchronize();
        for (auto i = 0u; i < scalar_lhs_source.size(); i++) {
            auto expected = std::copysign(scalar_lhs_source[i], scalar_rhs_source[i]);
            expect(std::bit_cast<uint32_t>(scalar_copysign_result[i]) ==
                   std::bit_cast<uint32_t>(expected));
            expect(scalar_not_equal_result[i] ==
                   static_cast<uint32_t>(scalar_lhs_source[i] != scalar_rhs_source[i]));
        }
        for (auto i = 0u; i < 4u; i++) {
            auto expected = std::copysign(vector_lhs_source[0][i], vector_rhs_source[0][i]);
            expect(std::bit_cast<uint32_t>(vector_copysign_result[0][i]) ==
                   std::bit_cast<uint32_t>(expected));
            expect(vector_not_equal_result[0][i] ==
                   static_cast<uint32_t>(vector_lhs_source[0][i] != vector_rhs_source[0][i]));
        }

        auto lower = device.create_buffer<float2>(1u);
        auto upper = device.create_buffer<float2>(1u);
        auto edge0 = device.create_buffer<float>(1u);
        auto edge1 = device.create_buffer<float>(1u);
        auto lerp_output = device.create_buffer<float2>(1u);
        auto step_output = device.create_buffer<float2>(1u);
        auto smoothstep_output = device.create_buffer<float2>(1u);
        Kernel1D broadcast_kernel = [](BufferFloat2 lower_buffer,
                                       BufferFloat2 upper_buffer,
                                       BufferFloat edge0_buffer,
                                       BufferFloat edge1_buffer,
                                       BufferFloat2 lerp_out,
                                       BufferFloat2 step_out,
                                       BufferFloat2 smoothstep_out) noexcept {
            auto lower_value = lower_buffer.read(0u);
            auto upper_value = upper_buffer.read(0u);
            auto edge0_value = edge0_buffer.read(0u);
            auto edge1_value = edge1_buffer.read(0u);
            auto builder = luisa::compute::detail::FunctionBuilder::current();
            auto lerp_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::LERP,
                {lower_value.expression(), upper_value.expression(), edge0_value.expression()}));
            auto step_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::STEP,
                {edge0_value.expression(), lower_value.expression()}));
            auto smoothstep_value = def<float2>(builder->call(
                Type::of<float2>(), CallOp::SMOOTHSTEP,
                {edge0_value.expression(), edge1_value.expression(), lower_value.expression()}));
            lerp_out.write(0u, lerp_value);
            step_out.write(0u, step_value);
            smoothstep_out.write(0u, smoothstep_value);
        };
        auto broadcast_shader = device.compile(broadcast_kernel, option);
        std::array lower_source{float2{0.1f, 0.5f}};
        std::array upper_source{float2{2.1f, 2.5f}};
        std::array edge0_source{0.25f};
        std::array edge1_source{0.75f};
        std::array<float2, 1u> lerp_result{};
        std::array<float2, 1u> step_result{};
        std::array<float2, 1u> smoothstep_result{};
        stream << lower.copy_from(luisa::span{lower_source})
               << upper.copy_from(luisa::span{upper_source})
               << edge0.copy_from(luisa::span{edge0_source})
               << edge1.copy_from(luisa::span{edge1_source})
               << broadcast_shader(lower, upper, edge0, edge1,
                                   lerp_output, step_output, smoothstep_output)
                      .dispatch(1u)
               << lerp_output.copy_to(luisa::span{lerp_result})
               << step_output.copy_to(luisa::span{step_result})
               << smoothstep_output.copy_to(luisa::span{smoothstep_result})
               << synchronize();
        expect(std::abs(lerp_result[0][0] - 0.6f) < 1e-5f);
        expect(std::abs(lerp_result[0][1] - 1.0f) < 1e-5f);
        expect_vector_equal(step_result[0], float2{0.0f, 1.0f});
        expect(std::abs(smoothstep_result[0][0] - 0.0f) < 1e-5f);
        expect(std::abs(smoothstep_result[0][1] - 0.5f) < 1e-5f);
    };

    "vk_user_compute_float_minimum_family_prefers_numbers_over_nan"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_number_minmax"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<float>(4u);
        auto output = dc.device.create_buffer<float>(10u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat in, BufferFloat out) noexcept {
            auto nan = in.read(0u);
            auto positive = in.read(1u);
            auto negative = in.read(2u);
            auto upper = in.read(3u);
            out.write(0u, min(nan, positive));
            out.write(1u, min(positive, nan));
            out.write(2u, max(nan, positive));
            out.write(3u, max(positive, nan));
            out.write(4u, clamp(nan, negative, upper));
            out.write(5u, clamp(positive, nan, 8.0f));
            out.write(6u, clamp(positive, -8.0f, nan));
            out.write(7u, saturate(nan));
            auto values = make_float4(nan, positive, negative, upper);
            out.write(8u, reduce_min(values));
            out.write(9u, reduce_max(values));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        auto nan = std::numeric_limits<float>::quiet_NaN();
        std::array input_values{nan, 4.0f, -3.0f, 2.0f};
        std::array<float, 10u> result{};
        stream << input.copy_from(luisa::span{input_values})
               << shader(input, output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        constexpr std::array expected{
            4.0f, 4.0f, 4.0f, 4.0f, -3.0f,
            4.0f, 4.0f, 0.0f, -3.0f, 4.0f};
        expect(result == expected)
            << "floating min/max/clamp/saturate/reductions must match minnum/maxnum when exactly one operand is NaN";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "number-minimum regression should emit one native SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_extended_instruction(disassembly, "NMin") == 5u);
            expect(count_spirv_extended_instruction(disassembly, "NMax") == 5u);
            expect(count_spirv_extended_instruction(disassembly, "NClamp") == 4u);
            expect(count_spirv_extended_instruction(disassembly, "FMin") == 0u);
            expect(count_spirv_extended_instruction(disassembly, "FMax") == 0u);
            expect(count_spirv_extended_instruction(disassembly, "FClamp") == 0u);
        }
    };

    "vk_user_compute_normalize_length_lower_to_native_spirv"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_native_normalize_length"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto input = dc.device.create_buffer<float4>(2u);
        auto normalized = dc.device.create_buffer<float4>(1u);
        auto lengths = dc.device.create_buffer<float>(1u);
        auto stream = dc.device.create_stream();
        Kernel1D kernel = [](BufferFloat4 in,
                             BufferFloat4 out_n,
                             BufferFloat out_l) noexcept {
            auto v = in.read(0u);
            auto w = in.read(1u);
            out_n.write(0u, normalize(v));
            out_l.write(0u, length(w));
        };
        auto shader = dc.device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array input_values{float4{1.0f, 2.0f, 3.0f, 4.0f},
                                float4{3.0f, 4.0f, 0.0f, 0.0f}};
        std::array<float4, 1u> normalized_result{};
        std::array<float, 1u> length_result{};
        stream << input.copy_from(luisa::span{input_values})
               << shader(input, normalized, lengths).dispatch(1u)
               << normalized.copy_to(luisa::span{normalized_result})
               << lengths.copy_to(luisa::span{length_result})
               << synchronize();
        auto expected_len = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
        auto n = normalized_result[0];
        expect(std::abs(n.x - 1.0f / expected_len) < 1e-5f);
        expect(std::abs(n.y - 2.0f / expected_len) < 1e-5f);
        expect(std::abs(n.z - 3.0f / expected_len) < 1e-5f);
        expect(std::abs(n.w - 4.0f / expected_len) < 1e-5f);
        expect(std::abs(length_result[0] - 5.0f) < 1e-5f);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "native normalize/length lowering should emit one SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_extended_instruction(disassembly, "Normalize") == 0u)
                << "vector normalize must lower to native SPIR-V, not GLSL.std.450";
            expect(count_spirv_extended_instruction(disassembly, "Length") == 0u)
                << "vector length must lower to native SPIR-V, not GLSL.std.450";
            expect(count_spirv_opcode(disassembly, "Dot") == 2u)
                << "normalize and length must each emit one OpDot";
            expect(count_spirv_opcode(disassembly, "VectorTimesScalar") == 1u)
                << "normalize must scale the vector by the reciprocal length";
            expect(count_spirv_opcode(disassembly, "FDiv") == 1u)
                << "normalize must emit one scalar reciprocal OpFDiv";
        }
    };

    "vk_user_compute_direct_surface_trace_uses_valid_primitive_culling"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_ray_primitive_culling"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedSourceDump source_dump;

        constexpr std::string_view dump_name =
            "spv_code_vk_direct_surface_trace.spvasm";
        // Store the farther primitive first so the fixture contains more than
        // one valid candidate and verifies the final closest committed hit,
        // rather than merely accepting the first hit recorded by traversal.
        const std::array vertices{
            float3{-0.5f, -0.5f, 0.0f},
            float3{0.5f, -0.5f, 0.0f},
            float3{0.0f, 0.5f, 0.0f},
            float3{-0.5f, -0.5f, 0.5f},
            float3{0.5f, -0.5f, 0.5f},
            float3{0.0f, 0.5f, 0.5f}};
        const std::array triangles{
            Triangle{0u, 1u, 2u},
            Triangle{3u, 4u, 5u}};
        auto vertex_buffer = device.create_buffer<float3>(vertices.size());
        auto triangle_buffer =
            device.create_buffer<Triangle>(triangles.size());
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
        auto accel = device.create_accel();
        accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, true);
        auto output = device.create_buffer<uint32_t>(6u);

        Kernel1D kernel = [](AccelVar accel_var,
                             BufferUInt result) noexcept {
            auto lane = dispatch_x();
            auto origin_x = cast<float>(lane) * 2.0f;
            auto ray = make_ray(
                make_float3(origin_x, 0.0f, 1.0f),
                make_float3(0.0f, 0.0f, -1.0f));
            auto closest = accel_var.intersect(ray, {});
            auto any = accel_var.intersect_any(ray, {});
            result.write(lane * 3u, closest->inst);
            result.write(lane * 3u + 1u, closest->prim);
            result.write(lane * 3u + 2u, ite(any, 1u, 0u));
        };
        auto shader = device.compile(
            kernel,
            ShaderOption{.enable_cache = false,
                         .name = "vk_direct_surface_trace"});

        expect(dump_exists(dump_name))
            << "direct surface tracing should emit a named native SPIR-V dump";
        if (dump_exists(dump_name)) {
            auto disassembly = read_text_file(
                std::filesystem::path{dump_name});
            expect(spirv_opcode_has_operand(
                disassembly, "Capability",
                "RayTraversalPrimitiveCullingKHR"))
                << "SkipAABBsKHR requires RayTraversalPrimitiveCullingKHR";
        }

        std::array<uint32_t, 6u> result{};
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << triangle_buffer.copy_from(luisa::span{triangles})
               << mesh.build()
               << accel.build()
               << shader(accel, output).dispatch(2u)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 0u &&
               result[1] == 1u &&
               result[2] == 1u &&
               result[3] == std::numeric_limits<uint32_t>::max() &&
               result[5] == 0u)
            << "direct closest/any surface tracing must select the nearest "
               "candidate and distinguish hit from miss; primitive index is "
               "intentionally unchecked for a miss";
    };

    "vk_user_compute_procedural_ray_query_commit_and_reject"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_procedural_ray_query"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        AABB box{};
        box.packed_min = {-0.5f, -0.5f, -0.5f};
        box.packed_max = {0.5f, 0.5f, 0.5f};
        const std::array boxes{box};
        auto box_buffer = device.create_buffer<AABB>(boxes.size());
        auto primitive =
            device.create_procedural_primitive(box_buffer.view());
        auto accel = device.create_accel();
        accel.emplace_back(primitive);
        auto result = device.create_buffer<uint32_t>(9u);
        auto distance = device.create_buffer<float>(3u);

        Kernel1D kernel = [](AccelVar accel_var, BufferUInt result_buffer,
                             BufferFloat distance_buffer) noexcept {
            auto lane = dispatch_x();
            auto origin_x = ite(lane == 2u, 2.0f, 0.0f);
            auto ray = make_ray(
                make_float3(origin_x, 0.0f, 2.0f),
                make_float3(0.0f, 0.0f, -1.0f), 0.0f, 10.0f);
            UInt callback_count = 0u;
            UInt candidate_primitive = ~0u;
            auto committed = accel_var.traverse(ray, {})
                                 .on_surface_candidate(
                                     [](SurfaceCandidate &) noexcept {})
                                 .on_procedural_candidate(
                                     [&](ProceduralCandidate &candidate) noexcept {
                                         callback_count += 1u;
                                         candidate_primitive =
                                             candidate.hit()->prim;
                                         $if (lane == 0u) {
                                             candidate.commit(1.75f);
                                         };
                                     })
                                 .trace();
            Float committed_distance = -1.0f;
            $if (committed->is_procedural()) {
                committed_distance = committed->distance();
            };
            result_buffer.write(lane * 3u, callback_count);
            result_buffer.write(lane * 3u + 1u, committed->hit_type);
            result_buffer.write(lane * 3u + 2u, candidate_primitive);
            distance_buffer.write(lane, committed_distance);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, 9u> result_values{};
        std::array<float, 3u> distance_values{};
        stream << box_buffer.copy_from(luisa::span{boxes})
               << primitive.build()
               << accel.build()
               << shader(accel, result, distance).dispatch(3u)
               << result.copy_to(luisa::span{result_values})
               << distance.copy_to(luisa::span{distance_values})
               << synchronize();

        constexpr auto miss = static_cast<uint32_t>(HitType::Miss);
        constexpr auto procedural =
            static_cast<uint32_t>(HitType::Procedural);
        constexpr auto invalid_primitive =
            std::numeric_limits<uint32_t>::max();
        expect(result_values == std::array<uint32_t, 9u>{
                                    1u, procedural, 0u,
                                    1u, miss, 0u,
                                    0u, miss, invalid_primitive})
            << "procedural traversal must distinguish committed, rejected, "
               "and absent AABB candidates";
        expect(std::abs(distance_values[0] - 1.75f) < 1.0e-6f &&
               distance_values[1] == -1.0f &&
               distance_values[2] == -1.0f)
            << "only the committed procedural candidate may expose its "
               "generated intersection distance";

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "procedural ray-query regression should emit one native "
               "SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(
                       disassembly,
                       "RayQueryGenerateIntersectionKHR") == 1u)
                << "procedural commit must lower to exactly one generated "
                   "intersection instruction at opt0";
            expect(count_spirv_opcode(
                       disassembly, "RayQueryProceedKHR") == 1u)
                << "the traversal loop must advance with one ray-query "
                   "proceed instruction at opt0";
        }
    };

    "vk_user_compute_triangle_ray_query_candidate_state"_test = [&] {
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_triangle_ray_query"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        const std::array vertices{
            float3{-0.5f, -0.5f, 0.0f},
            float3{0.5f, -0.5f, 0.0f},
            float3{0.0f, 0.5f, 0.0f}};
        const std::array triangles{Triangle{0u, 1u, 2u}};
        auto vertex_buffer = device.create_buffer<float3>(vertices.size());
        auto triangle_buffer =
            device.create_buffer<Triangle>(triangles.size());
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
        auto accel = device.create_accel();
        accel.emplace_back(
            mesh, make_float4x4(1.0f), 0xffu, false);
        auto uint_result = device.create_buffer<uint32_t>(12u);
        auto float_result = device.create_buffer<float>(36u);

        Kernel1D kernel = [](AccelVar accel_var, BufferUInt uint_output,
                             BufferFloat float_output) noexcept {
            auto lane = dispatch_x();
            auto origin_x = ite(lane == 2u, 2.0f, 0.0f);
            auto ray = make_ray(
                make_float3(origin_x, 0.0f, 2.0f),
                make_float3(0.0f, 0.0f, -1.0f), 0.25f, 8.0f);
            UInt callback_count = 0u;
            UInt candidate_inst = ~0u;
            UInt candidate_prim = ~0u;
            Float2 candidate_bary = make_float2(-1.0f);
            Float candidate_t = -1.0f;
            Float3 candidate_origin = make_float3(-1.0f);
            Float candidate_t_min = -1.0f;
            Float3 candidate_direction = make_float3(-1.0f);
            Float candidate_t_max = -1.0f;
            Float committed_t_max = -1.0f;
            auto committed = accel_var.traverse(ray, {})
                                 .on_surface_candidate(
                                     [&](SurfaceCandidate &candidate) noexcept {
                                         callback_count += 1u;
                                         auto hit = candidate.hit();
                                         candidate_inst = hit->inst;
                                         candidate_prim = hit->prim;
                                         candidate_bary = hit->bary;
                                         candidate_t = hit->distance();
                                         auto candidate_ray = candidate.ray();
                                         candidate_origin =
                                             candidate_ray->origin();
                                         candidate_t_min =
                                             candidate_ray->t_min();
                                         candidate_direction =
                                             candidate_ray->direction();
                                         candidate_t_max =
                                             candidate_ray->t_max();
                                         $if (lane == 0u) {
                                             candidate.commit();
                                             committed_t_max =
                                                 candidate.ray()->t_max();
                                         }
                                         $else {
                                             candidate.terminate();
                                         };
                                     })
                                 .on_procedural_candidate(
                                     [](ProceduralCandidate &) noexcept {})
                                 .trace();

            auto uint_base = lane * 4u;
            uint_output.write(uint_base, callback_count);
            uint_output.write(uint_base + 1u, candidate_inst);
            uint_output.write(uint_base + 2u, candidate_prim);
            uint_output.write(uint_base + 3u, committed->hit_type);
            auto float_base = lane * 12u;
            float_output.write(float_base, candidate_bary.x);
            float_output.write(float_base + 1u, candidate_bary.y);
            float_output.write(float_base + 2u, candidate_t);
            float_output.write(float_base + 3u, candidate_origin.x);
            float_output.write(float_base + 4u, candidate_origin.y);
            float_output.write(float_base + 5u, candidate_origin.z);
            float_output.write(float_base + 6u, candidate_t_min);
            float_output.write(float_base + 7u, candidate_direction.x);
            float_output.write(float_base + 8u, candidate_direction.y);
            float_output.write(float_base + 9u, candidate_direction.z);
            float_output.write(float_base + 10u, candidate_t_max);
            float_output.write(float_base + 11u, committed_t_max);
        };
        auto shader = device.compile(
            kernel, ShaderOption{.enable_cache = false,
                                 .enable_fast_math = false});

        std::array<uint32_t, 12u> uint_values{};
        std::array<float, 36u> float_values{};
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << triangle_buffer.copy_from(luisa::span{triangles})
               << mesh.build()
               << accel.build()
               << shader(accel, uint_result, float_result).dispatch(3u)
               << uint_result.copy_to(luisa::span{uint_values})
               << float_result.copy_to(luisa::span{float_values})
               << synchronize();

        constexpr auto surface = static_cast<uint32_t>(HitType::Surface);
        constexpr auto miss = static_cast<uint32_t>(HitType::Miss);
        constexpr auto invalid = std::numeric_limits<uint32_t>::max();
        expect(uint_values == std::array<uint32_t, 12u>{
                                  1u, 0u, 0u, surface,
                                  1u, 0u, 0u, miss,
                                  0u, invalid, invalid, miss})
            << "triangle traversal must distinguish committed, terminated, "
               "and absent candidates while preserving candidate IDs";
        auto expect_near = [&](size_t index, float expected) noexcept {
            expect(std::abs(float_values[index] - expected) < 1.0e-6f)
                << luisa::format(
                       "triangle ray-query float field {}: got {}, expected {}",
                       index, float_values[index], expected);
        };
        for (auto lane : {0u, 1u}) {
            auto base = static_cast<size_t>(lane) * 12u;
            expect_near(base, 0.25f);
            expect_near(base + 1u, 0.5f);
            expect_near(base + 2u, 2.0f);
            expect_near(base + 3u, 0.0f);
            expect_near(base + 4u, 0.0f);
            expect_near(base + 5u, 2.0f);
            expect_near(base + 6u, 0.25f);
            expect_near(base + 7u, 0.0f);
            expect_near(base + 8u, 0.0f);
            expect_near(base + 9u, -1.0f);
            expect_near(base + 10u, 8.0f);
        }
        expect_near(11u, 2.0f);
        expect_near(23u, -1.0f);

        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "triangle ray-query regression should emit one native "
               "SPIR-V module";
        if (dumps.size() == 1u) {
            auto disassembly = read_text_file(dumps.front());
            expect(count_spirv_opcode(
                       disassembly,
                       "RayQueryConfirmIntersectionKHR") == 1u);
            expect(count_spirv_opcode(
                       disassembly, "RayQueryTerminateKHR") == 1u);
            expect(count_spirv_opcode(
                       disassembly,
                       "RayQueryGetWorldRayOriginKHR") == 2u);
            expect(count_spirv_opcode(
                       disassembly,
                       "RayQueryGetWorldRayDirectionKHR") == 2u);
        }
    };

    "vk_user_compute_ray_instance_metadata_queries"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        std::array vertices{
            float3{-0.5f, -0.5f, 0.0f},
            float3{0.5f, -0.5f, 0.0f},
            float3{0.0f, 0.5f, 0.0f}};
        std::array indices{0u, 1u, 2u};
        auto vertex_buffer = device.create_buffer<float3>(vertices.size());
        auto triangle_buffer = device.create_buffer<Triangle>(1u);
        auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
        auto accel = device.create_accel({.allow_update = true});
        constexpr auto expected_visibility = static_cast<uint8_t>(0x5au);
        constexpr auto expected_user_id = 0x00c0ffeeu;
        const auto expected_transform = make_float4x4(
            float4{2.0f, 3.0f, 5.0f, 0.0f},
            float4{0.0f, 7.0f, 11.0f, 0.0f},
            float4{0.0f, 0.0f, 13.0f, 0.0f},
            float4{17.0f, 19.0f, 23.0f, 1.0f});
        const auto updated_transform = make_float4x4(
            float4{-2.0f, 1.0f, 0.0f, 0.0f},
            float4{0.0f, 3.0f, 1.0f, 0.0f},
            float4{1.0f, 0.0f, 4.0f, 0.0f},
            float4{-5.0f, 6.0f, 7.0f, 1.0f});
        accel.emplace_back(mesh, expected_transform,
                           expected_visibility, true, expected_user_id);

        Kernel1D query_kernel = [](AccelVar accel_var, BufferUInt output,
                                   BufferFloat4x4 transform_output) noexcept {
            output.write(0u, accel_var.instance_user_id(0));
            output.write(1u, accel_var.instance_visibility_mask(0u));
            transform_output.write(
                0u, accel_var.instance_transform(0u));
        };
        Kernel1D update_kernel = [](AccelVar accel_var,
                                    Float4x4 transform) noexcept {
            accel_var.set_instance_user_id(0u, 0x000badc0u);
            accel_var.set_instance_visibility(0, 0xa5u);
            accel_var.set_instance_transform(0u, transform);
        };
        auto shader = device.compile(query_kernel);
        auto update_shader = device.compile(update_kernel);
        auto output = device.create_buffer<uint32_t>(2u);
        auto transform_output = device.create_buffer<float4x4>(1u);
        std::array<uint32_t, 2u> result{};
        std::array<float4x4, 1u> transform_result{};
        stream << vertex_buffer.copy_from(luisa::span{vertices})
               << triangle_buffer.copy_from(luisa::span{indices})
               << mesh.build()
               << accel.build()
               << shader(accel, output, transform_output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << transform_output.copy_to(luisa::span{transform_result})
               << synchronize();

        expect(result[0] == expected_user_id);
        expect(result[1] == expected_visibility);
        for (auto column = 0u; column < 4u; ++column) {
            expect_vector_equal(
                transform_result[0][column],
                expected_transform[column]);
        }

        result.fill(0u);
        stream << update_shader(accel, updated_transform).dispatch(1u)
               << shader(accel, output, transform_output).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << transform_output.copy_to(luisa::span{transform_result})
               << synchronize();
        expect(result[0] == 0x000badc0u)
            << luisa::format("updated user ID mismatch: got 0x{:08x}", result[0]);
        expect(result[1] == 0xa5u)
            << luisa::format("updated visibility mismatch: got 0x{:08x}", result[1]);
        for (auto column = 0u; column < 4u; ++column) {
            expect_vector_equal(
                transform_result[0][column],
                updated_transform[column]);
        }
    };

    "vk_indirect_rejects_imported_native_writable_source_alias"_test = [&] {
        auto log_path = std::filesystem::absolute(
            "indirect_native_alias_rejection.log");
        auto command = luisa::format(
            "\"{}\" vk --indirect-native-alias-probe > \"{}\" 2>&1",
            executable_path, log_path.string());
        auto status = std::system(command.c_str());
        expect(status != 0)
            << "a distinct Luisa wrapper for the same VkBuffer must not bypass indirect-source alias validation";
        auto log = read_text_file(log_path);
        expect(log.find(
                   "aliases its GPU-authored source through a writable shader argument") !=
               std::string::npos)
            << "the child must fail for native VkBuffer alias validation, not an unrelated error";
    };

    "vk_indirect_allows_read_only_bindless_source_alias"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        auto commands = device.create_indirect_dispatch_buffer(1u);
        auto heap = device.create_bindless_array(1u);
        auto output = device.create_buffer<uint32_t>(1u);

        heap.emplace_buffer_handle_on_update(
            0u, commands.handle(), 0u, commands.size_bytes());
        Kernel1D author = [](Var<IndirectDispatchBuffer> target) noexcept {
            target.set_dispatch_count(1u);
            target.set_kernel(
                0u, make_uint3(1u), make_uint3(1u), 0u);
        };
        Kernel1D read = [](BindlessVar bindless,
                           BufferUInt result) noexcept {
            result.write(
                0u, bindless.buffer<uint32_t>(0u).read(0u));
        };
        auto author_shader = device.compile(
            author, ShaderOption{.enable_cache = false});
        auto read_shader = device.compile(
            read, ShaderOption{.enable_cache = false});

        std::array<uint32_t, 1u> result{};
        stream << heap.update()
               << author_shader(commands).dispatch(1u)
               << read_shader(heap, output).dispatch(commands)
               << output.copy_to(luisa::span{result})
               << synchronize();
        expect(result[0] == 1u)
            << "read-only bindless access to the indirect source is a valid read/read alias";
    };

    "vk_indirect_rejects_bindless_source_alias"_test = [&] {
        auto log_path = std::filesystem::absolute(
            "indirect_bindless_alias_rejection.log");
        auto command = luisa::format(
            "\"{}\" vk --indirect-bindless-alias-probe > \"{}\" 2>&1",
            executable_path, log_path.string());
        auto status = std::system(command.c_str());
        expect(status != 0)
            << "an indirect target must not reach its metadata source through a writable bindless descriptor";
        auto log = read_text_file(log_path);
        expect(log.find(
                   "binds its GPU-authored source through a writable bindless array") !=
               std::string::npos)
            << "the child must fail for bindless alias validation, not an unrelated error";
    };

    "vk_indirect_logical_size_is_authoritative_across_block_mismatch"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        constexpr auto writer_block_size = make_uint3(128u, 1u, 1u);
        constexpr auto target_block_size = make_uint3(64u, 1u, 1u);

        Kernel1D author = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(1u);
            commands.set_kernel(
                0u, writer_block_size, make_uint3(65u, 1u, 1u), 0u);
        };
        Kernel1D consume = [=](BufferUInt output) noexcept {
            set_block_size(target_block_size);
            output.atomic(0u).fetch_add(1u);
        };
        auto author_shader = device.compile(
            author, ShaderOption{.enable_cache = false});
        auto consume_shader = device.compile(
            consume, ShaderOption{.enable_cache = false});
        auto commands = device.create_indirect_dispatch_buffer(1u);
        auto output = device.create_buffer<uint32_t>(1u);
        std::array<uint32_t, 1u> zero{};
        std::array<uint32_t, 1u> result{};

        stream << output.copy_from(luisa::span{zero})
               << author_shader(commands).dispatch(1u)
               << consume_shader(output).dispatch(commands)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(eq(result[0], 65u))
            << "Vulkan preparation must recompute physical groups from the "
               "logical size and consuming shader block size";
    };

    "vk_indirect_first_record_write_initializes_zero_count_header"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();

        Kernel1D write_record_only = [](
                                         Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_kernel(
                0u, make_uint3(64u, 1u, 1u),
                make_uint3(17u, 1u, 1u), 0u);
        };
        Kernel1D consume = [](BufferUInt output) noexcept {
            output.atomic(0u).fetch_add(1u);
        };
        auto writer = device.compile(
            write_record_only, ShaderOption{.enable_cache = false});
        auto consumer = device.compile(
            consume, ShaderOption{.enable_cache = false});
        auto commands = device.create_indirect_dispatch_buffer(1u);
        auto output = device.create_buffer<uint32_t>(1u);
        std::array<uint32_t, 1u> zero{};
        std::array<uint32_t, 1u> result{};

        // Vulkan allocations are not required to start at zero. Poison the
        // count word through a non-owning wrapper so this test proves that the
        // first record-only authoring dispatch initializes the header rather
        // than merely observing convenient allocator contents.
        {
            auto raw_commands = device.import_external_buffer<uint32_t>(
                commands.native_handle(),
                commands.size_bytes() / sizeof(uint32_t));
            constexpr std::array poison{0xffffffffu};
            stream << raw_commands.view(0u, 1u).copy_from(
                          luisa::span{poison})
                   << synchronize();
        }

        stream << output.copy_from(luisa::span{zero})
               << writer(commands).dispatch(1u)
               << consumer(output).dispatch(commands)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(eq(result[0], 0u))
            << "the first GPU authoring use must zero the count header before a record-only write";
    };

    "vk_hlsl_writer_to_native_consumer_preserves_indirect_abi"_test = [&] {
        ScopedEnvironmentVariable disable_xir_optimization{
            "LUISA_XIR_DISABLE_OPTIMIZATION", "1"};
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        ScopedTemporaryCurrentPath work_dir{
            "luisa_vk_spirv_indirect_callable_metadata"};
        ScopedSourceDump source_dump;

        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        constexpr auto capacity = 8u;
        constexpr auto block_size = make_uint3(64u, 1u, 1u);

        Kernel1D set_count = [](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(3u);
        };
        Kernel1D write_records = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            auto i = dispatch_x();
            commands.set_kernel(
                i, block_size, make_uint3(i + 1u, 1u, 1u), i);
        };
        Callable read_identity = []() noexcept {
            return make_uint2(kernel_id(), dispatch_size_x());
        };
        Callable forward_identity = [&]() noexcept {
            return read_identity();
        };
        Kernel1D consume = [=, &forward_identity](BufferUInt output) noexcept {
            set_block_size(block_size);
            auto identity = forward_identity();
            output.atomic(identity.x).fetch_add(identity.y);
        };
        auto normalized_xir_path = std::filesystem::path{luisa::format(
            "kernel.{:016x}.norm.xir",
            consume.function()->function().hash())};

        ShaderOption fallback_option{.enable_cache = false};
        fallback_option.native_include = R"(
uint lc_indirect_writer_fallback_marker(uint value) { return value; }
)";
        auto count_shader = device.compile(set_count, fallback_option);
        auto record_shader = device.compile(write_records, fallback_option);
        auto consume_shader = device.compile(
            consume, ShaderOption{.enable_cache = false});
        auto commands =
            device.create_indirect_dispatch_buffer(capacity);
        auto output = device.create_buffer<uint32_t>(capacity);
        std::array<uint32_t, capacity> zeros{};
        std::array<uint32_t, capacity> result{};

        stream << output.copy_from(luisa::span{zeros})
               << count_shader(commands).dispatch(1u)
               << record_shader(commands).dispatch(capacity)
               // The host materializes records 1..4, and the GPU-authored
               // relative count of three permits records 1, 2, and 3.
               << consume_shader(output).dispatch(commands, 1u, 4u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        std::array<uint32_t, capacity> expected{};
        // Every logical invocation performs the atomic add, so records 1, 2,
        // and 3 contribute 2 * 2, 3 * 3, and 4 * 4. This checks both the
        // logical bounds guard and the dispatch-size/kernel-ID payload.
        expected[1] = 4u;
        expected[2] = 9u;
        expected[3] = 16u;
        expect(result == expected)
            << "Vulkan indirect dispatch must preserve logical size/kernel "
               "ID and zero stale records beyond the relative GPU count";
        expect(std::filesystem::exists(normalized_xir_path))
            << "indirect callable metadata fixture should emit normalized XIR";
        auto dumps = find_spirv_dumps();
        expect(dumps.size() == 1u)
            << "the native indirect consumer should emit one XIR-derived SPIR-V module";
        if (std::filesystem::exists(normalized_xir_path) &&
            dumps.size() == 1u) {
            auto normalized_xir = read_text_file(normalized_xir_path);
            auto disassembly = read_text_file(dumps.front());
            expect(count_substring(normalized_xir, "callable ") == 2u)
                << "both non-resource callables must survive to test transitive metadata forwarding";
            expect(count_spirv_opcode(disassembly, "FunctionCall") == 2u)
                << "indirect dispatch metadata must cross both callable boundaries";
            expect(count_spirv_opcode(disassembly, "Phi") >= 1u)
                << "the callable chain must receive metadata from a "
                   "control-dependent kernel merge; extra backend Phis are "
                   "not part of this source-level contract";
            expect(count_spirv_opcode(disassembly, "Select") == 0u)
                << "the indirect source loads must remain control-dependent";
        }
    };
    "vk_indirect_writer_rejects_out_of_capacity_and_invalid_records"_test = [&] {
        ScopedEnvironmentVariable disable_spirv_optimization{
            "LUISA_SPIRV_OPT_LEVEL", "0"};
        ScopedEnvironmentVariable clear_spirv_pass_override{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        constexpr auto capacity = 4u;
        constexpr auto block_size = make_uint3(64u, 1u, 1u);

        Kernel1D set_count = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_dispatch_count(capacity);
        };
        Kernel1D write_out_of_capacity = [=](
                                             Var<IndirectDispatchBuffer> commands) noexcept {
            commands.set_kernel(
                capacity, block_size,
                make_uint3(7u, 1u, 1u), 0u);
        };
        Kernel1D write_records = [=](Var<IndirectDispatchBuffer> commands) noexcept {
            auto i = dispatch_x();
            $if (i == 1u) {
                commands.set_kernel(
                    i, block_size, make_uint3(2u, 1u, 1u), i);
            }
            $else {
                // Any zero block component invalidates the complete physical
                // command, while preserving a well-defined logical record.
                commands.set_kernel(
                    i, make_uint3(0u, 1u, 1u),
                    make_uint3(99u, 1u, 1u), i);
            };
            $if (i == 0u) {
                // This record index is exactly one past the allocation and
                // must take no record-store path.
                commands.set_kernel(
                    capacity, block_size,
                    make_uint3(7u, 1u, 1u), 0u);
            };
        };
        Kernel1D consume = [=](BufferUInt output) noexcept {
            set_block_size(block_size);
            output.atomic(kernel_id()).fetch_add(dispatch_size_x());
        };

        auto count_shader = device.compile(
            set_count, ShaderOption{.enable_cache = false});
        auto out_of_capacity_shader = device.compile(
            write_out_of_capacity,
            ShaderOption{.enable_cache = false});
        auto record_shader = device.compile(
            write_records, ShaderOption{.enable_cache = false});
        auto consume_shader = device.compile(
            consume, ShaderOption{.enable_cache = false});

        auto structural = lc::spirv::SpirvCodegenEntry::compile_spirv(
            write_out_of_capacity.function()->function(),
            ShaderOption{.enable_cache = false});
        auto guard = inspect_spirv_indirect_record_guard(
            structural.spv_bin);
        expect(guard.exact_capacity_dataflow)
            << "the indirect writer must derive capacity as "
               "(OpArrayLength - header words) / record words and compare "
               "the requested index with OpULessThan";
        expect(guard.record_stores_are_control_dependent)
            << "all record OpStore instructions must be in the true target of "
               "the exact index-capacity branch";

        {
            auto guard_commands =
                device.create_indirect_dispatch_buffer(capacity);
            auto raw_commands = device.import_external_buffer<uint32_t>(
                guard_commands.native_handle(),
                guard_commands.size_bytes() / sizeof(uint32_t));
            constexpr auto canary = 0xa5a5a5a5u;
            std::vector<uint32_t> canaries(
                raw_commands.size(), canary);
            std::vector<uint32_t> observed(canaries.size());

            // Establish the resource's first-authoring state before poisoning
            // every in-allocation word. This separates the writer guard from
            // the runtime's one-time count-header initialization.
            stream << out_of_capacity_shader(guard_commands).dispatch(1u)
                   << synchronize()
                   << raw_commands.copy_from(luisa::span{canaries})
                   << synchronize()
                   << out_of_capacity_shader(guard_commands).dispatch(1u)
                   << raw_commands.copy_to(luisa::span{observed})
                   << synchronize();
            expect(observed == canaries)
                << "an index equal to capacity must not wrap into or alias any "
                   "word inside the indirect-dispatch allocation";
        }

        auto commands =
            device.create_indirect_dispatch_buffer(capacity);
        auto output = device.create_buffer<uint32_t>(capacity);
        std::array<uint32_t, capacity> zero{};
        std::array<uint32_t, capacity> result{};

        stream << output.copy_from(luisa::span{zero})
               << count_shader(commands).dispatch(1u)
               << record_shader(commands).dispatch(capacity)
               << consume_shader(output).dispatch(commands)
               << output.copy_to(luisa::span{result})
               << synchronize();

        std::array<uint32_t, capacity> expected{};
        expected[1] = 4u;
        expect(result == expected)
            << "OOB record writes and zero block sizes must "
               "leave only the valid indirect record executable";
    };
    "vk_indirect_host_empty_plan_is_a_true_noop"_test = [&] {
        auto dc = luisa::test::create_device(argc, argv);
        auto &device = dc.device;
        auto stream = device.create_stream();
        constexpr auto capacity = 4u;

        Kernel1D write = [](BufferUInt output, UInt index,
                            UInt value) noexcept {
            output.write(index, value);
        };
        auto shader = device.compile(
            write, ShaderOption{.enable_cache = false});
        ShaderOption fallback_option{.enable_cache = false};
        fallback_option.native_include = R"(
uint lc_empty_indirect_fallback_marker(uint value) { return value; }
)";
        auto fallback_shader = device.compile(write, fallback_option);
        auto commands =
            device.create_indirect_dispatch_buffer(capacity);
        auto output = device.create_buffer<uint32_t>(4u);
        std::array<uint32_t, 4u> zero{};
        std::array<uint32_t, 4u> result{};

        stream << output.copy_from(luisa::span{zero})
               // A host-proven empty plan must not inspect the target's
               // codegen ABI. This shader deliberately uses the HLSL fallback,
               // which is incompatible with nonempty Vulkan indirect dispatch.
               << fallback_shader(output, 0u, 5u)
                      .dispatch(commands, 0u, 0u)
               // A zero maximum count is empty regardless of its valid
               // source offset.
               << shader(output, 1u, 7u).dispatch(commands, 0u, 0u)
               // offset == capacity proves the host plan is empty. It must
               // neither require a GPU-authored header nor consume an
               // argument-offset entry for this uniform-bearing invocation.
               << shader(output, 2u, 11u)
                      .dispatch(commands, capacity, 1u)
               << shader(output, 3u, 29u).dispatch(1u)
               << output.copy_to(luisa::span{result})
               << synchronize();

        expect(result == std::array<uint32_t, 4u>{0u, 0u, 0u, 29u})
            << "an empty indirect host range must execute no shader and must "
               "not disturb the following direct dispatch's uniform payload";
    };
    return 0;
}
