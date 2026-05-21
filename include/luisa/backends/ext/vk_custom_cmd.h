#pragma once
#include <vulkan/vulkan_core.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/stack_allocator.h>
namespace luisa::compute {
static constexpr auto raster_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

static constexpr VkPipelineStageFlagBits2 BarrierSyncMap[] = {
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeAccelRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeUAV,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // CopySource,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // CopyDest,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // BuildAccel,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // BuildAccelScratch,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // CopyAccelSrc
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // CopyAccelDst
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//DepthRead
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//DepthWrite
    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,                                                     //IndirectArgs
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,                                            //VertexRead,
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,                                                       //  IndexRead,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,                                           //  RenderTarget
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // AccelInstanceBuffer
    raster_stage,                                                                              // RasterRead
    raster_stage,                                                                              //RasterAccelRead
    raster_stage                                                                               //RasterUAV
};
static constexpr VkAccessFlagBits2 BarrierAccessMap[] = {
    VK_ACCESS_2_SHADER_READ_BIT,                     // ComputeRead,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, // ComputeAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,            // ComputeUAV,
    VK_ACCESS_2_TRANSFER_READ_BIT,                   // CopySource,
    VK_ACCESS_2_TRANSFER_WRITE_BIT,                  // CopyDest,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,// BuildAccel,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,            // BuildAccelScratch,
    VK_ACCESS_2_TRANSFER_READ_BIT,                   // CopyAccelSrc
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,// CopyAccelDst
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,   //DepthRead
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,  //DepthWrite
    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,           // IndirectArgs
    VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,           //VertexRead,
    VK_ACCESS_2_INDEX_READ_BIT,                      //  IndexRead,
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,          //RenderTarget
    VK_ACCESS_2_SHADER_READ_BIT,                     //AccelInstanceBuffer
    VK_ACCESS_2_SHADER_READ_BIT,                     // RasterRead
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, // RasterAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,            // RasterUAV,
};
static constexpr VkAccessFlagBits2 write_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
static constexpr VkImageLayout BarrierLayoutMap[] = {
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // ComputeRead,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // ComputeAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // ComputeUAV,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,            // CopySource,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,            // CopyDest,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // BuildAccel,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // BuildAccelScratch,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // CopyAccelSrc
    VK_IMAGE_LAYOUT_UNDEFINED,                       // CopyAccelDst
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, //DepthRead
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,//DepthWrite
    VK_IMAGE_LAYOUT_UNDEFINED,                       // IndirectArgs
    VK_IMAGE_LAYOUT_UNDEFINED,                       //VertexRead,
    VK_IMAGE_LAYOUT_UNDEFINED,                       //  IndexRead,
    VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,              //RenderTarget
    VK_IMAGE_LAYOUT_UNDEFINED,                       //AccelInstanceBuffer
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // RasterRead
    VK_IMAGE_LAYOUT_UNDEFINED,                       // RasterAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // RasterUAV,
};
class VKCustomCmd : public CustomDispatchCommand {
public:
    enum class ResourceUsageType : uint {
        ComputeRead,
        ComputeAccelRead,
        ComputeUAV,
        CopySource,
        CopyDest,
        BuildAccel,
        BuildAccelScratch,
        CopyAccelSrc,
        CopyAccelDst,
        DepthRead,
        DepthWrite,
        IndirectArgs,
        VertexRead,
        IndexRead,
        RenderTarget,
        AccelInstanceBuffer,
        RasterRead,
        RasterAccelRead,
        RasterUAV,
    };
    using ResourceHandle = luisa::variant<
        Argument::Buffer,
        Argument::Texture,
        Argument::BindlessArray>;

    struct ResourceUsage {
        ResourceHandle resource;
        VkPipelineStageFlagBits2 stage;
        VkAccessFlagBits2 access;
        VkImageLayout texture_layout;
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        ResourceUsage(
            Arg &&resource,
            VkPipelineStageFlagBits2 stage,
            VkAccessFlagBits2 access,
            VkImageLayout texture_layout = VK_IMAGE_LAYOUT_UNDEFINED) noexcept
            : resource(std::forward<Arg>(resource)),
              stage(stage),
              access(access),
              texture_layout(texture_layout) {
        }
        template<typename Arg>
            requires(luisa::is_constructible_v<ResourceHandle, Arg &&>)
        ResourceUsage(
            Arg &&resource,
            ResourceUsageType type) noexcept
            : resource(std::forward<Arg>(resource)),
              stage(BarrierSyncMap[luisa::to_underlying(type)]),
              access(BarrierAccessMap[luisa::to_underlying(type)]),
              texture_layout(BarrierLayoutMap[luisa::to_underlying(type)]) {
        }
    };
private:
    [[nodiscard]] static auto resource_state_to_usage(
        VkAccessFlagBits2 state) noexcept {
        static VkAccessFlagBits2 read = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                                        VK_ACCESS_2_INDEX_READ_BIT |
                                        VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                        VK_ACCESS_2_UNIFORM_READ_BIT |
                                        VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_SHADER_READ_BIT |
                                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_2_TRANSFER_READ_BIT |
                                        VK_ACCESS_2_HOST_READ_BIT |
                                        VK_ACCESS_2_MEMORY_READ_BIT |
                                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                        VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                        VK_ACCESS_2_VIDEO_DECODE_READ_BIT_KHR |
                                        VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR |
                                        VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT |
                                        VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT |
                                        VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV |
                                        VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT |
                                        VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR |
                                        VK_ACCESS_2_SHADING_RATE_IMAGE_READ_BIT_NV |
                                        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                        VK_ACCESS_2_FRAGMENT_DENSITY_MAP_READ_BIT_EXT |
                                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT |
                                        VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT |
                                        VK_ACCESS_2_INVOCATION_MASK_READ_BIT_HUAWEI |
                                        VK_ACCESS_2_SHADER_BINDING_TABLE_READ_BIT_KHR |
                                        VK_ACCESS_2_MICROMAP_READ_BIT_EXT |
                                        VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV;
        static VkAccessFlagBits2 write = VK_ACCESS_2_SHADER_WRITE_BIT |
                                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                         VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                         VK_ACCESS_2_HOST_WRITE_BIT |
                                         VK_ACCESS_2_MEMORY_WRITE_BIT |
                                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                         VK_ACCESS_2_VIDEO_DECODE_WRITE_BIT_KHR |
                                         VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR |
                                         VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
                                         VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT |
                                         VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV |
                                         VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT |
                                         VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                         VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT |
                                         VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV;
        if (state == 0) return Usage::READ_WRITE;
        Usage usage{};
        if ((state & read) != 0) {
            usage = static_cast<Usage>(static_cast<uint32_t>(usage) | static_cast<uint32_t>(Usage::READ));
        }
        if ((state & write) != 0) {
            usage = static_cast<Usage>(static_cast<uint32_t>(usage) | static_cast<uint32_t>(Usage::WRITE));
        }
        return usage;
    }
public:
    [[nodiscard]] virtual luisa::span<ResourceUsage> get_resource_usages() noexcept {
        return {};
    }
    VKCustomCmd() noexcept = default;
    virtual ~VKCustomCmd() noexcept override = default;
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return static_cast<uint32_t>(CustomCommandUUID::CUSTOM_DISPATCH);
    }
    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        auto usages = const_cast<VKCustomCmd *>(this)->get_resource_usages();
        for (auto &&[handle, stage, access, layout] : usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(access));
            },
                         handle);
        }
    }
    void traverse_arguments(MutableArgumentVisitor &visitor) noexcept override {
        auto usages = get_resource_usages();
        for (auto &&[handle, stage, access, layout] : usages) {
            luisa::visit([&](auto &&v) {
                visitor.visit(v, resource_state_to_usage(access));
            },
                         handle);
        }
    }
    virtual void execute(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandBuffer cmdbuffer,
        VkDescriptorPool desc_pool) const noexcept = 0;
};
}// namespace luisa::compute