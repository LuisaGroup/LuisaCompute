#pragma once
#include <volk.h>
#include "resource.h"
#include "stream.h"
namespace lc::vk {
class Swapchain : public Resource {
    VkSurfaceKHR _surface{};
    VkSurfaceFormatKHR _swapchain_format{};
    VkSampler _texture_sampler{nullptr};
    VkDescriptorSetLayout _descriptor_set_layout{nullptr};
    VkRenderPass _render_pass{};
    VkPipeline _graphics_pipeline{};
    VkExtent2D _swapchain_extent{};
    VkPipelineLayout _pipeline_layout{};
    VkSwapchainKHR _swapchain{};
    luisa::vector<VkImage> _swapchain_images;
    luisa::vector<VkImageView> _swapchain_image_views;
    luisa::vector<VkFramebuffer> _swapchain_framebuffers;
    luisa::vector<VkSemaphore> _image_available_semaphores;
    luisa::vector<VkSemaphore> _render_finished_semaphores;
    luisa::vector<VkFence> _in_flight_fences;

    size_t _current_frame{0u};

    VkBuffer _vertex_buffer{nullptr};
    VkDeviceMemory _vertex_buffer_memory{nullptr};
    uint64_t _display_handle{};
    uint64_t _window_handle{};
    uint2 _requested_size;
    // The surface's currentTransform reported by the swapchain. On Android the
    // swapchain is created with preTransform=IDENTITY and the application is
    // responsible for rotating content by this amount (90/180/270 degrees).
    VkSurfaceTransformFlagBitsKHR _surface_transform{
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR};
    bool _requested_hdr{false};
    bool _requested_vsync{false};
    bool _requested_transparent{false};
    void _recreate_swapchain();
    void _destroy_swapchain();

public:
    explicit Swapchain(Device *device);
    [[nodiscard]] auto swapchain() const { return _swapchain; }
    [[nodiscard]] auto surface_transform() const noexcept {
        return _surface_transform;
    }
    [[nodiscard]] bool is_hdr() const noexcept;
    ~Swapchain();
    void create_swapchain(
        uint64_t display_handle,
        uint64_t window_handle,
        uint width, uint height, uint back_buffers, bool is_recreation, bool allow_hdr, bool vsync,
        bool transparent = false);
    void present(
        CommandBuffer &cmdbuffer,
        // submit info
        VkSemaphore &wait, VkSemaphore &signal,
        VkPipelineStageFlags &wait_stage,
        // present info
        VkSemaphore &present_wait,
        uint &image_index,
        Texture const *tex,
        VkFence &fence,
        uint mip);
    // Handle swapchain recreation due to present-time errors (OUT_OF_DATE/SUBOPTIMAL)
    void handle_present_error();
};
}// namespace lc::vk