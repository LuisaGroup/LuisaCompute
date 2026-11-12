#pragma once
#include <volk.h>
#include "resource.h"
#include "stream.h"
#include "present_plan.h"
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

    // ---- optional present / frame-pacing extensions ----------------------
    // Instance extensions this swapchain may rely on. Vulkan cannot report the
    // enabled instance extensions, so they are attested from
    // VulkanDeviceConfigExt::extra_instance_exts() (see present_plan.h); an
    // extension that is missing here is never used, not even its entry points.
    detail::PresentInstanceExtensions _present_instance_extensions{};
    // Negotiated plans. `chain_*`/`collect_*` are the only switches the
    // swapchain and the present path look at, so a disabled or unsupported
    // extension always falls back to the pre-existing behaviour.
    detail::PresentModeSelection _present_mode_selection{};
    detail::PresentId2Plan _present_id_plan{};
    detail::PresentTimingPlan _present_timing_plan{};
    bool _present_mode_logged{false};
    // Monotonic present-id namespace. Never reset, not even by recreation (see
    // plan_present_id in present_plan.h).
    uint64_t _next_present_id{1u};
    uint64_t _last_present_id{0u};
    // First id of the current VkSwapchainKHR generation. Ids below it belong to
    // a destroyed swapchain and must never be waited on.
    uint64_t _present_id_generation_floor{1u};
    // Value referenced by `_present_id2_chain.pPresentIds`. Kept alive here so
    // the pNext node handed to the caller's vkQueuePresentKHR stays valid until
    // the next present() on this swapchain.
    uint64_t _present_id_value{0u};
    VkPresentId2KHR _present_id2_chain{};

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

    // ---- optional present extensions -------------------------------------
    // All of these report the negotiated fast path. `false` means the
    // pre-existing path (plain VkPresentInfoKHR + timeline/queue-idle
    // synchronization) is in use, which is and stays fully correct.
    [[nodiscard]] bool fifo_latest_ready_active() const noexcept {
        return _present_mode_selection.used_fifo_latest_ready;
    }
    [[nodiscard]] bool present_id2_active() const noexcept {
        return _present_id_plan.chain_present_ids;
    }
    [[nodiscard]] bool present_wait2_active() const noexcept {
        return _present_id_plan.wait_for_present_ids;
    }
    [[nodiscard]] detail::PresentId2Plan const &present_id_plan() const noexcept {
        return _present_id_plan;
    }
    [[nodiscard]] detail::PresentTimingPlan const &present_timing_plan() const noexcept {
        return _present_timing_plan;
    }
    // Present id of the most recent present request, 0 while present-id2 is
    // inactive.
    [[nodiscard]] uint64_t last_present_id() const noexcept {
        return _last_present_id;
    }
    // pNext node to chain into the VkPresentInfoKHR that presents the image
    // recorded by the last present() call:
    //     present_info.pNext = swapchain->present_id_pnext();
    // nullptr while present-id2 is inactive, so the unmodified present call
    // stays legal. The node is owned by this swapchain and stays valid until
    // the next present() call.
    [[nodiscard]] const void *present_id_pnext() const noexcept;
    // Wait until the presentation engine started presenting `present_id`
    // (vkWaitForPresent2KHR). Returns false when present-wait2 is inactive or
    // the wait did not succeed; the caller then keeps its existing
    // synchronization. Never blocks longer than `timeout_ns`.
    [[nodiscard]] bool wait_for_present_id(uint64_t present_id, uint64_t timeout_ns) noexcept;
    [[nodiscard]] bool wait_for_last_present(uint64_t timeout_ns) noexcept;
};
}// namespace lc::vk
