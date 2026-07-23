#pragma once
#include <volk.h>
#include <atomic>

namespace lc::vk {

class Device;

// Reference-counted VkPipeline + VkPipelineCache pair.
// Both ComputeShader and RayTracingShader share this pattern:
// their pipelines must outlive any command buffer that references them.
//
// Usage:
//   PipelineRef *ref = PipelineRef::create(device, pipeline, cache);
//   ref->retain();  // on dispatch into a command buffer
//   ref->release(); // when the command buffer completes, or shader destruction
struct PipelineRef {

    VkPipeline pipeline{VK_NULL_HANDLE};
    VkPipelineCache pipeline_cache{VK_NULL_HANDLE};
    VkDevice device{};
    VkAllocationCallbacks const *alloc_callbacks{};
    std::atomic<uint32_t> ref_count{1u};

    static PipelineRef *create(VkDevice device,
                               VkPipeline pipeline,
                               VkPipelineCache pipeline_cache,
                               VkAllocationCallbacks const *alloc_callbacks) noexcept {
        auto ptr = new PipelineRef{};
        ptr->pipeline = pipeline;
        ptr->pipeline_cache = pipeline_cache;
        ptr->device = device;
        ptr->alloc_callbacks = alloc_callbacks;
        ptr->ref_count.store(1u, std::memory_order_relaxed);
        return ptr;
    }

    void retain() noexcept {
        ref_count.fetch_add(1u, std::memory_order_relaxed);
    }

    void release() noexcept {
        if (ref_count.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            do_destroy();
            delete this;
        }
    }

private:
    PipelineRef() noexcept = default;
    ~PipelineRef() noexcept = default;

    void do_destroy() noexcept {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, alloc_callbacks);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipeline_cache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device, pipeline_cache, alloc_callbacks);
            pipeline_cache = VK_NULL_HANDLE;
        }
    }
};

// Movable, non-trivially-destructible holder for use with
// CommandBufferState::dispose_after_flush (which requires a type
// with a non-trivial destructor).
struct PipelineRefHolder {
    PipelineRef *ref{nullptr};

    explicit PipelineRefHolder(PipelineRef *r) noexcept
        : ref{r} {
        if (ref) { ref->retain(); }
    }

    PipelineRefHolder(const PipelineRefHolder &other) noexcept
        : ref{other.ref} {
        if (ref) { ref->retain(); }
    }

    PipelineRefHolder(PipelineRefHolder &&other) noexcept
        : ref{other.ref} {
        other.ref = nullptr;
    }

    ~PipelineRefHolder() noexcept {
        if (ref) { ref->release(); }
    }

    PipelineRefHolder &operator=(const PipelineRefHolder &) = delete;
    PipelineRefHolder &operator=(PipelineRefHolder &&) = delete;
};

}// namespace lc::vk
