#include "stream.h"
#include "device.h"
#include "compute_shader.h"
#include "bindless_array.h"
#include <luisa/core/logging.h>
#include "log.h"
#include "blas.h"
#include "tlas.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include <luisa/runtime/swapchain.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
#include "../common/shader_print_formatter.h"
#include "raster_shader.h"
namespace lc::vk {
struct PresentCommand {
    luisa::fixed_vector<VkSemaphore, 1> submit_wait_semaphores;
    luisa::fixed_vector<VkSemaphore, 1> signal_semaphores;
    luisa::fixed_vector<VkPipelineStageFlags, 1> wait_stages;
    luisa::fixed_vector<VkSemaphore, 1> present_wait_semaphores;
    luisa::fixed_vector<uint, 1> image_indices;
};
template<typename Visitor>
void DecodeCmd(vstd::span<const Argument> args, Visitor &&visitor) {
    using Tag = Argument::Tag;
    for (auto &&i : args) {
        switch (i.tag) {
            case Tag::BUFFER: {
                visitor(i.buffer);
            } break;
            case Tag::TEXTURE: {
                visitor(i.texture);
            } break;
            case Tag::UNIFORM: {
                visitor(i.uniform);
            } break;
            case Tag::BINDLESS_ARRAY: {
                visitor(i.bindless_array);
            } break;
            case Tag::ACCEL: {
                visitor(i.accel);
            } break;
            default: {
                LUISA_ASSUME(false);
            } break;
        }
    }
}
ResourceBarrier::ResourceView get_resource_view(VKCustomCmd::ResourceHandle const &res) {
    return luisa::visit(
        [&]<typename T>(T const &t) -> ResourceBarrier::ResourceView {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                auto buffer = reinterpret_cast<Buffer const *>(t.handle);
                return BufferView(buffer, t.offset, t.size);
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                auto tex = reinterpret_cast<Texture const *>(t.handle);
                return TexView(tex, t.level);
            } else {
                auto bdls = reinterpret_cast<BindlessArray const *>(t.handle);
                auto &buffer = bdls->indices_buffer();
                return BufferView(&buffer, 0, buffer.byte_size());
            }
        },
        res);
}
bool ReorderFuncTable::is_res_in_bindless(uint64_t bindless_handle, uint64_t resource_handle) const noexcept {
    return reinterpret_cast<BindlessArray *>(bindless_handle)->is_ptr_in_bindless(resource_handle);
}
void ReorderFuncTable::lock_bindless(uint64_t bindless_handle) const noexcept {
    reinterpret_cast<BindlessArray *>(bindless_handle)->mtx.lock();
}
void ReorderFuncTable::unlock_bindless(uint64_t bindless_handle) const noexcept {
    reinterpret_cast<BindlessArray *>(bindless_handle)->mtx.unlock();
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Modification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::BufferModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture2DModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind(modifications);
}
void ReorderFuncTable::update_bindless(uint64_t handle, luisa::span<const BindlessArrayUpdateCommand::Texture3DModification> modifications) const noexcept {
    reinterpret_cast<BindlessArray *>(handle)->bind({reinterpret_cast<const BindlessArrayUpdateCommand::Texture2DModification *>(modifications.data()),
                                                     modifications.size()});
}
struct ResourceBarrierVisitor {
    ResourceBarrier *barrier;
    SavedArgument const *arg;
    vstd::vector<std::byte> *arg_buffer;
    ShaderDispatchCommandBase const &cmd;
    ResourceBarrier::Usage uav_usage;
    ResourceBarrier::Usage read_usage;
    ResourceBarrier::Usage accel_read_usage;
    template<typename T>
    void emplace_data(T const &data, size_t alignment) {
        size_t sz = arg_buffer->size();
        alignment -= 1;
        auto aligned_size = (sz + alignment) & (~alignment);
        luisa::enlarge_by(*arg_buffer, sizeof(T) + aligned_size - sz);
        using PlaceHolder = luisa::aligned_storage_t<sizeof(T), 1>;
        *reinterpret_cast<PlaceHolder *>(arg_buffer->data() + aligned_size) =
            *reinterpret_cast<PlaceHolder const *>(&data);
    }
    template<typename T>
    void emplace_data(T const *data, size_t size, size_t alignment) {
        alignment -= 1;
        size_t sz = arg_buffer->size();
        auto aligned_size = (sz + alignment) & (~alignment);
        auto byteSize = size * sizeof(T);
        luisa::enlarge_by(*arg_buffer, byteSize + aligned_size - sz);
        std::memcpy(arg_buffer->data() + aligned_size, data, byteSize);
    }
    ResourceBarrierVisitor(
        ResourceBarrier *barrier,
        SavedArgument const *arg,
        vstd::vector<std::byte> *arg_buffer,
        ShaderDispatchCommandBase const &cmd,
        bool is_raster) : barrier(barrier), arg(arg), arg_buffer(arg_buffer), cmd(cmd) {
        if (is_raster) {
            uav_usage = ResourceBarrier::Usage::RasterUAV;
            read_usage = ResourceBarrier::Usage::RasterRead;
            accel_read_usage = ResourceBarrier::Usage::RasterAccelRead;
        } else {
            uav_usage = ResourceBarrier::Usage::ComputeUAV;
            read_usage = ResourceBarrier::Usage::ComputeRead;
            accel_read_usage = ResourceBarrier::Usage::ComputeAccelRead;
        }
    }
    void operator()(Argument::Buffer const &bf) {
        auto res = reinterpret_cast<Buffer const *>(bf.handle);
        if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
            // LUISA_ASSERT(is_device_buffer(res), "Unordered access buffer can not be host-buffer.");
            barrier->record(
                BufferView{res, bf.offset, bf.size},
                uav_usage);
        } else {
            barrier->record(
                BufferView{res, bf.offset, bf.size},
                read_usage);
        }
        ++arg;
    }
    void operator()(Argument::Texture const &bf) {
        auto rt = reinterpret_cast<Texture *>(bf.handle);
        //UAV
        if (((uint)arg->varUsage & (uint)Usage::WRITE) != 0) {
            if (!rt->allow_uav()) {
                LUISA_ERROR("Texture not allowed for Unordered-Access.");
            }
            barrier->record(
                TexView{rt, bf.level},
                uav_usage);
        }
        // SRV
        else {
            barrier->record(
                TexView{rt, bf.level},
                read_usage);
        }
        ++arg;
    }
    void operator()(Argument::BindlessArray const &bf) {
        auto bdls = reinterpret_cast<BindlessArray *>(bf.handle);
        auto &buffer = bdls->indices_buffer();
        barrier->record(
            BufferView(&buffer, 0, buffer.byte_size()),
            read_usage);
        barrier->process_bindless(bdls, read_usage);
        ++arg;
    }
    void operator()(Argument::Uniform const &a) {
        auto bf = cmd.uniform(a);
        emplace_data(bf.data(), bf.size_bytes(), a.alignment);
        ++arg;
    }
    void operator()(Argument::Accel const &bf) {
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        if (!tlas->instance_buffer()) [[unlikely]] {
            LUISA_ERROR("Accel not initialized.");
        }
        if ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) {
            barrier->record(
                BufferView(tlas->instance_buffer()),
                ResourceBarrier::Usage::ComputeUAV);
        } else {
            if (!tlas->accel_buffer()) [[unlikely]] {
                LUISA_ERROR("Accel not initialized.");
            }
            barrier->record(
                BufferView(tlas->instance_buffer()),
                ResourceBarrier::Usage::ComputeRead);
            barrier->record(
                BufferView(tlas->accel_buffer()),
                ResourceBarrier::Usage::ComputeAccelRead);
        }
        ++arg;
    }
};
struct BindPropVisitor {
    // Each sets
    CommandBuffer *cmdbuffer;
    VkDescriptorSet desc_set;
    uint desc_index;
    vstd::vector<VkImageView> *img_views;
    SavedArgument const *arg;
    void operator()(Argument::Buffer const &bf) {
        auto idx = desc_index++;
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        *buffer_descs = VkDescriptorBufferInfo{
            reinterpret_cast<Buffer const *>(bf.handle)->vk_buffer(),
            bf.offset,
            bf.size};
        cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Texture const &bf) {
        auto idx = desc_index++;
        auto tex = reinterpret_cast<Texture const *>(bf.handle);

        VkImageViewCreateInfo imgview_create_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            tex->vk_image(),
            VkImageViewType(tex->dimension() - 1),
            Texture::to_vk_format(tex->format()),
            VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
            VkImageSubresourceRange{
                tex->get_aspect(),
                bf.level,
                1,
                0,
                1}};
        VkImageView img_view;
        VK_CHECK_RESULT(vkCreateImageView(cmdbuffer->device()->logic_device(), &imgview_create_info, Device::alloc_callbacks(), &img_view));
        img_views->emplace_back(img_view);
        auto image_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorImageInfo>();
        *image_descs = VkDescriptorImageInfo{
            VkSampler{nullptr},
            img_view,
            cmdbuffer->resource_barrier->get_layout(tex, bf.level)};

        cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            image_descs,
            nullptr,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Uniform const &a) {
        ++arg;
    }
    void operator()(Argument::BindlessArray const &bf) {
        auto &buffer = reinterpret_cast<BindlessArray const *>(bf.handle)->indices_buffer();
        auto idx = desc_index++;
        auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
        *buffer_descs = VkDescriptorBufferInfo{
            buffer.vk_buffer(),
            0,
            buffer.byte_size()};
        auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            desc_set,
            idx,
            0,
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            nullptr,
            buffer_descs,
            nullptr});
        ++arg;
    }
    void operator()(Argument::Accel const &bf) {
        auto tlas = reinterpret_cast<Tlas *>(bf.handle);
        if ((luisa::to_underlying(arg->varUsage) & luisa::to_underlying(Usage::WRITE)) != 0) {
            auto idx = desc_index++;
            auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
            *buffer_descs = VkDescriptorBufferInfo{
                tlas->instance_buffer()->vk_buffer(),
                0,
                tlas->instance_buffer()->byte_size()};
            auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                desc_set,
                idx,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                buffer_descs,
                nullptr});
        } else {
            // accel
            {
                auto idx = desc_index++;
                auto accel_info = cmdbuffer->temp_desc->allocate_memory<VkWriteDescriptorSetAccelerationStructureKHR>();
                accel_info->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                accel_info->accelerationStructureCount = 1;
                accel_info->pAccelerationStructures = &tlas->accel();
                auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    accel_info,
                    desc_set,
                    idx,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    nullptr,
                    nullptr,
                    nullptr});
            }
            // instance
            {
                auto idx = desc_index++;
                auto buffer_descs = cmdbuffer->temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *buffer_descs = VkDescriptorBufferInfo{
                    tlas->instance_buffer()->vk_buffer(),
                    0,
                    tlas->instance_buffer()->byte_size()};
                auto &a = cmdbuffer->write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    desc_set,
                    idx,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    buffer_descs,
                    nullptr});
            }
        }
        ++arg;
    }
};
namespace temp_buffer {
uint64 DefaultBufferDeferredVisitor::allocate(uint64 size) {
    auto bf = new DefaultBuffer(device, size);
    _buffers.try_emplace(
        reinterpret_cast<uint64_t>(bf),
        bf);
    return reinterpret_cast<uint64>(bf);
}
void DefaultBufferDeferredVisitor::deallocate(uint64 handle) {
    if (_buffers.empty()) return;
    auto iter = _buffers.find(handle);
    LUISA_ASSERT(iter != _buffers.end());
    cmdbuffer->states()->dispose_after_flush(std::move(iter->second));
    _buffers.erase(iter);
}
template<typename Pack>
uint64 Visitor<Pack>::allocate(uint64 size) {
    return reinterpret_cast<uint64_t>(new Pack(device, size));
}
template<typename Pack>
void Visitor<Pack>::deallocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
}
template<typename Pack>
auto Visitor<Pack>::Create(uint64 size) -> Pack * {
    return new Pack{device, size};
}
template<typename T>
void BufferAllocator<T>::clear() {
    largeBuffers.clear();
    alloc.dispose();
}
template<typename T>
BufferAllocator<T>::BufferAllocator(size_t initCapacity)
    : alloc(initCapacity, &visitor) {
}
template<typename T>
BufferAllocator<T>::~BufferAllocator() {
}
template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}

template<typename T>
BufferView BufferAllocator<T>::allocate(size_t size, size_t align) {
    if (size <= kLargeBufferSize) [[likely]] {
        auto chunk = alloc.allocate(size, align);
        return BufferView(reinterpret_cast<T const *>(chunk.handle), chunk.offset, size);
    } else {
        auto &v = largeBuffers.emplace_back(visitor.Create(size));
        return BufferView(v.get(), 0, size);
    }
}
}// namespace temp_buffer

static size_t TEMP_SIZE = 1024ull * 1024ull;
CommandBufferState::CommandBufferState()
    : upload_alloc(TEMP_SIZE),
      readback_alloc(TEMP_SIZE) {
}
void CommandBufferState::init(Device &device, StreamTag tag) {
    this->device = &device;
    upload_alloc.visitor.device = &device;
    readback_alloc.visitor.device = &device;
    {
        VkDescriptorPoolSize pool_sizes[3];
        pool_sizes[0].descriptorCount = 65536;
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[1].descriptorCount = 65536;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pool_sizes[2].descriptorCount = 65536;
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 262144,
            .poolSizeCount = vstd::array_count(pool_sizes),
            .pPoolSizes = pool_sizes};
        VK_CHECK_RESULT(vkCreateDescriptorPool(device.logic_device(), &createInfo, Device::alloc_callbacks(), &_desc_pool));
    }
    if (!_pool) {
        VkCommandPoolCreateInfo pool_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
        switch (tag) {
            case StreamTag::GRAPHICS:
                pool_ci.queueFamilyIndex = device.graphics_queue_index();
                break;
            case StreamTag::COPY:
                pool_ci.queueFamilyIndex = device.copy_queue_index();
                break;
            case StreamTag::COMPUTE:
                pool_ci.queueFamilyIndex = device.compute_queue_index();
                break;
            default:
                LUISA_ERROR("Illegal stream tag.");
        }
        VK_CHECK_RESULT(vkCreateCommandPool(device.logic_device(), &pool_ci, Device::alloc_callbacks(), &_pool));
    }
}
CommandBufferState::~CommandBufferState() {
    vkDestroyCommandPool(device->logic_device(), _pool, Device::alloc_callbacks());
    vkDestroyDescriptorPool(device->logic_device(), _desc_pool, Device::alloc_callbacks());
}
void CommandBufferState::reset(Stream *stream, Device &device) {
    for (auto &i : _callbacks) {
        i();
    }
    _callbacks.clear();
    for (auto &i : _dispose_pool) {
        i.second(stream, this, i.first);
    }
    _dispose_pool.clear();
    upload_alloc.clear();
    readback_alloc.clear();
    for (auto i : img_views) {
        vkDestroyImageView(device.logic_device(), i, Device::alloc_callbacks());
    }
    img_views.clear();
    VK_CHECK_RESULT(vkResetDescriptorPool(device.logic_device(), _desc_pool, 0));
}
void CommandBuffer::reset() {
    VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
    _state->reset(&stream, *device());
}

Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      logger([](luisa::string_view str) {
          LUISA_INFO("[DEVICE] {}", str);
      }),
      reorder({}),
      _thd([this]() {
          auto loop_cmd = [&]() {
              while (true) {
                  _mtx.lock();
                  auto p = _exec.pop();
                  _mtx.unlock();
                  if (!p) {
                      break;
                  }
                  p->visit(
                      [&]<typename T>(T &t) {
                          if constexpr (std::is_same_v<T, Callbacks>) {
                              for (auto &i : t) {
                                  i();
                              }
                          } else if constexpr (std::is_same_v<T, SyncExt>) {
                              t.evt->host_wait(t.value);
                          } else if constexpr (std::is_same_v<T, NotifyEvt>) {
                              t.evt->notify(t.value);
                          } else if constexpr (std::is_same_v<T, CommandBuffer>) {
                              t.reset();
                              _cmdbuffers.push(std::move(t));
                          }
                      });
              }
          };
          while (_enabled) {
              loop_cmd();
              while (_enabled && _exec.length() == 0) {
                  std::this_thread::yield();
              }
          }
          loop_cmd();
      }),
      temp_desc(65536, &temp_desc_visitor, 2),
      scratch_buffer_alloc(TEMP_SIZE, &scratch_buffer_alloc_visitor),
      _stream_tag(tag) {
    switch (tag) {
        case StreamTag::GRAPHICS:
            _queue = device->graphics_queue();
            resource_barrier.queue_type = ResourceBarrier::QueueType::Graphics;
            resource_barrier.queue_index = device->graphics_queue_index();
            _queue_mtx = &device->graphics_queue_mtx();
            break;
        case StreamTag::COPY:
            resource_barrier.queue_type = ResourceBarrier::QueueType::Copy;
            _queue = device->copy_queue();
            resource_barrier.queue_index = device->copy_queue_index();
            _queue_mtx = &device->copy_queue_mtx();
            break;
        case StreamTag::COMPUTE:
            resource_barrier.queue_type = ResourceBarrier::QueueType::Compute;
            _queue = device->compute_queue();
            resource_barrier.queue_index = device->compute_queue_index();
            _queue_mtx = &device->compute_queue_mtx();
            break;
        default:
            LUISA_ERROR("Illegal stream tag.");
    }
}
Stream::~Stream() {
    sync();
    {
        std::lock_guard lck{_mtx};
        _enabled = false;
    }
    _thd.join();
    scratch_buffer_alloc_visitor._buffers.clear();
    while (auto p = _cmdbuffers.pop()) {
    }
}

void Stream::present(
    Texture const *tex,
    uint mip,
    Swapchain *swapchain,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    temp_desc.clear();
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
    }
    auto fence = _evt.last_fence() + 1;
    {
        CommandBuffer cmdbuffer = [&]() {
            auto p = _cmdbuffers.pop();
            if (p) return std::move(*p);
            return CommandBuffer{*this};
        }();
        scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
        scratch_buffer_alloc_visitor.device = device();

        auto cb = cmdbuffer.cmdbuffer();

        cmdbuffer.resource_barrier = &resource_barrier;
        cmdbuffer.uniform_data = &uniform_data;
        cmdbuffer.desc_sets = &desc_sets;
        cmdbuffer.logger = logger ? &logger : nullptr;
        cmdbuffer.dispatch_offsets = &dispatch_offsets;
        cmdbuffer.write_desc_sets = &write_desc_sets;
        cmdbuffer.bindless_cache = &bindless_cache;
        cmdbuffer.temp_desc = &temp_desc;
        cmdbuffer.scratch_buffer_alloc = &scratch_buffer_alloc;
        cmdbuffer.begin();
        PresentCommand present_cmd;
        present_cmd.submit_wait_semaphores.emplace_back();
        present_cmd.signal_semaphores.emplace_back();
        present_cmd.wait_stages.emplace_back();
        present_cmd.present_wait_semaphores.emplace_back();
        present_cmd.image_indices.emplace_back();
        swapchain->present(
            cmdbuffer,
            present_cmd.submit_wait_semaphores.back(), present_cmd.signal_semaphores.back(),
            present_cmd.wait_stages.back(),
            present_cmd.present_wait_semaphores.back(),
            present_cmd.image_indices.back(),
            tex,
            mip);

        resource_barrier.restore_states(cmdbuffer.cmdbuffer());
        cmdbuffer.end();

        {
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.waitSemaphoreCount = present_cmd.submit_wait_semaphores.size();
            submit_info.pWaitSemaphores = present_cmd.submit_wait_semaphores.data();
            submit_info.pWaitDstStageMask = present_cmd.wait_stages.data();
            submit_info.signalSemaphoreCount = present_cmd.signal_semaphores.size();
            submit_info.pSignalSemaphores = present_cmd.signal_semaphores.data();
            submit_info.commandBufferCount = 1;
            auto _cmdbuffer = cmdbuffer.cmdbuffer();
            submit_info.pCommandBuffers = &_cmdbuffer;
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueueSubmit(_queue, 1u, &submit_info, nullptr));
            _queue_mtx->unlock();
        }
        {
            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            auto swp_ptr = swapchain->swapchain();
            present_info.pSwapchains = &swp_ptr;
            present_info.swapchainCount = 1u;
            present_info.waitSemaphoreCount = present_cmd.present_wait_semaphores.size();
            present_info.pWaitSemaphores = present_cmd.present_wait_semaphores.data();
            present_info.pImageIndices = present_cmd.image_indices.data();
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueuePresentKHR(_queue, &present_info));
            _queue_mtx->unlock();
        }
        _evt.signal(*this, fence);
        _mtx.lock();
        _exec.push(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.push(std::move(cmdbuffer));
        _exec.push(NotifyEvt{
            .evt = &_evt,
            .value = fence});

        _mtx.unlock();
    }
}
void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    vstd::span<const SwapchainPresent> presents,
    bool inqueue_limit) {
    std::lock_guard lck{_dispatch_mtx};
    PresentCommand present_cmd;
    luisa::fixed_vector<VkSwapchainKHR, 1> vk_swapchains;
    temp_desc.clear();
    if (cmds.empty() && callbacks.empty()) {
        return;
    }
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
    }
    auto fence = _evt.last_fence() + 1;
    if (!cmds.empty()) {
        CommandBuffer cmdbuffer = [&]() {
            auto p = _cmdbuffers.pop();
            if (p) return std::move(*p);
            return CommandBuffer{*this};
        }();
        scratch_buffer_alloc_visitor.cmdbuffer = &cmdbuffer;
        scratch_buffer_alloc_visitor.device = device();

        auto cb = cmdbuffer.cmdbuffer();
        auto cb_ptr = &cb;
        resource_barrier.restoreStates.clear();
        if (device()->config_ext()) {
            auto after_states = device()->config_ext()->after_states(reinterpret_cast<uint64_t>(this));
            auto before_states = device()->config_ext()->before_states(reinterpret_cast<uint64_t>(this));
            for (auto &i : before_states) {
                resource_barrier.set_res(get_resource_view(i.resource), i.stage, i.access, i.texture_layout);
            }
            for (auto &i : after_states) {
                resource_barrier.restoreStates.emplace(
                    reinterpret_cast<Resource const *>(luisa::visit([](auto &&t) { return t.handle; }, i.resource)),
                    ResourceBarrier::ResotreStates{
                        get_resource_view(i.resource),
                        i.stage,
                        i.access,
                        i.texture_layout});
            }
        }
        cmdbuffer.resource_barrier = &resource_barrier;
        cmdbuffer.uniform_data = &uniform_data;
        cmdbuffer.desc_sets = &desc_sets;
        cmdbuffer.logger = logger ? &logger : nullptr;
        cmdbuffer.dispatch_offsets = &dispatch_offsets;
        cmdbuffer.write_desc_sets = &write_desc_sets;
        cmdbuffer.bindless_cache = &bindless_cache;
        cmdbuffer.temp_desc = &temp_desc;
        cmdbuffer.scratch_buffer_alloc = &scratch_buffer_alloc;
        cmdbuffer.begin();
        cmdbuffer.execute(cmds);
        for (auto &i : presents) {
            auto swapchain = reinterpret_cast<lc::vk::Swapchain *>(i.chain->handle());
            auto tex = reinterpret_cast<Texture *>(i.frame.handle());
            auto mip = i.frame.level();

            present_cmd.submit_wait_semaphores.emplace_back();
            present_cmd.signal_semaphores.emplace_back();
            present_cmd.wait_stages.emplace_back();
            present_cmd.present_wait_semaphores.emplace_back();
            present_cmd.image_indices.emplace_back();
            vk_swapchains.emplace_back(swapchain->swapchain());

            swapchain->present(
                cmdbuffer,
                present_cmd.submit_wait_semaphores.back(), present_cmd.signal_semaphores.back(),
                present_cmd.wait_stages.back(),
                present_cmd.present_wait_semaphores.back(),
                present_cmd.image_indices.back(),
                tex,
                mip);
        }
        resource_barrier.restore_states(cmdbuffer.cmdbuffer());
        cmdbuffer.end();
        if (!presents.empty()) {
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.waitSemaphoreCount = present_cmd.submit_wait_semaphores.size();
            submit_info.pWaitSemaphores = present_cmd.submit_wait_semaphores.data();
            submit_info.pWaitDstStageMask = present_cmd.wait_stages.data();
            submit_info.signalSemaphoreCount = present_cmd.signal_semaphores.size();
            submit_info.pSignalSemaphores = present_cmd.signal_semaphores.data();
            submit_info.commandBufferCount = 1;
            if (device()->config_ext() && device()->config_ext()->execute_command_buffer(cb)) {
                cb_ptr = nullptr;
            }
            submit_info.pCommandBuffers = cb_ptr;
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueueSubmit(_queue, 1u, &submit_info, nullptr));
            _queue_mtx->unlock();

            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present_info.pSwapchains = vk_swapchains.data();
            present_info.swapchainCount = vk_swapchains.size();
            present_info.waitSemaphoreCount = present_cmd.present_wait_semaphores.size();
            present_info.pWaitSemaphores = present_cmd.present_wait_semaphores.data();
            present_info.pImageIndices = present_cmd.image_indices.data();
            _queue_mtx->lock();
            VK_CHECK_RESULT(vkQueuePresentKHR(_queue, &present_info));
            _queue_mtx->unlock();
        }
        if (cb_ptr && device()->config_ext() && device()->config_ext()->execute_command_buffer(cb)) {
            cb_ptr = nullptr;
        }
        _evt.signal(*this, fence, cb_ptr);
        _mtx.lock();
        _exec.push(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.push(std::move(cmdbuffer));
    } else {
        _evt.update_fence(fence);
        _mtx.lock();
    }
    if (!callbacks.empty()) {
        _exec.push(std::move(callbacks));
    }
    _exec.push(NotifyEvt{
        .evt = &_evt,
        .value = fence});

    _mtx.unlock();
}
void Stream::update_sparse_resources(luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    std::lock_guard lck{_dispatch_mtx};
    temp_desc.clear();
    if (textures_update.empty()) [[unlikely]]
        return;
    VkBindSparseInfo info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    };
    auto fence = _evt.last_fence() + 1;
    struct Alloc {
        size_t size{};
        void *ptr{};
        bool is_buffer{};
    };
    vstd::unordered_map<uint64_t, Alloc> counter;
    size_t buffer_bind_count = 0;
    size_t img_bind_count = 0;
    for (auto &i : textures_update) {
        auto iter = counter.try_emplace(i.handle, 0);
        auto &v = iter.first->second;
        v.size += 1;
        luisa::visit(
            [&]<typename T>(T const &op) {
                if constexpr (std::is_same_v<SparseTextureMapOperation, T> || std::is_same_v<SparseTextureUnMapOperation, T>) {
                    if (iter.second)
                        img_bind_count += 1;
                    v.is_buffer = false;
                } else {
                    if (iter.second)
                        buffer_bind_count += 1;
                    v.is_buffer = true;
                }
            },
            i.operations);
    }
    auto buffer_ptr_chunk = temp_desc.allocate(sizeof(VkSparseBufferMemoryBindInfo) * buffer_bind_count, alignof(VkSparseBufferMemoryBindInfo));
    auto img_ptr_chunk = temp_desc.allocate(sizeof(VkSparseImageMemoryBindInfo) * img_bind_count, alignof(VkSparseImageMemoryBindInfo));
    auto buffer_ptr = reinterpret_cast<VkSparseBufferMemoryBindInfo *>(buffer_ptr_chunk.handle + buffer_ptr_chunk.offset);
    auto img_ptr = reinterpret_cast<VkSparseImageMemoryBindInfo *>(img_ptr_chunk.handle + img_ptr_chunk.offset);
    info.pBufferBinds = buffer_ptr;
    info.pImageBinds = img_ptr;
    info.bufferBindCount = buffer_bind_count;
    info.imageBindCount = img_bind_count;
    // Bind ptr
    for (auto &i : counter) {
        auto &a = i.second;
        if (a.is_buffer) {
            auto chunk = temp_desc.allocate(sizeof(VkSparseMemoryBind) * a.size, alignof(VkSparseMemoryBind));
            auto ptr = reinterpret_cast<VkSparseMemoryBind *>(chunk.handle + chunk.offset);
            a.ptr = ptr;
            buffer_ptr->buffer = reinterpret_cast<SparseBuffer *>(i.first)->vk_buffer();
            buffer_ptr->bindCount = a.size;
            buffer_ptr->pBinds = ptr;
            ++buffer_ptr;
        } else {
            auto chunk = temp_desc.allocate(sizeof(VkSparseImageMemoryBind) * a.size, alignof(VkSparseImageMemoryBind));
            auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(chunk.handle + chunk.offset);
            a.ptr = ptr;
            img_ptr->image = reinterpret_cast<Texture *>(i.first)->vk_image();
            img_ptr->bindCount = a.size;
            img_ptr->pBinds = ptr;
            ++img_ptr;
        }
    }
    // Write value
    for (auto &i : textures_update) {
        auto &v = counter.try_emplace(i.handle, 0).first->second;
        luisa::visit([&]<typename T>(T const &op) {
            if constexpr (std::is_same_v<SparseTextureMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(v.ptr);
                auto heap = reinterpret_cast<std::pair<VmaAllocation, VmaAllocationInfo> *>(op.allocated_heap);
                auto tex = reinterpret_cast<Texture const *>(i.handle);
                ptr->subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                ptr->subresource.mipLevel = op.mip_level;
                ptr->subresource.arrayLayer = 0;
                auto tile_size = tex->tile_size();
                auto start_size = op.start_tile * tile_size;
                auto extent = op.tile_count * tile_size;
                ptr->offset = VkOffset3D{(int)start_size.x, (int)start_size.y, (int)start_size.z};
                ptr->extent = VkExtent3D{extent.x, extent.y, extent.z};
                ptr->memory = heap->second.deviceMemory;
                ptr->memoryOffset = heap->second.offset;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseTextureUnMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseImageMemoryBind *>(v.ptr);
                auto tex = reinterpret_cast<Texture const *>(i.handle);
                ptr->subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                ptr->subresource.mipLevel = op.mip_level;
                ptr->subresource.arrayLayer = 0;
                auto tile_size = tex->tile_size();
                auto start_size = op.start_tile * tile_size;
                auto extent = op.tile_count * tile_size;
                ptr->offset = VkOffset3D{(int)start_size.x, (int)start_size.y, (int)start_size.z};
                ptr->extent = VkExtent3D{extent.x, extent.y, extent.z};
                ptr->memory = VK_NULL_HANDLE;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseBufferMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseMemoryBind *>(v.ptr);
                auto heap = reinterpret_cast<std::pair<VmaAllocation, VmaAllocationInfo> *>(op.allocated_heap);
                ptr->memory = heap->second.deviceMemory;
                ptr->memoryOffset = heap->second.offset;
                ptr->resourceOffset = op.start_tile * sparse_buffer_size;
                ptr->size = sparse_buffer_size * op.tile_count;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            } else if constexpr (std::is_same_v<SparseBufferUnMapOperation, T>) {
                auto ptr = reinterpret_cast<VkSparseMemoryBind *>(v.ptr);
                ptr->memory = VK_NULL_HANDLE;
                ptr->resourceOffset = op.start_tile * sparse_buffer_size;
                ptr->size = sparse_buffer_size * op.tile_count;
                ptr->flags = 0;
                ++ptr;
                v.ptr = ptr;
            }
        },
                     i.operations);
    }
    VkTimelineSemaphoreSubmitInfo timeline;
    _evt.signal_sparse(*this, &fence, &info, &timeline);
    _queue_mtx->lock();
    VK_CHECK_RESULT(vkQueueBindSparse(
        _queue,
        1,
        &info,
        VK_NULL_HANDLE));
    _queue_mtx->unlock();
    _evt.mark_signal_fence(fence);
    _mtx.lock();
    _exec.push(NotifyEvt{
        .evt = &_evt,
        .value = fence});
    _mtx.unlock();
}
void Stream::sync() {
    _evt.sync(_evt.last_fence());
}
CommandBuffer::CommandBuffer(Stream &stream) noexcept
    : Resource(stream.device()),
      stream(stream),
      _state(vstd::make_unique<CommandBufferState>()) {
    _state->init(*stream.device(), stream.stream_tag());
    _cmdbuffer = nullptr;
    if (device()->config_ext()) {
        _cmdbuffer = device()->config_ext()->borrow_command_buffer(stream.stream_tag());
    }
    if (!_cmdbuffer) {
        VkCommandBufferAllocateInfo cb_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = _state->_pool,
            .commandBufferCount = 1};
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device()->logic_device(), &cb_ci, &_cmdbuffer));
    }
    // VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    // VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), nullptr));
}
CommandBuffer::~CommandBuffer() {
    if (_cmdbuffer)
        vkFreeCommandBuffers(device()->logic_device(), _state->_pool, 1, &_cmdbuffer);
}
void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdbuffer, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmdbuffer));
}
CommandBuffer::CommandBuffer(CommandBuffer &&rhs) noexcept
    : Resource(std::move(rhs)),
      stream(rhs.stream),
      _cmdbuffer(rhs._cmdbuffer),
      _state(std::move(rhs._state)) {
    rhs._cmdbuffer = nullptr;
}
void Stream::signal(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    event->signal(*this, value);
    _mtx.lock();
    _exec.push(SyncExt{event, value});
    _exec.push(NotifyEvt{event, value});
    _mtx.unlock();
}
void Stream::wait(Event *event, uint64_t value) {
    std::lock_guard lck{_dispatch_mtx};
    event->wait(*this, value);
}
void CommandBuffer::execute(vstd::span<const luisa::unique_ptr<Command>> cmds) {
    // collect argument buffer
    size_t uniform_buffer_size = 0;
    auto add_size = [&](ShaderDispatchCommandBase const &c, Argument const &a) {
        if (a.tag != Argument::Tag::UNIFORM) [[likely]]
            return;

        auto bf = c.uniform(a.uniform);
        auto aligned_size = a.uniform.alignment - 1;
        uniform_buffer_size = (uniform_buffer_size + aligned_size) & (~(aligned_size));
        uniform_buffer_size += std::max<size_t>(4, bf.size_bytes());
    };
    auto dispatch_shader = [&](ShaderDispatchCommandBase const *c, Shader const *shader) {
        uniform_buffer_size = (uniform_buffer_size + 31) & (~(31ull));
        for (auto &i : shader->captured()) {
            add_size(*c, i);
        }
        for (auto &i : c->arguments()) {
            add_size(*c, i);
        }
    };
    for (auto &&command : cmds) {
        command->accept(stream.reorder);
        switch (command->tag()) {
            case Command::Tag::EShaderDispatchCommand: {
                auto c = static_cast<ShaderDispatchCommand const *>(command.get());
                auto shader = reinterpret_cast<Shader const *>(c->handle());
                dispatch_shader(c, shader);
            } break;
            case Command::Tag::ECustomCommand: {
                auto cmd = static_cast<CustomCommand const *>(command.get());
                if (cmd->custom_cmd_uuid() == to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE)) {
                    auto c = static_cast<DrawRasterSceneCommand const *>(cmd);
                    auto shader = reinterpret_cast<Shader const *>(c->handle());
                    dispatch_shader(c, shader);
                }
            } break;
            default: break;
        }
    }
    auto cmd_lists = stream.reorder.command_lists();
    auto clear_reorder = vstd::scope_exit([&] {
        stream.reorder.clear();
    });
    uniform_data->clear();
    uniform_data->reserve(uniform_buffer_size);
#ifndef NDEBUG
    auto check_uniform = vstd::scope_exit([&]() {
        auto aligned_size = (luisa::size_bytes(*uniform_data) + 31ull) & (~31ull);
        auto origin = uniform_buffer_size;
        uniform_buffer_size = (uniform_buffer_size + 31ull) & (~(31ull));

        if (aligned_size != uniform_buffer_size) [[unlikely]] {
            LUISA_ERROR("Bad uniform size {} {} {} {}.", aligned_size, uniform_buffer_size, luisa::size_bytes(*uniform_data), origin);
        }
    });
#endif
    BufferView arg_buffer;
    if (uniform_buffer_size > 0) {
        arg_buffer = _state->upload_alloc.allocate(uniform_buffer_size, 256);
    }
    auto preprocess_arguments = [this](Shader const *shader, ShaderDispatchCommandBase const *c, bool is_raster) {
        luisa::vector_resize(*uniform_data, (uniform_data->size() + 31) & (~(31ull)));
        std::pair<size_t, size_t> sizes;
        sizes.first = luisa::size_bytes(*uniform_data);
        ResourceBarrierVisitor visitor{
            resource_barrier,
            shader->saved_arguments().data(),
            uniform_data,
            *c,
            is_raster};
        DecodeCmd(shader->captured(), visitor);
        DecodeCmd(c->arguments(), visitor);
        sizes.second = uniform_data->size() - sizes.first;
        dispatch_offsets->emplace_back(sizes);
    };
    for (auto &&lst : cmd_lists) {
        dispatch_offsets->clear();
        auto delay_clear = vstd::scope_exit([&]() {
            scratch_buffer_alloc->clear();
        });
        // Preprocess: record resources' states
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->handle()),
                            c->offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->dst_handle()),
                            c->dst_offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->src_handle()),
                            c->src_offset(),
                            c->size()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::CopySource);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    auto shader = reinterpret_cast<Shader const *>(c->handle());
                    preprocess_arguments(shader, c, false);
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->handle()),
                            c->level()},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->src_handle()),
                            c->src_level()),
                        ResourceBarrier::Usage::CopySource);
                    resource_barrier->record(
                        TexView(
                            reinterpret_cast<Texture const *>(c->dst_handle()),
                            c->dst_level()),
                        ResourceBarrier::Usage::CopyDest);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    resource_barrier->record(
                        TexView{
                            reinterpret_cast<Texture const *>(c->texture()),
                            c->level()},
                        ResourceBarrier::Usage::CopyDest);
                    resource_barrier->record(
                        BufferView{
                            reinterpret_cast<Buffer const *>(c->buffer()),
                            c->buffer_offset(),
                            pixel_storage_size(c->storage(), c->size())},
                        ResourceBarrier::Usage::CopySource);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->pre_build(*this, c->instance_count(), *write_desc_sets, *bindless_cache, c->modifications(), c->request());
                } break;
                case Command::Tag::EMeshBuildCommand: {
                    auto c = static_cast<MeshBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                    auto c = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->pre_build(*this, c);
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    reinterpret_cast<BindlessArray *>(c->handle())->pre_update(resource_barrier);
                } break;
                case Command::Tag::ECustomCommand: {
                    auto c = static_cast<CustomCommand const *>(cmd);
                    switch (c->custom_cmd_uuid()) {
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                            auto cmd = static_cast<ClearDepthCommand const *>(c);
                            auto tex = reinterpret_cast<Texture const *>(cmd->handle());
                            resource_barrier->record(
                                TexView(tex, 0),
                                ResourceBarrier::Usage::DepthClear);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET): {
                            auto cmd = static_cast<ClearRenderTargetCommand const *>(c);
                            auto tex = reinterpret_cast<Texture const *>(cmd->handle());
                            resource_barrier->record(
                                TexView(tex, cmd->level()),
                                ResourceBarrier::Usage::RenderTargetClear);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                            auto cmd = static_cast<DrawRasterSceneCommand const *>(c);
                            auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
                            preprocess_arguments(shader, cmd, true);
                            for (auto &i : cmd->rtv_texs()) {
                                auto tex = reinterpret_cast<Texture const *>(i.handle);
                                resource_barrier->record(
                                    TexView(tex, i.level),
                                    ResourceBarrier::Usage::RenderTarget);
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto tex = reinterpret_cast<Texture const *>(cmd->dsv_tex().handle);
                                resource_barrier->record(
                                    TexView(tex, 0),
                                    ResourceBarrier::Usage::DepthWrite);
                            }
                        } break;
                        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                            auto custom_cmd = static_cast<VKCustomCmd const *>(c);
                            for (auto &&i : const_cast<VKCustomCmd *>(custom_cmd)->get_resource_usages()) {
                                resource_barrier->record(get_resource_view(i.resource), i.stage, i.access, i.texture_layout);
                            }
                        } break;
                        //TODO: other commands
                        default: {
                            LUISA_ERROR("Command type not supported.");
                        } break;
                    }
                } break;
                default: break;
            }
        }
        resource_barrier->update_states(_cmdbuffer);
        auto offset_ptr = dispatch_offsets->data();
        auto set_dispatch_args = [&](BindPropVisitor &visitor, ShaderDispatchCommandBase const *c, Shader const *shader) {
            uint desc_index = 0;
            visitor.cmdbuffer = this;
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = _state->_desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = shader->desc_set_layout().data()};
            VK_CHECK_RESULT(
                vkAllocateDescriptorSets(
                    device()->logic_device(),
                    &alloc_info,
                    &visitor.desc_set));
            if (offset_ptr->second > 0) {
                auto arg_buffer_info = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                *arg_buffer_info = VkDescriptorBufferInfo{
                    arg_buffer.buffer->vk_buffer(),
                    arg_buffer.offset + offset_ptr->first,
                    offset_ptr->second};
                write_desc_sets->emplace_back(VkWriteDescriptorSet{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    nullptr,
                    visitor.desc_set,
                    desc_index++,
                    0,
                    1,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    nullptr,
                    arg_buffer_info,
                    nullptr});
            }
            offset_ptr++;
            visitor.desc_index = desc_index;
            visitor.img_views = &_state->img_views;
            auto size = shader->saved_arguments().size();

            visitor.arg = shader->saved_arguments().data();
            DecodeCmd(shader->captured(), visitor);
            DecodeCmd(c->arguments(), visitor);
            return visitor.desc_index;
        };
        auto bind_shader_desc = [&](BindPropVisitor &visitor, Shader const *shader, VkPipelineBindPoint bind_point) {
            if (!write_desc_sets->empty()) {
                vkUpdateDescriptorSets(
                    device()->logic_device(),
                    write_desc_sets->size(),
                    write_desc_sets->data(), 0,
                    nullptr);
                write_desc_sets->clear();
            }
            desc_sets->clear();
            desc_sets->push_back(visitor.desc_set);
            desc_sets->push_back(device()->sampler_set());
            if (shader->use_buffer_bindless()) {
                desc_sets->push_back(device()->bdls_buffer_set());
            }
            if (shader->use_tex2d_bindless()) {
                desc_sets->push_back(device()->bdls_tex2d_set());
            }
            if (shader->use_tex3d_bindless()) {
                desc_sets->push_back(device()->bdls_tex3d_set());
            }
            vkCmdBindDescriptorSets(
                _cmdbuffer,
                bind_point,
                shader->pipeline_layout(),
                0,
                desc_sets->size(),
                desc_sets->data(),
                0,
                nullptr);
        };
        // Execute
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {
                    auto c = static_cast<BufferUploadCommand const *>(cmd);
                    auto chunk = _state->upload_alloc.allocate(c->size(), 16);
                    static_cast<UploadBuffer const *>(chunk.buffer)->copy_from(c->data(), chunk.offset, c->size());
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        chunk.offset,
                        c->offset(),
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        chunk.buffer->vk_buffer(),
                        reinterpret_cast<Buffer const *>(c->handle())->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferDownloadCommand: {
                    auto c = static_cast<BufferDownloadCommand const *>(cmd);
                    auto chunk = _state->readback_alloc.allocate(c->size(), 16);
                    _state->_callbacks.emplace_back([chunk, data = c->data(), size = c->size()]() {
                        static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(data, chunk.offset, size);
                    });
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        c->offset(),
                        chunk.offset,
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        reinterpret_cast<Buffer const *>(c->handle())->vk_buffer(),
                        chunk.buffer->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferCopyCommand: {
                    auto c = static_cast<BufferCopyCommand const *>(cmd);
                    VkBufferCopy2 buffer_copy{
                        VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                        nullptr,
                        c->src_offset(),
                        c->dst_offset(),
                        c->size()};
                    VkCopyBufferInfo2 copy_info2{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                        nullptr,
                        reinterpret_cast<Buffer const *>(c->src_handle())->vk_buffer(),
                        reinterpret_cast<Buffer const *>(c->dst_handle())->vk_buffer(),
                        1,
                        &buffer_copy};
                    vkCmdCopyBuffer2(
                        _cmdbuffer,
                        &copy_info2);
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                    auto c = static_cast<BufferToTextureCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        reinterpret_cast<Buffer const *>(c->buffer())->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                    auto c = static_cast<ShaderDispatchCommand const *>(cmd);
                    auto shader = reinterpret_cast<ComputeShader *>(c->handle());
                    BindPropVisitor visitor{};
                    set_dispatch_args(visitor, c, shader);
                    constexpr size_t max_printer_count = 1024ull * 1024ull;
                    BufferView count_buffer;
                    BufferView data_buffer;
                    if (!shader->printers().empty()) {
                        auto count_chunk = scratch_buffer_alloc->allocate(4, 16);
                        auto data_chunk = scratch_buffer_alloc->allocate(max_printer_count, 16);
                        count_buffer = BufferView(
                            reinterpret_cast<Buffer const *>(count_chunk.handle),
                            count_chunk.offset,
                            4);
                        auto upload_buffer = states()->upload_alloc.allocate(4);
                        uint zero = 0;
                        static_cast<UploadBuffer const *>(upload_buffer.buffer)->copy_from(&zero, upload_buffer.offset, 4);

                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::CopyDest);
                        resource_barrier->update_states(_cmdbuffer);
                        VkBufferCopy2 buffer_copy{
                            VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                            nullptr,
                            upload_buffer.offset,
                            count_buffer.offset,
                            4};
                        VkCopyBufferInfo2 copy_info2{
                            VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                            nullptr,
                            upload_buffer.buffer->vk_buffer(),
                            count_buffer.buffer->vk_buffer(),
                            1,
                            &buffer_copy};
                        vkCmdCopyBuffer2(
                            _cmdbuffer,
                            &copy_info2);

                        data_buffer = BufferView{
                            reinterpret_cast<Buffer const *>(data_chunk.handle),
                            data_chunk.offset,
                            max_printer_count};
                        // bind counter
                        {
                            auto idx = visitor.desc_index++;
                            auto buffer_descs = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                            *buffer_descs = VkDescriptorBufferInfo{
                                count_buffer.buffer->vk_buffer(),
                                count_buffer.offset,
                                4};
                            auto &a = write_desc_sets->emplace_back(VkWriteDescriptorSet{
                                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                nullptr,
                                visitor.desc_set,
                                idx,
                                0,
                                1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                nullptr,
                                buffer_descs,
                                nullptr});
                        }
                        // bind data
                        {
                            auto idx = visitor.desc_index++;
                            auto buffer_descs = temp_desc->allocate_memory<VkDescriptorBufferInfo>();
                            *buffer_descs = VkDescriptorBufferInfo{
                                data_buffer.buffer->vk_buffer(),
                                data_buffer.offset,
                                max_printer_count};
                            auto &a = write_desc_sets->emplace_back(VkWriteDescriptorSet{
                                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                nullptr,
                                visitor.desc_set,
                                idx,
                                0,
                                1,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                nullptr,
                                buffer_descs,
                                nullptr});
                        }
                        resource_barrier->record(
                            data_buffer,
                            ResourceBarrier::Usage::ComputeUAV);
                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::ComputeUAV);
                        resource_barrier->update_states(_cmdbuffer);
                    }
                    bind_shader_desc(visitor, shader, VK_PIPELINE_BIND_POINT_COMPUTE);
                    vkCmdBindPipeline(_cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader->pipeline());
                    auto calc = [](uint disp, uint thd) {
                        return (disp + thd - 1u) / thd;
                    };
                    auto blk = shader->block_size();
                    if (c->is_indirect()) {
                        LUISA_ERROR("Indirect not implemented.");
                    } else if (c->is_multiple_dispatch()) {
                        uint idx = 0;
                        for (auto &disp_size : c->dispatch_sizes()) {
                            uint4 value{make_uint4(disp_size, idx)};
                            ++idx;
                            vkCmdPushConstants(
                                _cmdbuffer,
                                shader->pipeline_layout(),
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                0,
                                16,
                                &value);
                            vkCmdDispatch(_cmdbuffer, calc(disp_size.x, blk.x), calc(disp_size.y, blk.y), calc(disp_size.z, blk.z));
                        }
                    } else {
                        auto disp_size = c->dispatch_size();
                        uint4 value{make_uint4(disp_size, 0)};
                        vkCmdPushConstants(
                            _cmdbuffer,
                            shader->pipeline_layout(),
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            16,
                            &value);
                        vkCmdDispatch(_cmdbuffer, calc(disp_size.x, blk.x), calc(disp_size.y, blk.y), calc(disp_size.z, blk.z));
                    }
                    if (logger && !shader->printers().empty()) {
                        resource_barrier->record(
                            count_buffer,
                            ResourceBarrier::Usage::CopySource);
                        resource_barrier->record(
                            data_buffer,
                            ResourceBarrier::Usage::CopySource);
                        resource_barrier->update_states(_cmdbuffer);
                        auto counter_readback = states()->readback_alloc.allocate(4, 16);
                        auto data_readback = states()->readback_alloc.allocate(max_printer_count, 16);
                        // Copy counter
                        {
                            VkBufferCopy2 buffer_copy{
                                VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                nullptr,
                                count_buffer.offset,
                                counter_readback.offset,
                                4};
                            VkCopyBufferInfo2 copy_info2{
                                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                nullptr,
                                count_buffer.buffer->vk_buffer(),
                                reinterpret_cast<Buffer const *>(counter_readback.buffer)->vk_buffer(),
                                1,
                                &buffer_copy};

                            vkCmdCopyBuffer2(
                                _cmdbuffer,
                                &copy_info2);
                        }
                        // Copy data
                        {
                            VkBufferCopy2 buffer_copy{
                                VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                                nullptr,
                                data_buffer.offset,
                                data_readback.offset,
                                max_printer_count};
                            VkCopyBufferInfo2 copy_info2{
                                VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                                nullptr,
                                data_buffer.buffer->vk_buffer(),
                                reinterpret_cast<Buffer const *>(data_readback.buffer)->vk_buffer(),
                                1,
                                &buffer_copy};
                            vkCmdCopyBuffer2(
                                _cmdbuffer,
                                &copy_info2);
                        }
                        states()->_callbacks.emplace_back(
                            [printers = shader->printers(),
                             logger = this->logger,
                             counter_readback,
                             data_readback]() {
                                uint counter = 0;
                                static_cast<ReadbackBuffer const *>(counter_readback.buffer)->copy_to(&counter, counter_readback.offset, 4);
                                luisa::vector<std::byte> data;
                                luisa::enlarge_by(data, counter);
                                static_cast<ReadbackBuffer const *>(data_readback.buffer)->copy_to(data.data(), data_readback.offset, counter);
                                size_t offset = 0;
                                const auto ptr = data.data();
                                const auto end = data.size();
                                while (offset < end) {
                                    uint flagTypeIdx = *reinterpret_cast<uint32_t *>(ptr + offset);
                                    auto &type = printers[flagTypeIdx];
                                    ShaderPrintFormatter formatter{type.first, type.second, false};
                                    luisa::string result;
                                    auto align = std::max<size_t>(4, type.second->alignment());
                                    formatter(result, {ptr + offset + align, type.second->size()});
                                    size_t ele_size = align + type.second->size();
                                    ele_size = ((ele_size + 15ull) & (~15ull));
                                    offset += ele_size;
                                    (*logger)(result);
                                }
                            });
                    }
                } break;
                case Command::Tag::ETextureUploadCommand: {
                    auto c = static_cast<TextureUploadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->upload_alloc.allocate(pixel_size, 16);
                    static_cast<UploadBuffer const *>(buffer.buffer)->copy_from(c->data(), buffer.offset, pixel_size);
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};

                    VkCopyBufferToImageInfo2 copy_info{
                        VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
                        nullptr,
                        buffer.buffer->vk_buffer(),
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        1,
                        &region};
                    vkCmdCopyBufferToImage2(_cmdbuffer, &copy_info);
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                    auto c = static_cast<TextureDownloadCommand const *>(cmd);
                    auto pixel_size = pixel_storage_size(c->storage(), c->size());
                    auto buffer = _state->readback_alloc.allocate(pixel_size, 16);
                    _state->_callbacks.emplace_back([buffer = buffer.buffer,
                                                     offset = buffer.offset,
                                                     pixel_size,
                                                     data = c->data()]() {
                        static_cast<ReadbackBuffer const *>(buffer)->copy_to(data, offset, pixel_size);
                    });
                    auto tex = reinterpret_cast<Texture const *>(c->handle());
                    int3 tex_offset = make_int3(c->offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        buffer.offset,
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        buffer.buffer->vk_buffer(),
                        1,
                        &region};
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::ETextureCopyCommand: {
                    auto c = static_cast<TextureCopyCommand const *>(cmd);
                    auto src_tex = reinterpret_cast<Texture const *>(c->src_handle());
                    int3 src_tex_offset = make_int3(c->src_offset());
                    int3 dst_tex_offset = make_int3(c->dst_offset());
                    auto dst_tex = reinterpret_cast<Texture const *>(c->dst_handle());
                    auto size = c->size();
                    VkImageCopy2 copy{
                        VK_STRUCTURE_TYPE_IMAGE_COPY_2,
                        nullptr,
                        VkImageSubresourceLayers{src_tex->get_aspect(), c->src_level(), 0, 1},
                        VkOffset3D{src_tex_offset.x, src_tex_offset.y, src_tex_offset.z},
                        VkImageSubresourceLayers{dst_tex->get_aspect(), c->dst_level(), 0, 1},
                        VkOffset3D{dst_tex_offset.x, dst_tex_offset.y, dst_tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
                        nullptr,
                        src_tex->vk_image(),
                        resource_barrier->get_layout(src_tex, c->src_level()),
                        dst_tex->vk_image(),
                        resource_barrier->get_layout(dst_tex, c->src_level()),
                        1,
                        &copy};
                    vkCmdCopyImage2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                    auto c = static_cast<TextureToBufferCopyCommand const *>(cmd);
                    auto tex = reinterpret_cast<Texture const *>(c->texture());
                    int3 tex_offset = make_int3(c->texture_offset());
                    auto size = c->size();
                    VkBufferImageCopy2 region{
                        VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                        nullptr,
                        c->buffer_offset(),
                        0,
                        0,
                        VkImageSubresourceLayers{tex->get_aspect(), c->level(), 0, 1},
                        VkOffset3D{tex_offset.x, tex_offset.y, tex_offset.z},
                        VkExtent3D{size.x, size.y, size.z}};
                    VkCopyImageToBufferInfo2 info{
                        VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
                        nullptr,
                        tex->vk_image(),
                        resource_barrier->get_layout(tex, c->level()),
                        reinterpret_cast<Buffer const *>(c->buffer())->vk_buffer(),
                        1,
                        &region};
                    vkCmdCopyImageToBuffer2(
                        _cmdbuffer,
                        &info);
                } break;
                case Command::Tag::EAccelBuildCommand: {
                    auto c = static_cast<AccelBuildCommand const *>(cmd);
                    reinterpret_cast<Tlas *>(c->handle())->build(*this, c->instance_count());
                    auto &bf = *reinterpret_cast<Tlas *>(c->handle())->instance_buffer();
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::CopySource);
                    // resource_barrier->update_states(_cmdbuffer);
                    // luisa::vector<VkAccelerationStructureInstanceKHR> vec(bf.byte_size() / sizeof(VkAccelerationStructureInstanceKHR));
                    // auto chunk = _state->readback_alloc.allocate(vec.size_bytes(), 16);

                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->_callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(
                    //             "Matrix: {}\n{}\n{}\ninstanceCustomIndex{}\nmask{}\ninstanceShaderBindingTableRecordOffset{}\nflags{}\naccelerationStructureReference{}",
                    //             (float4&)i.transform.matrix[0],
                    //             (float4&)i.transform.matrix[1],
                    //             (float4&)i.transform.matrix[2],
                    //             (uint)i.instanceCustomIndex,
                    //             (uint)i.mask,
                    //             (uint)i.instanceShaderBindingTableRecordOffset,
                    //             (uint)i.flags,
                    //             i.accelerationStructureReference
                    //         );
                    //     }
                    // });
                } break;
                case Command::Tag::EMeshBuildCommand: {
                    auto c = static_cast<MeshBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                    auto c = static_cast<ProceduralPrimitiveBuildCommand const *>(cmd);
                    reinterpret_cast<Blas *>(c->handle())->build(*this, c);
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                    auto c = static_cast<BindlessArrayUpdateCommand const *>(cmd);
                    auto bdls = reinterpret_cast<BindlessArray *>(c->handle());
                    c->visit_modifications([&](auto &&t) {
                        bdls->update(this, *write_desc_sets, *bindless_cache, luisa::span{t});
                    });
                    // LOG bindless indices

                    // auto &bf = reinterpret_cast<BindlessArray *>(c->handle())->indices_buffer();
                    // resource_barrier->record(
                    //     BufferView{&bf},
                    //     ResourceBarrier::Usage::CopySource);
                    // resource_barrier->update_states(_cmdbuffer);

                    // luisa::vector<std::array<uint, 3>> vec(3);
                    // auto chunk = _state->readback_alloc.allocate(vec.size(), 16);
                    // VkBufferCopy2 buffer_copy{
                    //     VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    //     nullptr,
                    //     0,
                    //     chunk.offset,
                    //     vec.size_bytes()};
                    // int x = 0;
                    // VkCopyBufferInfo2 copy_info2{
                    //     VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    //     nullptr,
                    //     bf.vk_buffer(),
                    //     chunk.buffer->vk_buffer(),
                    //     1,
                    //     &buffer_copy};
                    // vkCmdCopyBuffer2(
                    //     _cmdbuffer,
                    //     &copy_info2);
                    // _state->_callbacks.emplace_back([chunk, vec = std::move(vec)]() mutable {
                    //     static_cast<ReadbackBuffer const *>(chunk.buffer)->copy_to(vec.data(), chunk.offset, vec.size_bytes());
                    //     for (auto &i : vec) {
                    //         LUISA_INFO(uint3(i[0], i[1] & ((1u<<24u) - 1), i[2] & ((1u<<24u) - 1)));
                    //     }
                    // });
                } break;
                case Command::Tag::ECustomCommand: {
                    auto c = static_cast<CustomCommand const *>(cmd);
                    switch (c->custom_cmd_uuid()) {
                        // TODO
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH): {
                            auto cmd = static_cast<ClearDepthCommand const *>(c);
                            auto tex = reinterpret_cast<Texture *>(cmd->handle());
                            VkClearDepthStencilValue depth_stencil_value{
                                .depth = cmd->value(),
                                .stencil = 0};
                            auto depth_format = tex->depth_format();
                            VkImageSubresourceRange sub_range{
                                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
                            if (depth_format == DepthFormat::D24S8 || depth_format == DepthFormat::D32S8A24) {
                                sub_range.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
                            }
                            vkCmdClearDepthStencilImage(
                                _cmdbuffer,
                                tex->vk_image(),
                                resource_barrier->get_layout(tex, 0),
                                &depth_stencil_value,
                                1,
                                &sub_range);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET): {
                            auto cmd = static_cast<ClearRenderTargetCommand const *>(c);
                            auto tex = reinterpret_cast<Texture *>(cmd->handle());
                            VkClearColorValue color_value;
                            float4 values = cmd->value();
                            std::memcpy(color_value.float32, &values, sizeof(float4));
                            VkImageSubresourceRange sub_range{
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
                            vkCmdClearColorImage(
                                _cmdbuffer,
                                tex->vk_image(),
                                resource_barrier->get_layout(tex, cmd->level()),
                                &color_value,
                                1,
                                &sub_range);
                        } break;
                        case to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE): {
                            auto cmd = static_cast<DrawRasterSceneCommand const *>(c);
                            auto shader = reinterpret_cast<RasterShader *>(cmd->handle());
                            auto pipe = shader->create_pipeline(cmd->rtv_texs(), cmd->dsv_tex(), cmd->mesh_format(), cmd->raster_state());
                            BindPropVisitor visitor{};
                            // bind arguments
                            set_dispatch_args(visitor, cmd, shader);
                            bind_shader_desc(visitor, shader, VK_PIPELINE_BIND_POINT_GRAPHICS);
                            vkCmdBindPipeline(_cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline);
                            // framebuffer
                            VkFramebuffer fb;
                            uint img_view_offset = _state->img_views.size();
                            auto emplace_img_view = [&](VkImageView &img_view, Texture *tex, uint level) {
                                VkImageViewCreateInfo imgview_create_info{
                                    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                    nullptr,
                                    0,
                                    tex->vk_image(),
                                    VkImageViewType(tex->dimension() - 1),
                                    Texture::to_vk_format(tex->format()),
                                    VkComponentMapping{VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
                                    VkImageSubresourceRange{
                                        tex->get_aspect(),
                                        level,
                                        1,
                                        0,
                                        1}};
                                VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &imgview_create_info, Device::alloc_callbacks(), &img_view));
                            };
                            uint2 resolution;
                            for (auto &i : cmd->rtv_texs()) {
                                auto &img_view = _state->img_views.emplace_back();
                                auto tex = reinterpret_cast<Texture *>(i.handle);
                                emplace_img_view(img_view, tex, i.level);
                                VkClearValue clear_value;
                                std::memset(&clear_value, 0, sizeof(VkClearValue));
                            }
                            if (!cmd->rtv_texs().empty()) {
                                auto &&i = cmd->rtv_texs()[0];
                                auto tex = reinterpret_cast<Texture *>(i.handle);
                                resolution = tex->size().xy();
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto &img_view = _state->img_views.emplace_back();
                                auto tex = reinterpret_cast<Texture *>(cmd->dsv_tex().handle);
                                emplace_img_view(img_view, tex, cmd->dsv_tex().level);
                                resolution = tex->size().xy();
                                VkClearValue clear_value;
                                std::memset(&clear_value, 0, sizeof(VkClearValue));
                                clear_value.depthStencil.depth = 0.f;// TODO: may set depth_buffer default value
                            }

                            VkFramebufferCreateInfo framebuffer_create_info{
                                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                .renderPass = pipe.render_pass,
                                .attachmentCount = (uint)(_state->img_views.size() - img_view_offset),
                                .pAttachments = _state->img_views.data() + img_view_offset,
                                .width = resolution.x,
                                .height = resolution.y,
                                .layers = 1};
                            VK_CHECK_RESULT(vkCreateFramebuffer(
                                device()->logic_device(),
                                &framebuffer_create_info,
                                Device::alloc_callbacks(),
                                &fb));
                            VkRenderPassBeginInfo begin_pass_info{
                                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                .renderPass = pipe.render_pass,
                                .framebuffer = fb,
                                .renderArea = VkRect2D{
                                    .offset = VkOffset2D{0, 0},
                                    .extent = VkExtent2D{resolution.x, resolution.y}},
                                .clearValueCount = 0,
                                .pClearValues = nullptr};
                            vkCmdBeginRenderPass(_cmdbuffer, &begin_pass_info, VK_SUBPASS_CONTENTS_INLINE);
                            auto &&cmd_vp = cmd->viewport();
                            VkViewport viewport{};
                            viewport.x = static_cast<float>(cmd_vp.start.x);
                            viewport.y = static_cast<float>(cmd_vp.start.y);
                            viewport.width = std::max(1.f, static_cast<float>(cmd_vp.size.x));
                            viewport.height = std::max(1.f, static_cast<float>(cmd_vp.size.y));
                            viewport.minDepth = 0.0f;
                            viewport.maxDepth = 1.0f;
                            vkCmdSetViewport(_cmdbuffer, 0, 1, &viewport);

                            VkRect2D scissor{};
                            scissor.offset = {(int)cmd_vp.start.x, (int)cmd_vp.start.y};
                            scissor.extent = {cmd_vp.size.x, cmd_vp.size.y};
                            vkCmdSetScissor(_cmdbuffer, 0, 1, &scissor);
                            vstd::fixed_vector<VkBuffer, 4> vertex_buffers;
                            vstd::fixed_vector<VkDeviceSize, 4> vertex_buffer_offsets;
                            for (auto &mesh : cmd->scene()) {
                                auto vb = mesh.vertex_buffers();
                                vertex_buffers.clear();
                                vertex_buffer_offsets.clear();
                                vertex_buffers.reserve(vb.size());
                                vertex_buffer_offsets.reserve(vb.size());
                                for (auto &i : vb) {
                                    vertex_buffers.emplace_back(reinterpret_cast<Buffer *>(i.handle())->vk_buffer());
                                    vertex_buffer_offsets.emplace_back(i.offset());
                                }
                                vkCmdBindVertexBuffers(_cmdbuffer, 0, vb.size(), vertex_buffers.data(), vertex_buffer_offsets.data());
                                auto before_draw = [&]() {
                                    uint value = mesh.object_id();
                                    vkCmdPushConstants(
                                        _cmdbuffer,
                                        shader->pipeline_layout(),
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        0,
                                        4,
                                        &value);
                                };
                                luisa::visit([&]<typename T>(T const &t) {
                                    // Draw
                                    if constexpr (std::is_integral_v<T>) {
                                        before_draw();
                                        vkCmdDraw(
                                            _cmdbuffer,
                                            t,
                                            mesh.instance_count(),
                                            mesh.vertex_offset(),
                                            0);
                                    } else {
                                        auto buffer = reinterpret_cast<Buffer *>(t.handle());
                                        vkCmdBindIndexBuffer(_cmdbuffer, buffer->vk_buffer(), t.offset_bytes(), VK_INDEX_TYPE_UINT32);
                                        before_draw();
                                        vkCmdDrawIndexed(
                                            _cmdbuffer,
                                            t.size_bytes() / sizeof(uint),
                                            mesh.instance_count(),
                                            0,
                                            mesh.vertex_offset(),
                                            0);
                                    }
                                    // Draw indexed
                                },
                                             mesh.index());
                            }
                            vkCmdEndRenderPass(_cmdbuffer);
                            _state->_dispose_pool.emplace_back(fb, [](Stream *stream, CommandBufferState *state, void *ptr) {
                                vkDestroyFramebuffer(
                                    stream->device()->logic_device(),
                                    static_cast<VkFramebuffer>(ptr),
                                    Device::alloc_callbacks());
                            });
                            for (auto &i : cmd->rtv_texs()) {
                                auto tex = reinterpret_cast<Texture const *>(i.handle);
                                resource_barrier->force_refresh_layout(
                                    tex, i.level,
                                    VK_IMAGE_LAYOUT_GENERAL);
                            }
                            if (cmd->dsv_tex().handle != invalid_resource_handle) {
                                auto tex = reinterpret_cast<Texture const *>(cmd->dsv_tex().handle);
                                resource_barrier->force_refresh_layout(
                                    tex, 0,
                                    VK_IMAGE_LAYOUT_GENERAL);
                            }

                        } break;
                        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
                            static_cast<VKCustomCmd const *>(c)->execute(
                                device()->physical_device(),
                                device()->logic_device(),
                                stream.queue(),
                                _cmdbuffer,
                                _state->_desc_pool);
                        } break;
                        //TODO: other commands
                        default: {
                            LUISA_ERROR("Command type not supported.");
                        } break;
                    }
                } break;
                default: break;
            }
        }
    }
    if (uniform_buffer_size > 0) {
        static_cast<UploadBuffer const *>(arg_buffer.buffer)->copy_from(uniform_data->data(), arg_buffer.offset, uniform_data->size());
    }
    for (auto &i : _state->upload_alloc.alloc.allocated_buffer()) {
        reinterpret_cast<UploadBuffer *>(i.handle)->flush_host();
    }
    for (auto &i : _state->readback_alloc.alloc.allocated_buffer()) {
        reinterpret_cast<ReadbackBuffer *>(i.handle)->flush_host();
    }
    for (auto &i : _state->upload_alloc.largeBuffers) {
        i->flush_host();
    }
    for (auto &i : _state->readback_alloc.largeBuffers) {
        i->flush_host();
    }
}

vstd::span<VkDescriptorSet> Shader::allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs) const {
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = static_cast<uint>(_desc_set_layout.size()),
        .pSetLayouts = _desc_set_layout.data()};
    auto last_size = descs.size();
    luisa::enlarge_by(descs, _desc_set_layout.size());
    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(
            device()->logic_device(),
            &alloc_info,
            descs.data() + last_size));
    return vstd::span<VkDescriptorSet>{descs.data() + last_size, _desc_set_layout.size()};
}
void Shader::update_desc_set(
    VkDescriptorSet set,
    vstd::vector<VkWriteDescriptorSet> &write_buffer,
    vstd::vector<VkImageView> &img_view_buffer,
    vstd::span<vstd::variant<BufferView, TexView>> texs) {
    write_buffer.clear();
    write_buffer.reserve(texs.size());
    uint arg_idx = 0;
    VkDescriptorBufferInfo buffer_info;
    VkDescriptorImageInfo image_info;
    auto make_desc = [&]<typename T>(T const &t) {
        auto &v = write_buffer.emplace_back();
        v.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        v.dstSet = set;
        v.dstBinding = arg_idx;
        v.dstArrayElement = 1;
        v.descriptorCount = 1;
        auto &&b = _binds[arg_idx];

        switch (b.type) {
            case lc::hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::ConstantBuffer:
            case lc::hlsl::ShaderVariableType::ConstantValue:
            case lc::hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            case lc::hlsl::ShaderVariableType::SRVBufferHeap:
            case lc::hlsl::ShaderVariableType::UAVBufferHeap:
            case lc::hlsl::ShaderVariableType::StructuredBuffer:
            case lc::hlsl::ShaderVariableType::RWStructuredBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            default: break;
        }
        if constexpr (std::is_same_v<T, Argument::Buffer>) {
            buffer_info.buffer = reinterpret_cast<Buffer *>(t.handle)->vk_buffer();
            buffer_info.offset = t.offset;
            buffer_info.range = t.size;
            v.pBufferInfo = &buffer_info;
        }
        if constexpr (std::is_same_v<T, Argument::Texture>) {
            image_info.sampler = nullptr;
            auto &img_view = img_view_buffer.emplace_back();
            auto tex = reinterpret_cast<Texture *>(t.handle);
            VkImageViewCreateInfo img_view_create_info = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .flags = 0,
                .image = tex->vk_image(),
                .viewType = [&]() {
                    switch (tex->dimension()) {
                        case 1:
                            return VK_IMAGE_VIEW_TYPE_1D;
                        case 2:
                            return VK_IMAGE_VIEW_TYPE_2D;
                        default:
                            return VK_IMAGE_VIEW_TYPE_3D;
                    }
                }(),
                .format = Texture::to_vk_format(tex->format()),
                .subresourceRange = VkImageSubresourceRange{.aspectMask = tex->get_aspect(), .baseMipLevel = t.level, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
            VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &img_view_create_info, Device::alloc_callbacks(), &img_view));
            image_info.imageView = img_view;
            image_info.imageLayout = tex->layout(t.level);
            v.pImageInfo = &image_info;
        }
        arg_idx++;
    };
    for (auto i : vstd::range((int64_t)texs.size())) {

        // v.descriptorType = view.index() ==
    }
}
}// namespace lc::vk
