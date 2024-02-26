#include "stream.h"
#include "device.h"
#include <luisa/core/logging.h>
#include "log.h"
namespace lc::vk {
namespace temp_buffer {

template<typename Pack>
uint64 Visitor<Pack>::allocate(uint64 size) {
    return reinterpret_cast<uint64_t>(new Pack(device, size));
}
template<typename Pack>
void Visitor<Pack>::deallocate(uint64 handle) {
    delete reinterpret_cast<Pack *>(handle);
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
      default_alloc(TEMP_SIZE),
      readback_alloc(TEMP_SIZE) {
}
void CommandBufferState::reset(Device &device) {
    upload_alloc.clear();
    default_alloc.clear();
    readback_alloc.clear();
    if (!_desc_sets.empty()) {
        VK_CHECK_RESULT(
            vkFreeDescriptorSets(
                device.logic_device(), device.desc_pool(), _desc_sets.size(), _desc_sets.data()));
        _desc_sets.clear();
    }
}
void CommandBuffer::reset() {
    _state->reset(*device());
    VK_CHECK_RESULT(vkResetCommandBuffer(_cmdbuffer, 0));
}

Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      reorder({}),
      _thd([this]() {
          while (_enabled) {
              while (auto p = _exec.pop()) {
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
              std::unique_lock lck{_mtx};
              while (_enabled && _exec.length() == 0) {
                  _cv.wait(lck);
              }
          }
      }) {
    VkCommandPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    switch (tag) {
        case StreamTag::GRAPHICS:
            pool_ci.queueFamilyIndex = device->graphics_queue_index();
            _queue = device->graphics_queue();
            break;
        case StreamTag::COPY:
            pool_ci.queueFamilyIndex = device->copy_queue_index();
            _queue = device->compute_queue();
            break;
        case StreamTag::COMPUTE:
            pool_ci.queueFamilyIndex = device->compute_queue_index();
            _queue = device->copy_queue();
            break;
        default:
            LUISA_ASSERT(false, "Illegal stream tag.");
    }
    VK_CHECK_RESULT(vkCreateCommandPool(device->logic_device(), &pool_ci, Device::alloc_callbacks(), &_pool));
}
Stream::~Stream() {
    sync();
    {
        std::lock_guard lck{_mtx};
        _enabled = false;
    }
    _cv.notify_one();
    _thd.join();
    vkDestroyCommandPool(device()->logic_device(), _pool, Device::alloc_callbacks());
}
void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    bool inqueue_limit) {

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
        auto cb = cmdbuffer.cmdbuffer();
        cmdbuffer.begin();
        cmdbuffer.execute(cmds);
        cmdbuffer.end();
        _evt.signal(*this, fence, &cb);
        _exec.push(SyncExt{
            .evt = &_evt,
            .value = fence});
        _exec.push(std::move(cmdbuffer));
    } else {
        _evt.update_fence(fence);
    }
    if (!callbacks.empty()) {
        _exec.push(std::move(callbacks));
    }
    _exec.push(NotifyEvt{
        .evt = &_evt,
        .value = fence});

    _mtx.lock();
    _mtx.unlock();
    _cv.notify_one();
}
void Stream::sync() {
    _evt.sync(_evt.last_fence());
}
CommandBuffer::CommandBuffer(Stream &stream)
    : Resource(stream.device()),
      stream(stream),
      _state(vstd::make_unique<CommandBufferState>()) {
    VkCommandBufferAllocateInfo cb_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = stream.pool(),
        .commandBufferCount = 1};
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device()->logic_device(), &cb_ci, &_cmdbuffer));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), nullptr));
}
CommandBuffer::~CommandBuffer() {
    if (_cmdbuffer)
        vkFreeCommandBuffers(device()->logic_device(), stream.pool(), 1, &_cmdbuffer);
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
CommandBuffer::CommandBuffer(CommandBuffer &&rhs)
    : Resource(std::move(rhs)),
      stream(rhs.stream),
      _cmdbuffer(rhs._cmdbuffer),
      _state(std::move(rhs._state)) {
    rhs._cmdbuffer = nullptr;
}
void Stream::signal(Event *event, uint64_t value) {
    event->signal(*this, value);
    _exec.push(SyncExt{event, value});
    _exec.push(NotifyEvt{event, value});
    _mtx.lock();
    _mtx.unlock();
    _cv.notify_one();
}
void Stream::wait(Event *event, uint64_t value) {
    event->wait(*this, value);
}
void CommandBuffer::execute(vstd::span<const luisa::unique_ptr<Command>> cmds) {
    for (auto &&command : cmds) {
        command->accept(stream.reorder);
    }
    auto cmd_lists = stream.reorder.command_lists();
    auto clear_reorder = vstd::scope_exit([&] {
        stream.reorder.clear();
    });
    for (auto &&lst : cmd_lists) {
        // Preprocess: record resources' states
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {

                } break;
                case Command::Tag::EBufferDownloadCommand: {
                } break;
                case Command::Tag::EBufferCopyCommand: {
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                } break;
                case Command::Tag::ETextureUploadCommand: {
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                } break;
                case Command::Tag::ETextureCopyCommand: {
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                } break;
                case Command::Tag::EAccelBuildCommand: {
                } break;
                case Command::Tag::EMeshBuildCommand: {
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                } break;
                default: break;
            }
        }
        // Execute
        for (auto i = lst; i != nullptr; i = i->p_next) {
            auto cmd = i->cmd;
            switch (cmd->tag()) {
                case Command::Tag::EBufferUploadCommand: {

                } break;
                case Command::Tag::EBufferDownloadCommand: {
                } break;
                case Command::Tag::EBufferCopyCommand: {
                } break;
                case Command::Tag::EBufferToTextureCopyCommand: {
                } break;
                case Command::Tag::EShaderDispatchCommand: {
                } break;
                case Command::Tag::ETextureUploadCommand: {
                } break;
                case Command::Tag::ETextureDownloadCommand: {
                } break;
                case Command::Tag::ETextureCopyCommand: {
                } break;
                case Command::Tag::ETextureToBufferCopyCommand: {
                } break;
                case Command::Tag::EAccelBuildCommand: {
                } break;
                case Command::Tag::EMeshBuildCommand: {
                } break;
                case Command::Tag::ECurveBuildCommand: {
                } break;
                case Command::Tag::EProceduralPrimitiveBuildCommand: {
                } break;
                case Command::Tag::EBindlessArrayUpdateCommand: {
                } break;
                default: break;
            }
        }
    }
}

vstd::span<VkDescriptorSet> Shader::allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs) {
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = static_cast<uint>(_desc_set_layout.size()),
        .pSetLayouts = _desc_set_layout.data()};
    auto last_size = _desc_set_layout.size();
    descs.push_back_uninitialized(_desc_set_layout.size());
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
            case lc::hlsl::ShaderVariableType::ConstantBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case lc::hlsl::ShaderVariableType::SRVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::UAVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            case lc::hlsl::ShaderVariableType::StructuredBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::RWStructuredBuffer:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case lc::hlsl::ShaderVariableType::ConstantValue:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
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
                        case 3:
                            return VK_IMAGE_VIEW_TYPE_3D;
                    }
                }(),
                .format = Texture::to_vk_format(tex->format()),
                .subresourceRange = VkImageSubresourceRange{.baseMipLevel = t.level, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
            VK_CHECK_RESULT(vkCreateImageView(device()->logic_device(), &img_view_create_info, Device::alloc_callbacks(), &img_view));
            image_info.imageView = img_view;
            image_info.imageLayout = tex->layout(t.level);
            v.pImageInfo = &image_info;
        }
        arg_idx++;
    };
    for (auto i : vstd::range(texs.size())) {

        // v.descriptorType = view.index() ==
    }
}
}// namespace lc::vk
