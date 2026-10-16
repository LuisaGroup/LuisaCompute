#include "native_shader.h"

#include "device.h"
#include "stream.h"
#include "log.h"
#include "buffer.h"
#include <luisa/core/logging.h>

#include <limits>

namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;

namespace {

// Owns the descriptor sets of one native dispatch; the sets are released when
// the recording command buffer is recycled, i.e. after its fence has passed.
// It holds a reference to the shader's descriptor pool so that the pool cannot
// be destroyed - by the shader instance going away - while sets allocated from
// it are still pending (see NativeShaderDescriptorPool).
struct DeferredDescriptorSetRelease {
    luisa::shared_ptr<NativeShaderDescriptorPool> pool;
    vstd::vector<VkDescriptorSet> sets;
    DeferredDescriptorSetRelease(
        luisa::shared_ptr<NativeShaderDescriptorPool> pool,
        vstd::vector<VkDescriptorSet> &&sets) noexcept
        : pool{std::move(pool)}, sets{std::move(sets)} {}
    DeferredDescriptorSetRelease(DeferredDescriptorSetRelease const &) = delete;
    DeferredDescriptorSetRelease(DeferredDescriptorSetRelease &&) = default;
    DeferredDescriptorSetRelease &operator=(DeferredDescriptorSetRelease const &) = delete;
    DeferredDescriptorSetRelease &operator=(DeferredDescriptorSetRelease &&) = delete;
    ~DeferredDescriptorSetRelease() noexcept {
        if (pool != nullptr && pool->pool != VK_NULL_HANDLE && !sets.empty()) {
            vkFreeDescriptorSets(pool->device, pool->pool,
                                 static_cast<uint32_t>(sets.size()),
                                 sets.data());
        }
    }
};

[[nodiscard]] VkDescriptorType descriptor_type_of(
    NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
}

}// namespace

NativeShaderDescriptorPool::~NativeShaderDescriptorPool() noexcept {
    if (pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pool, Device::alloc_callbacks());
    }
}

NativeShader::NativeShader(
    Device *device, uint3 block_size,
    luisa::span<const NativeShaderResourceBinding> bindings,
    luisa::span<const std::byte> spirv,
    luisa::string_view entry_point, uint32_t push_constant_size) noexcept
    : Resource{device},
      _block_size{block_size},
      _push_constant_size{push_constant_size},
      _entry_point{entry_point},
      _bindings{bindings.begin(), bindings.end()} {
    LUISA_ASSERT(!_bindings.empty() || push_constant_size > 0u,
                 "A native Vulkan shader must declare at least one binding or a "
                 "push-constant block.");
    // ---- descriptor set layouts -----------------------------------------
    for (auto i = 0u; i < _bindings.size(); i++) {
        auto &&binding = _bindings[i];
        LUISA_ASSERT(binding.array_size == 1u,
                     "Native Vulkan shader binding (set {}, binding {}) is an "
                     "array of {} descriptors, which is not supported yet.",
                     binding.space_index, binding.register_index, binding.array_size);
        auto max_set = binding.space_index;
        if (_set_layouts.size() <= max_set) {
            _set_layouts.resize(max_set + 1u, VK_NULL_HANDLE);
        }
        LayoutBinding layout_binding;
        layout_binding.binding_index = i;
        layout_binding.set = binding.space_index;
        layout_binding.binding = binding.register_index;
        layout_binding.descriptor_count = binding.array_size;
        layout_binding.descriptor_type = descriptor_type_of(binding.kind);
        _layout_bindings.emplace_back(layout_binding);
    }
    _descriptor_set_count = static_cast<uint32_t>(_set_layouts.size());
    auto logic_device = device->logic_device();
    for (auto set = 0u; set < _set_layouts.size(); set++) {
        vstd::vector<VkDescriptorSetLayoutBinding> set_bindings;
        for (auto &&binding : _layout_bindings) {
            if (binding.set != set) { continue; }
            VkDescriptorSetLayoutBinding layout_binding{
                .binding = binding.binding,
                .descriptorType = binding.descriptor_type,
                .descriptorCount = binding.descriptor_count,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr};
            set_bindings.emplace_back(layout_binding);
        }
        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(set_bindings.size()),
            .pBindings = set_bindings.data()};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
            logic_device, &layout_info, Device::alloc_callbacks(),
            &_set_layouts[set]));
    }
    auto ok = false;
    auto dispose_partial = vstd::scope_exit([&] {
        if (ok) { return; }
        auto callbacks = Device::alloc_callbacks();
        for (auto layout : _set_layouts) {
            vkDestroyDescriptorSetLayout(logic_device, layout, callbacks);
        }
        _set_layouts.clear();
        if (_descriptor_pool != nullptr) {
            vkDestroyDescriptorPool(logic_device, _descriptor_pool->pool, callbacks);
            _descriptor_pool->pool = VK_NULL_HANDLE;
            _descriptor_pool = nullptr;
        }
        if (_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(logic_device, _pipeline, callbacks);
            _pipeline = VK_NULL_HANDLE;
        }
        if (_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(logic_device, _pipeline_layout, callbacks);
            _pipeline_layout = VK_NULL_HANDLE;
        }
    });
    // ---- pipeline layout -------------------------------------------------
    VkPushConstantRange push_constant_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0u,
        .size = push_constant_size};
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(_set_layouts.size()),
        .pSetLayouts = _set_layouts.data(),
        .pushConstantRangeCount = push_constant_size > 0u ? 1u : 0u,
        .pPushConstantRanges = push_constant_size > 0u ? &push_constant_range : nullptr};
    VK_CHECK_RESULT(vkCreatePipelineLayout(
        logic_device, &pipeline_layout_info, Device::alloc_callbacks(),
        &_pipeline_layout));
    // ---- pipeline --------------------------------------------------------
    VkShaderModuleCreateInfo module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size_bytes(),
        .pCode = reinterpret_cast<const uint32_t *>(spirv.data())};
    VkShaderModule module{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreateShaderModule(
        logic_device, &module_info, Device::alloc_callbacks(), &module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(logic_device, module, Device::alloc_callbacks());
    });
    VkPipelineCacheCreateInfo cache_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache pipeline_cache{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreatePipelineCache(
        logic_device, &cache_info, Device::alloc_callbacks(), &pipeline_cache));
    auto dispose_cache = vstd::scope_exit([&] {
        vkDestroyPipelineCache(logic_device, pipeline_cache,
                               Device::alloc_callbacks());
    });
    VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = 0u,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0u,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = _entry_point.c_str(),
            .pSpecializationInfo = nullptr},
        .layout = _pipeline_layout};
    VK_CHECK_RESULT(vkCreateComputePipelines(
        logic_device, pipeline_cache, 1u, &pipeline_info,
        Device::alloc_callbacks(), &_pipeline));
    // ---- descriptor pool -------------------------------------------------
    vstd::vector<VkDescriptorPoolSize> pool_sizes;
    auto count_of = [&](VkDescriptorType type) noexcept {
        auto count = 0u;
        for (auto &&binding : _layout_bindings) {
            if (binding.descriptor_type == type) {
                count += binding.descriptor_count;
            }
        }
        return count;
    };
    for (auto type : {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}) {
        auto per_set = count_of(type);
        if (per_set > 0u) {
            pool_sizes.emplace_back(
                VkDescriptorPoolSize{type, per_set * max_in_flight_descriptor_sets});
        }
    }
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = max_in_flight_descriptor_sets,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data()};
    VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreateDescriptorPool(
        logic_device, &pool_info, Device::alloc_callbacks(), &descriptor_pool));
    _descriptor_pool = luisa::make_shared<NativeShaderDescriptorPool>(
        logic_device, descriptor_pool);
    ok = true;
}

NativeShader::~NativeShader() noexcept {
    auto logic_device = device()->logic_device();
    auto callbacks = Device::alloc_callbacks();
    // The descriptor pool is released by the shared holder, once no dispatch
    // still references its sets (see NativeShaderDescriptorPool).
    _descriptor_pool = nullptr;
    if (_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(logic_device, _pipeline, callbacks);
    }
    if (_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(logic_device, _pipeline_layout, callbacks);
    }
    for (auto layout : _set_layouts) {
        vkDestroyDescriptorSetLayout(logic_device, layout, callbacks);
    }
}

void NativeShader::encode(
    CommandBufferState *state, VkCommandBuffer cmdbuffer,
    NativeShaderDispatchCommand const *cmd) noexcept {
    auto logic_device = device()->logic_device();
    auto arguments = cmd->arguments();
    // ---- descriptor sets -------------------------------------------------
    // One set per descriptor set index; the recording view is kept in `sets`
    // while a copy is handed to the command-buffer state, which releases them
    // once the frame has been retired.
    vstd::vector<VkDescriptorSet> sets(_descriptor_set_count, VK_NULL_HANDLE);
    if (_descriptor_set_count > 0u) {
        LUISA_ASSERT(_descriptor_pool != nullptr,
                     "Native shader has no descriptor pool.");
        VkDescriptorSetAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _descriptor_pool->pool,
            .descriptorSetCount = static_cast<uint32_t>(_set_layouts.size()),
            .pSetLayouts = _set_layouts.data()};
        auto allocate_result = vkAllocateDescriptorSets(
            logic_device, &allocate_info, sets.data());
        if (allocate_result != VK_SUCCESS) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader dispatch could not allocate {} descriptor sets "
                "from the shader's pool (VkResult {}); at most {} dispatches of "
                "one native shader may be recorded per command buffer.",
                sets.size(), luisa::to_underlying(allocate_result),
                max_in_flight_descriptor_sets);
            return;
        }
        state->dispose_after_flush(DeferredDescriptorSetRelease{
            _descriptor_pool, vstd::vector<VkDescriptorSet>{sets}});
    }
    // ---- descriptor writes -----------------------------------------------
    vstd::vector<VkDescriptorBufferInfo> buffer_infos;
    vstd::vector<VkWriteDescriptorSet> writes;
    buffer_infos.reserve(arguments.size());
    writes.reserve(arguments.size());
    for (auto i = 0u; i < arguments.size(); i++) {
        auto &&argument = arguments[i];
        if (argument.tag == Argument::Tag::UNIFORM) { continue; }
        if (i >= _bindings.size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader dispatch supplied {} resource arguments for {} "
                "reflected bindings.",
                arguments.size(), _bindings.size());
            return;
        }
        auto &&binding = _bindings[i];
        if (argument.tag != Argument::Tag::BUFFER) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader dispatch argument {} is not a buffer; textures, "
                "bindless arrays and acceleration structures are rejected at "
                "load().",
                i);
            return;
        }
        auto buffer = reinterpret_cast<Buffer const *>(argument.buffer.handle);
        if (buffer == nullptr) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader dispatch argument {} has a null buffer.", i);
            return;
        }
        auto offset = static_cast<VkDeviceSize>(argument.buffer.offset);
        auto addressable = static_cast<VkDeviceSize>(buffer->addressable_byte_size());
        if (offset >= addressable) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader dispatch argument {} has offset {} beyond the "
                "buffer's addressable size {}.",
                i, offset, addressable);
            return;
        }
        auto requested = static_cast<VkDeviceSize>(argument.buffer.size);
        auto range = requested == 0u ? addressable - offset :
                                       std::min(requested, addressable - offset);
        auto descriptor_type = descriptor_type_of(binding.kind);
        if (descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
            range > device()->properties().limits.maxUniformBufferRange) {
            LUISA_ERROR_WITH_LOCATION(
                "Native shader constant buffer at set {} binding {} needs {} "
                "bytes, exceeding maxUniformBufferRange ({}).",
                binding.space_index, binding.register_index, range,
                device()->properties().limits.maxUniformBufferRange);
            return;
        }
        buffer_infos.emplace_back(
            VkDescriptorBufferInfo{buffer->vk_buffer(), offset, range});
        writes.emplace_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = sets[binding.space_index],
            .dstBinding = binding.register_index,
            .dstArrayElement = 0u,
            .descriptorCount = 1u,
            .descriptorType = descriptor_type,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_infos.back(),
            .pTexelBufferView = nullptr});
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(logic_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0u, nullptr);
    }
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    if (!sets.empty()) {
        vkCmdBindDescriptorSets(
            cmdbuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline_layout,
            0u, static_cast<uint32_t>(sets.size()), sets.data(), 0u, nullptr);
    }
    // ---- push constants (the launcher's uniform block) -------------------
    if (_push_constant_size > 0u) {
        auto begin = std::numeric_limits<uint64_t>::max();
        auto end = uint64_t{0u};
        for (auto &&argument : arguments) {
            if (argument.tag == Argument::Tag::UNIFORM) {
                begin = std::min<uint64_t>(begin, argument.uniform.offset);
                end = std::max<uint64_t>(end, argument.uniform.offset +
                                                   argument.uniform.size);
            }
        }
        if (end > begin) {
            auto bytes = end - begin;
            if (bytes % sizeof(uint32_t) != 0u || bytes > _push_constant_size) {
                LUISA_ERROR_WITH_LOCATION(
                    "Native shader uniform payload ({} bytes) does not fit the "
                    "declared push-constant block ({} bytes).",
                    bytes, _push_constant_size);
                return;
            }
            auto payload = cmd->uniform(Argument::Uniform{
                static_cast<size_t>(begin), static_cast<size_t>(bytes), 4u});
            vkCmdPushConstants(
                cmdbuffer, _pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0u, static_cast<uint32_t>(bytes), payload.data());
        }
    }
    auto block_size = cmd->block_size();
    auto threads = cmd->dispatch_size();
    auto ceil_div = [](uint32_t value, uint32_t divisor) noexcept {
        return (value + divisor - 1u) / divisor;
    };
    vkCmdDispatch(cmdbuffer, ceil_div(threads.x, block_size.x),
                  ceil_div(threads.y, block_size.y),
                  ceil_div(threads.z, block_size.z));
}

}// namespace lc::vk
