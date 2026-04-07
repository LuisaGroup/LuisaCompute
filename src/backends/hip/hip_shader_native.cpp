#include <luisa/runtime/rhi/command.h>
#include <hip/hiprtc.h>
#include <hiprt/hiprt.h>
#include "hip_device.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_accel.h"
#include "hip_command_encoder.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_check.h"

namespace luisa::compute::hip {

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _is_rt{false} {

    hiprtcLinkState link_state{};
    auto create_result = hiprtcLinkCreate(0, nullptr, nullptr, &link_state);
    if (create_result != hiprtcResult::HIPRTC_SUCCESS) {
        LUISA_ERROR_WITH_LOCATION("Failed to create hiprtc link state: {}",
                                  hiprtcGetErrorString(create_result));
    }

    auto add_result = hiprtcLinkAddData(link_state,
                                        hipJitInputLLVMBitcode,
                                        code.data(),
                                        code.size(),
                                        entry,
                                        0, nullptr, nullptr);
    if (add_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to add LLVM bitcode to hiprtc linker: {}",
                                  hiprtcGetErrorString(add_result));
    }

    void *linked_binary = nullptr;
    size_t linked_binary_size = 0;
    auto complete_result = hiprtcLinkComplete(link_state, &linked_binary, &linked_binary_size);
    if (complete_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to complete hiprtc linking: {}",
                                  hiprtcGetErrorString(complete_result));
    }

    auto ret = hipModuleLoadData(&_module, linked_binary);
    hiprtcLinkDestroy(link_state);
    if (ret != hipSuccess) {
        LUISA_ERROR_WITH_LOCATION("Failed to load HIP module: {}", hipGetErrorString(ret));
    }
    LUISA_CHECK_HIP(hipModuleGetFunction(&_function, _module, entry));
}

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 hiprtContext hiprt_ctx,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _is_rt{true} {

    // RT shaders now have HIPRT library fully inlined by our LLVM pipeline.
    // Use the same hiprtc ISA-generation path as non-RT shaders.
    hiprtcLinkState link_state{};
    auto create_result = hiprtcLinkCreate(0, nullptr, nullptr, &link_state);
    if (create_result != hiprtcResult::HIPRTC_SUCCESS) {
        LUISA_ERROR_WITH_LOCATION("Failed to create hiprtc link state for RT shader: {}",
                                  hiprtcGetErrorString(create_result));
    }

    auto add_result = hiprtcLinkAddData(link_state,
                                        hipJitInputLLVMBitcode,
                                        code.data(),
                                        code.size(),
                                        entry,
                                        0, nullptr, nullptr);
    if (add_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to add LLVM bitcode to hiprtc linker for RT shader: {}",
                                  hiprtcGetErrorString(add_result));
    }

    void *linked_binary = nullptr;
    size_t linked_binary_size = 0;
    auto complete_result = hiprtcLinkComplete(link_state, &linked_binary, &linked_binary_size);
    if (complete_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to complete hiprtc linking for RT shader: {}",
                                  hiprtcGetErrorString(complete_result));
    }

    auto ret = hipModuleLoadData(&_module, linked_binary);
    hiprtcLinkDestroy(link_state);
    if (ret != hipSuccess) {
        LUISA_ERROR_WITH_LOCATION("Failed to load HIP RT module: {}", hipGetErrorString(ret));
    }
    LUISA_CHECK_HIP(hipModuleGetFunction(&_function, _module, entry));
    LUISA_INFO("RT shader compiled via LTO pipeline (bypassing HIPRT compiler).");
}

HIPShaderNative::~HIPShaderNative() noexcept {
    if (_module != nullptr) {
        LUISA_CHECK_HIP(hipModuleUnload(_module));
    }
}

void HIPShaderNative::_launch(HIPCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    static thread_local std::array<std::byte, 65536u> argument_buffer;

    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        LUISA_ASSERT(offset + bytes <= argument_buffer.size(),
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return argument_buffer.data() + offset;
    };

    auto encode_argument = [&allocate_argument, command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        switch (arg.tag) {
            case Tag::BUFFER: {
                auto buffer = reinterpret_cast<const HIPBuffer *>(arg.buffer.handle);
                auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<HIPTexture *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                LUISA_ASSERT(arg.uniform.alignment <= 16u, "Invalid uniform alignment {}.",
                             arg.uniform.alignment);
                auto ptr = allocate_argument(uniform.size_bytes());
                std::memcpy(ptr, uniform.data(), uniform.size_bytes());
                break;
            }
            case Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<HIPBindlessArray *>(arg.bindless_array.handle);
                auto binding = array->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::ACCEL: {
                auto accel = reinterpret_cast<const HIPAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
        }
    };

    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }

    auto ptr = allocate_argument(sizeof(uint4));
    auto dispatch_size = command->dispatch_size();
    uint4 launch_size_and_kernel_id = make_uint4(dispatch_size, 0u);
    std::memcpy(ptr, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));

    if (_is_rt) {
        // RT stack fields must be packed contiguously to match LLVM struct layout:
        //   i32 (stack_size) | i32 (stack_count) | ptr (stack_data) = 16 bytes total
        // We allocate a single 16-byte-aligned block rather than 3 separate aligned fields.
        struct alignas(16) RTStackArgs {
            uint32_t stack_size;
            uint32_t stack_count;
            void *stack_data;
        };
        static_assert(sizeof(RTStackArgs) == 16u);
        auto stack_buf = _device->hiprt_global_stack_buffer();
        RTStackArgs rt_args{
            .stack_size = stack_buf.stackSize,
            .stack_count = stack_buf.stackCount,
            .stack_data = stack_buf.stackData,
        };
        auto p_rt = allocate_argument(sizeof(RTStackArgs));
        std::memcpy(p_rt, &rt_args, sizeof(RTStackArgs));
    }

    auto hip_stream = encoder.stream()->handle();
    auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    auto blocks = (dispatch_size + block_size - 1u) / block_size;

    auto arg_size = argument_buffer_offset;
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, argument_buffer.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END};
    LUISA_CHECK_HIP(hipModuleLaunchKernel(
        _function,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, hip_stream, nullptr, extra));
}

}// namespace luisa::compute::hip
