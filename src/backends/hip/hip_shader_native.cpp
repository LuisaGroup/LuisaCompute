#include <luisa/runtime/rhi/command.h>
#include <hip/hiprtc.h>
#include <hiprt/hiprt.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "hip_device.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_accel.h"
#include "hip_command_encoder.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_shader_printer.h"
#include "hip_check.h"

namespace luisa::compute::hip {

namespace {

void validate_register_limit(hipFunction_t function,
                             uint32_t requested_limit) noexcept {
    if (requested_limit == 0u) { return; }
    auto effective_limit = std::min(requested_limit, 256u);
    auto actual_register_count = 0;
    LUISA_CHECK_HIP(hipFuncGetAttribute(
        &actual_register_count, HIP_FUNC_ATTRIBUTE_NUM_REGS, function));
    LUISA_ASSERT(
        actual_register_count <= static_cast<int>(effective_limit),
        "HIP compiler ignored ShaderOption::max_registers: requested at most "
        "{} VGPRs, but the loaded kernel uses {}.",
        effective_limit, actual_register_count);
    LUISA_VERBOSE("HIP kernel register limit satisfied: {} / {} VGPRs.",
                  actual_register_count, effective_limit);
}

}// namespace

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{HIPShaderPrinter::create(metadata.format_types),
                metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _requires_global_rt_stack{false} {

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

    if (auto dump_dir = std::getenv("LUISA_DUMP_HIP_ISA")) {
        static int _isa_counter = 0;
        auto path = fmt::format("{}/hip_isa_{}.co", dump_dir, _isa_counter++);
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(static_cast<const char *>(linked_binary), linked_binary_size);
        LUISA_INFO("Dumped HIP code object ({} bytes) to: {}", linked_binary_size, path);
    }

    hiprtcLinkDestroy(link_state);
    if (ret != hipSuccess) {
        LUISA_ERROR_WITH_LOCATION("Failed to load HIP module: {}", hipGetErrorString(ret));
    }
    LUISA_CHECK_HIP(hipModuleGetFunction(&_function, _module, entry));
    validate_register_limit(_function, metadata.max_register_count);
}

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 hiprtContext hiprt_ctx,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{HIPShaderPrinter::create(metadata.format_types),
                metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _requires_global_rt_stack{metadata.requires_global_rt_stack} {

    LUISA_ASSERT(hiprt_ctx != nullptr, "HIPRT context is null for ray-tracing shader.");

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

    if (auto dump_dir = std::getenv("LUISA_DUMP_HIP_ISA")) {
        static int _rt_isa_counter = 0;
        auto path = fmt::format("{}/hip_rt_isa_{}.co", dump_dir, _rt_isa_counter++);
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(static_cast<const char *>(linked_binary), linked_binary_size);
        LUISA_INFO("Dumped HIP RT code object ({} bytes) to: {}", linked_binary_size, path);
    }

    hiprtcLinkDestroy(link_state);
    if (ret != hipSuccess) {
        LUISA_ERROR_WITH_LOCATION("Failed to load HIP RT module: {}", hipGetErrorString(ret));
    }
    LUISA_CHECK_HIP(hipModuleGetFunction(&_function, _module, entry));
    validate_register_limit(_function, metadata.max_register_count);
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
                if (buffer->is_indirect()) {
                    auto binding = buffer->indirect_binding(
                        arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                } else {
                    auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                    auto ptr = allocate_argument(sizeof(binding));
                    std::memcpy(ptr, &binding, sizeof(binding));
                }
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

    auto printer_encode = HIPShaderPrinter::Encode{};
    if (printer() != nullptr) {
        printer_encode = printer()->encode(encoder);
        auto binding = printer_encode.binding();
        auto printer_arg = allocate_argument(sizeof(binding));
        std::memcpy(printer_arg, &binding, sizeof(binding));
    }

    auto launch_size_and_kernel_id = allocate_argument(sizeof(uint4));

    if (_requires_global_rt_stack) {
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
    auto arg_size = argument_buffer_offset;
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, argument_buffer.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END};
    auto launch = [&](uint3 dispatch_size, uint kernel_id) noexcept {
        if (any(dispatch_size == make_uint3(0u))) { return; }
        auto launch_info = make_uint4(dispatch_size, kernel_id);
        std::memcpy(launch_size_and_kernel_id, &launch_info, sizeof(launch_info));
        auto blocks = (dispatch_size + block_size - 1u) / block_size;
        LUISA_CHECK_HIP(hipModuleLaunchKernel(
            _function,
            blocks.x, blocks.y, blocks.z,
            block_size.x, block_size.y, block_size.z,
            0u, hip_stream, nullptr, extra));
    };

    if (command->is_indirect()) {
        // HIP has no CUDA-device-runtime equivalent for launching arbitrary
        // GPU-authored grids. Preserve correctness by resolving the launch
        // records at this command boundary, then enqueue the resulting grids
        // back onto the same stream.
        auto indirect = command->indirect_dispatch();
        auto buffer = reinterpret_cast<const HIPBuffer *>(indirect.handle);
        LUISA_ASSERT(buffer->is_indirect(),
                     "Indirect dispatch command references a regular HIP buffer.");
        auto binding = buffer->indirect_binding(
            indirect.offset, indirect.max_dispatch_size);
        auto offset = static_cast<uint32_t>(binding.offset_and_capacity);
        auto end = static_cast<uint32_t>(binding.offset_and_capacity >> 32u);
        HIPBuffer::IndirectHeader header{};
        LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
        LUISA_CHECK_HIP(hipMemcpyDtoH(
            &header, binding.ptr, sizeof(header)));
        auto count = std::min<uint32_t>(header.size, end - offset);
        luisa::vector<HIPBuffer::IndirectDispatch> dispatches(count);
        if (count != 0u) {
            auto src = static_cast<const std::byte *>(binding.ptr) +
                       sizeof(HIPBuffer::IndirectHeader) +
                       sizeof(HIPBuffer::IndirectDispatch) * offset;
            LUISA_CHECK_HIP(hipMemcpyDtoH(
                dispatches.data(), const_cast<std::byte *>(src),
                sizeof(HIPBuffer::IndirectDispatch) * count));
        }
        for (auto &&dispatch : dispatches) {
            auto record_block_size = make_uint3(
                dispatch.block_size[0], dispatch.block_size[1],
                dispatch.block_size[2]);
            LUISA_ASSERT(all(record_block_size == block_size),
                         "Indirect HIP block-size mismatch: record is ({}, {}, {}), "
                         "shader requires ({}, {}, {}).",
                         record_block_size.x, record_block_size.y,
                         record_block_size.z, block_size.x, block_size.y,
                         block_size.z);
            launch(make_uint3(
                       dispatch.dispatch_size_and_kernel_id[0],
                       dispatch.dispatch_size_and_kernel_id[1],
                       dispatch.dispatch_size_and_kernel_id[2]),
                   dispatch.dispatch_size_and_kernel_id[3]);
        }
    } else if (command->is_multiple_dispatch()) {
        for (auto dispatch_size : command->dispatch_sizes()) {
            launch(dispatch_size, 0u);
        }
    } else {
        launch(command->dispatch_size(), 0u);
    }
    printer_encode.commit(encoder);
}

}// namespace luisa::compute::hip
