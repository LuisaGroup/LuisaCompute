#include <luisa/runtime/rhi/command.h>
#include <luisa/core/clock.h>
#include <hip/hiprtc.h>
#include <hiprt/hiprt.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include "hip_device.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_accel.h"
#include "hip_command_encoder.h"
#include "hip_shader.h"
#include "hip_shader_link_options.h"
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

class HIPRTCLinkState {

private:
    // LLVM's AMDGPU TTI has a target-independent profitability model followed
    // by amdgpu-inline-max-bb, a separate compile-time guard. A zero guard
    // removes only that arbitrary basic-block ceiling: LLVM's ordinary inline
    // advisor still owns every profitability decision. This matches the HIP
    // device-function hierarchy used by Cycles without attaching inline policy
    // to individual Luisa callables.
    HIPRTCLinkOptions _options;
    hiprtcLinkState _state{};

public:
    HIPRTCLinkState() noexcept {
        auto result = hiprtcLinkCreate(
            _options.jit_option_count(),
            _options.jit_options(),
            _options.jit_option_values(),
            &_state);
        LUISA_ASSERT(
            result == hiprtcResult::HIPRTC_SUCCESS,
            "Failed to create hiprtc link state: {}.",
            hiprtcGetErrorString(result));
    }

    ~HIPRTCLinkState() noexcept {
        if (_state != nullptr) {
            auto result = hiprtcLinkDestroy(_state);
            LUISA_ASSERT(
                result == hiprtcResult::HIPRTC_SUCCESS,
                "Failed to destroy hiprtc link state: {}.",
                hiprtcGetErrorString(result));
        }
    }

    HIPRTCLinkState(HIPRTCLinkState &&) noexcept = delete;
    HIPRTCLinkState(const HIPRTCLinkState &) noexcept = delete;
    HIPRTCLinkState &operator=(HIPRTCLinkState &&) noexcept = delete;
    HIPRTCLinkState &operator=(const HIPRTCLinkState &) noexcept = delete;

    void add_llvm_bitcode(
        luisa::string_view bitcode,
        const char *entry) noexcept {
        auto result = hiprtcLinkAddData(
            _state, hipJitInputLLVMBitcode,
            const_cast<char *>(bitcode.data()),
            bitcode.size(), entry, 0, nullptr, nullptr);
        LUISA_ASSERT(
            result == hiprtcResult::HIPRTC_SUCCESS,
            "Failed to add LLVM bitcode to hiprtc linker: {}.",
            hiprtcGetErrorString(result));
    }

    [[nodiscard]] luisa::vector<std::byte>
    complete() noexcept {
        void *linked_binary = nullptr;
        size_t linked_binary_size = 0u;
        auto result = hiprtcLinkComplete(
            _state, &linked_binary, &linked_binary_size);
        LUISA_ASSERT(
            result == hiprtcResult::HIPRTC_SUCCESS,
            "Failed to complete hiprtc linking: {}.",
            hiprtcGetErrorString(result));
        LUISA_ASSERT(
            linked_binary != nullptr &&
                linked_binary_size != 0u,
            "hiprtc produced an empty code object.");
        auto bytes =
            static_cast<const std::byte *>(linked_binary);
        return luisa::vector<std::byte>{
            bytes, bytes + linked_binary_size};
    }
};

}// namespace

luisa::vector<std::byte> hip_link_llvm_bitcode(
    luisa::string_view bitcode, const char *entry) noexcept {
    Clock clock;
    HIPRTCLinkState linker;
    linker.add_llvm_bitcode(bitcode, entry);
    auto code_object = linker.complete();
    LUISA_INFO(
        "Linked HIP LLVM bitcode to AMDGPU code object "
        "({} bytes) in {} ms.",
        code_object.size(), clock.toc());
    return code_object;
}

void HIPShaderNative::_load_code_object(
    luisa::span<const std::byte> code_object,
    const HIPShaderMetadata &metadata,
    bool ray_tracing) noexcept {
    LUISA_ASSERT(
        !code_object.empty(),
        "Cannot load an empty HIP code object.");
    Clock clock;
    LUISA_CHECK_HIP(hipModuleLoadData(
        &_module, code_object.data()));

    if (auto dump_dir = std::getenv("LUISA_DUMP_HIP_ISA")) {
        static int compute_isa_counter = 0;
        static int ray_tracing_isa_counter = 0;
        auto index = ray_tracing ?
                         ray_tracing_isa_counter++ :
                         compute_isa_counter++;
        auto path = fmt::format(
            "{}/hip_{}isa_{}.co", dump_dir,
            ray_tracing ? "rt_" : "", index);
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(
            reinterpret_cast<const char *>(
                code_object.data()),
            static_cast<std::streamsize>(
                code_object.size_bytes()));
        LUISA_INFO(
            "Dumped HIP{} code object ({} bytes) to: {}",
            ray_tracing ? " RT" : "",
            code_object.size_bytes(), path);
    }

    LUISA_CHECK_HIP(hipModuleGetFunction(
        &_function, _module, _entry.c_str()));
    validate_register_limit(
        _function, metadata.max_register_count);
    LUISA_INFO(
        "Loaded HIP{} code object in {} ms.",
        ray_tracing ? " RT" : "", clock.toc());
}

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
      _requires_global_rt_stack{false},
      _uses_static_global_rt_stack{false} {
    auto code_object =
        hip_link_llvm_bitcode(code, entry);
    _load_code_object(
        code_object, metadata, false);
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
      _requires_global_rt_stack{metadata.requires_global_rt_stack},
      _uses_static_global_rt_stack{
          metadata.uses_static_global_rt_stack} {

    LUISA_ASSERT(hiprt_ctx != nullptr, "HIPRT context is null for ray-tracing shader.");
    auto code_object =
        hip_link_llvm_bitcode(code, entry);
    _load_code_object(
        code_object, metadata, true);
}

HIPShaderNative::HIPShaderNative(
    HIPDevice *device,
    luisa::span<const std::byte> code_object,
    const char *entry,
    const HIPShaderMetadata &metadata,
    luisa::vector<ShaderDispatchCommand::Argument>
        bound_arguments) noexcept
    : HIPShader{
          HIPShaderPrinter::create(metadata.format_types),
          metadata.argument_usages},
      _entry{entry},
      _block_size{
          metadata.block_size.x,
          metadata.block_size.y,
          metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _requires_global_rt_stack{false},
      _uses_static_global_rt_stack{false} {
    _load_code_object(
        code_object, metadata, false);
}

HIPShaderNative::HIPShaderNative(
    HIPDevice *device,
    luisa::span<const std::byte> code_object,
    const char *entry,
    const HIPShaderMetadata &metadata,
    hiprtContext hiprt_ctx,
    luisa::vector<ShaderDispatchCommand::Argument>
        bound_arguments) noexcept
    : HIPShader{
          HIPShaderPrinter::create(metadata.format_types),
          metadata.argument_usages},
      _entry{entry},
      _block_size{
          metadata.block_size.x,
          metadata.block_size.y,
          metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _device{device},
      _requires_global_rt_stack{
          metadata.requires_global_rt_stack},
      _uses_static_global_rt_stack{
          metadata.uses_static_global_rt_stack} {
    LUISA_ASSERT(
        hiprt_ctx != nullptr,
        "HIPRT context is null for ray-tracing shader.");
    _load_code_object(
        code_object, metadata, true);
}

HIPShaderNative::~HIPShaderNative() noexcept {
    if (_module != nullptr) {
        LUISA_CHECK_HIP(hipModuleUnload(_module));
    }
}

void HIPShaderNative::_launch(HIPCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    auto hip_stream = encoder.stream()->handle();
    auto block_size =
        make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    struct LaunchRecord {
        uint3 dispatch_size;
        uint32_t kernel_id;
    };
    luisa::vector<LaunchRecord> launch_records;
    if (command->is_indirect()) {
        // HIP has no CUDA-device-runtime equivalent for launching arbitrary
        // GPU-authored grids. Resolve the records once before encoding the
        // kernargs; the same records also determine exact static-stack
        // capacity.
        auto indirect = command->indirect_dispatch();
        auto buffer = reinterpret_cast<const HIPBuffer *>(indirect.handle);
        LUISA_ASSERT(
            buffer->is_indirect(),
            "Indirect dispatch command references a regular HIP buffer.");
        auto binding = buffer->indirect_binding(
            indirect.offset, indirect.max_dispatch_size);
        auto offset = static_cast<uint32_t>(
            binding.offset_and_capacity);
        auto end = static_cast<uint32_t>(
            binding.offset_and_capacity >> 32u);
        HIPBuffer::IndirectHeader header{};
        LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
        LUISA_CHECK_HIP(hipMemcpyDtoH(
            &header, binding.ptr, sizeof(header)));
        auto count =
            std::min<uint32_t>(header.size, end - offset);
        luisa::vector<HIPBuffer::IndirectDispatch> dispatches(count);
        if (count != 0u) {
            auto src =
                static_cast<const std::byte *>(binding.ptr) +
                sizeof(HIPBuffer::IndirectHeader) +
                sizeof(HIPBuffer::IndirectDispatch) * offset;
            LUISA_CHECK_HIP(hipMemcpyDtoH(
                dispatches.data(),
                const_cast<std::byte *>(src),
                sizeof(HIPBuffer::IndirectDispatch) * count));
        }
        launch_records.reserve(count);
        for (auto &&dispatch : dispatches) {
            auto record_block_size = make_uint3(
                dispatch.block_size[0],
                dispatch.block_size[1],
                dispatch.block_size[2]);
            LUISA_ASSERT(
                all(record_block_size == block_size),
                "Indirect HIP block-size mismatch: record is ({}, {}, {}), "
                "shader requires ({}, {}, {}).",
                record_block_size.x, record_block_size.y,
                record_block_size.z, block_size.x,
                block_size.y, block_size.z);
            launch_records.emplace_back(LaunchRecord{
                .dispatch_size = make_uint3(
                    dispatch.dispatch_size_and_kernel_id[0],
                    dispatch.dispatch_size_and_kernel_id[1],
                    dispatch.dispatch_size_and_kernel_id[2]),
                .kernel_id =
                    dispatch.dispatch_size_and_kernel_id[3]});
        }
    } else if (command->is_multiple_dispatch()) {
        auto dispatch_sizes = command->dispatch_sizes();
        launch_records.reserve(dispatch_sizes.size());
        for (auto dispatch_size : dispatch_sizes) {
            launch_records.emplace_back(LaunchRecord{
                .dispatch_size = dispatch_size,
                .kernel_id = 0u});
        }
    } else {
        launch_records.emplace_back(LaunchRecord{
            .dispatch_size = command->dispatch_size(),
            .kernel_id = 0u});
    }

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
        auto stack_buf = [&] {
            if (!_uses_static_global_rt_stack) {
                return _device->hiprt_global_stack_buffer();
            }
            auto maximum_physical_threads = size_t{0u};
            auto ceil_div = [](uint32_t x, uint32_t y) noexcept {
                return x / y + static_cast<uint32_t>(x % y != 0u);
            };
            for (auto &&record : launch_records) {
                auto size = record.dispatch_size;
                if (any(size == make_uint3(0u))) { continue; }
                auto blocks = make_uint3(
                    ceil_div(size.x, block_size.x),
                    ceil_div(size.y, block_size.y),
                    ceil_div(size.z, block_size.z));
                auto physical_threads = uint64_t{blocks.x} * blocks.y *
                                        blocks.z * block_size.x *
                                        block_size.y * block_size.z;
                LUISA_ASSERT(
                    physical_threads <=
                        std::numeric_limits<uint32_t>::max(),
                    "HIPRT global-stack launch requires {} physical "
                    "threads; the stack ABI supports at most {}.",
                    physical_threads,
                    std::numeric_limits<uint32_t>::max());
                maximum_physical_threads = std::max(
                    maximum_physical_threads,
                    static_cast<size_t>(physical_threads));
            }
            return encoder.stream()->rt_global_stack_buffer(
                maximum_physical_threads);
        }();
        RTStackArgs rt_args{
            .stack_size = stack_buf.stackSize,
            .stack_count = stack_buf.stackCount,
            .stack_data = stack_buf.stackData,
        };
        auto p_rt = allocate_argument(sizeof(RTStackArgs));
        std::memcpy(p_rt, &rt_args, sizeof(RTStackArgs));
    }

    auto arg_size = argument_buffer_offset;
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, argument_buffer.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END};
    auto launch = [&](uint3 dispatch_size, uint kernel_id) noexcept {
        if (any(dispatch_size == make_uint3(0u))) { return; }
        auto launch_info = make_uint4(dispatch_size, kernel_id);
        std::memcpy(launch_size_and_kernel_id, &launch_info, sizeof(launch_info));
        auto blocks = dispatch_size / block_size +
                      make_uint3(
                          dispatch_size.x % block_size.x != 0u,
                          dispatch_size.y % block_size.y != 0u,
                          dispatch_size.z % block_size.z != 0u);
        LUISA_CHECK_HIP(hipModuleLaunchKernel(
            _function,
            blocks.x, blocks.y, blocks.z,
            block_size.x, block_size.y, block_size.z,
            0u, hip_stream, nullptr, extra));
    };

    for (auto &&record : launch_records) {
        launch(record.dispatch_size, record.kernel_id);
    }
    printer_encode.commit(encoder);
}

}// namespace luisa::compute::hip
