// Test for the VK_NV_cuda_kernel_launch custom dispatch command.
// This test covers (no device required):
// - custom command UUID registration and to_string
// - stream tag / dispatch metadata of CudaKernelLaunchCommand
// - KernelLauncher argument packing order, uniform offsets and alignment
// - traverse_arguments coverage with stored usages
// - uniform blob roundtrip through ShaderDispatchCommandBase::uniform

#include "ut/ut.hpp"

#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/vk_cuda_interop.h>

#include <cstring>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::vk_cuda_interop;
using namespace boost::ut;
using namespace boost::ut::literals;

int main() {

    "uuid_and_metadata"_test = [] {
        KernelLauncher launcher;
        auto cmd = std::move(launcher).build(
            0xfeedu, uint3{4u, 2u, 1u}, uint3{128u, 1u, 1u}, 1024u);
        expect(cmd->custom_cmd_uuid() ==
               luisa::to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL));
        expect(luisa::to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL) == 0x0500u);
        expect(to_string(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL) ==
               luisa::string_view{"VK_CUDA_LAUNCH_KERNEL"});
        expect(cmd->stream_tag() == StreamTag::COMPUTE);
        expect(cmd->cuda_function() == 0xfeedu);
        expect(luisa::all(cmd->grid_dim() == uint3{4u, 2u, 1u}));
        expect(luisa::all(cmd->block_dim() == uint3{128u, 1u, 1u}));
        // max_dispatch_size reports threads (grid * block), matching the
      // command-reorder budget's units.
      expect(luisa::all(cmd->max_dispatch_size() == uint3{512u, 2u, 1u}));
        expect(cmd->shared_mem_bytes() == 1024u);
        expect(cmd->requires_resource_state_isolation());
        expect(cmd->arguments().empty());
    };

    "argument_packing_order_and_offsets"_test = [] {
        KernelLauncher launcher;
        launcher.add_buffer(0x1000u, 32u, 256u, Usage::READ_WRITE)
            .add_uniform(2.5f)
            .add_texture(0x2000u, 1u, Usage::READ)
            .add_uniform(7u);
        auto cmd = std::move(launcher).build(
            1u, uint3{1u}, uint3{64u}, 0u);

        auto args = cmd->arguments();
        expect(args.size() == 4u);

        // Argument order is preserved.
        expect(args[0].tag == Argument::Tag::BUFFER);
        expect(args[0].buffer.handle == 0x1000u);
        expect(args[0].buffer.offset == 32u);
        expect(args[0].buffer.size == 256u);
        expect(args[1].tag == Argument::Tag::UNIFORM);
        expect(args[2].tag == Argument::Tag::TEXTURE);
        expect(args[2].texture.handle == 0x2000u);
        expect(args[2].texture.level == 1u);
        expect(args[3].tag == Argument::Tag::UNIFORM);

        // Uniforms live in the blob after the argument header, in order.
        const auto header_size = 4u * sizeof(Argument);
        expect(args[1].uniform.offset == header_size);
        expect(args[1].uniform.size == sizeof(float));
        expect(args[1].uniform.alignment == alignof(float));
        expect(args[3].uniform.offset == header_size + sizeof(float));
        expect(args[3].uniform.size == sizeof(uint32_t));

        // Uniform blob roundtrip.
        float a = 0.f;
        uint32_t n = 0u;
        auto span_a = cmd->uniform(args[1].uniform);
        auto span_n = cmd->uniform(args[3].uniform);
        expect(span_a.size_bytes() == sizeof(float));
        expect(span_n.size_bytes() == sizeof(uint32_t));
        std::memcpy(&a, span_a.data(), sizeof(float));
        std::memcpy(&n, span_n.data(), sizeof(uint32_t));
        expect(a == 2.5f);
        expect(n == 7u);
    };

    "uniform_alignment_padding"_test = [] {
        KernelLauncher launcher;
        launcher.add_uniform(uint8_t{0xab})
            .add_uniform(uint64_t{0x0123456789abcdefull});
        auto cmd = std::move(launcher).build(1u, uint3{1u}, uint3{1u}, 0u);
        auto args = cmd->arguments();
        expect(args.size() == 2u);
        const auto header_size = 2u * sizeof(Argument);
        // The uint64 uniform must be aligned to 8 within the blob.
        expect(args[0].uniform.offset == header_size);
        expect(args[1].uniform.offset == header_size + 8u);
        uint8_t small = 0u;
        uint64_t big = 0u;
        std::memcpy(&small, cmd->uniform(args[0].uniform).data(), sizeof(small));
        std::memcpy(&big, cmd->uniform(args[1].uniform).data(), sizeof(big));
        expect(small == 0xab);
        expect(big == 0x0123456789abcdefull);
    };

    "traverse_arguments_coverage"_test = [] {
        KernelLauncher launcher;
        launcher.add_buffer(0x1000u, 0u, 128u, Usage::READ)
            .add_uniform(1.0f)
            .add_buffer(0x2000u, 64u, 64u, Usage::WRITE)
            .add_texture(0x3000u, 2u, Usage::READ_WRITE);
        auto cmd = std::move(launcher).build(1u, uint3{1u}, uint3{1u}, 0u);

        struct Visit {
            Argument::Tag tag;
            uint64_t handle;
            Usage usage;
        };
        luisa::vector<Visit> visited;
        cmd->traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                visited.emplace_back(Visit{Argument::Tag::BUFFER, arg.handle, usage});
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                visited.emplace_back(Visit{Argument::Tag::TEXTURE, arg.handle, usage});
            } else if constexpr (std::is_same_v<T, Argument::BindlessArray>) {
                visited.emplace_back(Visit{Argument::Tag::BINDLESS_ARRAY, arg.handle, usage});
            } else {
                visited.emplace_back(Visit{Argument::Tag::ACCEL, arg.handle, usage});
            }
        });

        // Uniforms are skipped; every resource argument is visited exactly
        // once, in order, with the stored usage.
        expect(visited.size() == 3u);
        expect(visited[0].tag == Argument::Tag::BUFFER);
        expect(visited[0].handle == 0x1000u);
        expect(visited[0].usage == Usage::READ);
        expect(visited[1].tag == Argument::Tag::BUFFER);
        expect(visited[1].handle == 0x2000u);
        expect(visited[1].usage == Usage::WRITE);
        expect(visited[2].tag == Argument::Tag::TEXTURE);
        expect(visited[2].handle == 0x3000u);
        expect(visited[2].usage == Usage::READ_WRITE);

        // The parallel usage array holds exactly the non-uniform arguments.
        expect(cmd->argument_usages().size() == 3u);
    };
}
