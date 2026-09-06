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

    "typed_kernel_api_packing"_test = [] {
        // The DSL-style CudaKernelInvoke must pack arguments identically to
        // the manual KernelLauncher path.
        auto manual = [] {
            KernelLauncher launcher;
            launcher.add_buffer(0x1000u, 0u, 256u, Usage::READ)
                .add_buffer(0x2000u, 64u, 128u, Usage::WRITE)
                .add_uniform(2.5f)
                .add_uniform(7u);
            return std::move(launcher).build(
                0xbeefu, uint3{4u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);
        }();

        // Untyped kernel with usage-tagged raw handles via BufferView-less
        // add_buffer equivalents: exercise the ratchet with synthetic views
        // is not possible without a device, so drive CudaKernelInvoke with
        // scalar uniforms and compare against add_uniform order; resource
        // encoding is covered by the helpers below through KernelLauncher.
        CudaKernelInvoke invoke{0xbeefu};
        invoke << 2.5f << 7u;
        auto typed = std::move(invoke).dispatch(
            uint3{4u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);

        // Uniform-only packing must match the trailing uniforms of manual.
        expect(typed->cuda_function() == 0xbeefu);
        expect(luisa::all(typed->grid_dim() == uint3{4u, 1u, 1u}));
        expect(luisa::all(typed->block_dim() == uint3{256u, 1u, 1u}));
        auto typed_args = typed->arguments();
        expect(typed_args.size() == 2u);
        expect(typed_args[0].tag == Argument::Tag::UNIFORM);
        expect(typed_args[1].tag == Argument::Tag::UNIFORM);
        float a = 0.f;
        uint32_t n = 0u;
        std::memcpy(&a, typed->uniform(typed_args[0].uniform).data(), sizeof(a));
        std::memcpy(&n, typed->uniform(typed_args[1].uniform).data(), sizeof(n));
        expect(a == 2.5f);
        expect(n == 7u);

        // The manual command's uniforms must carry the same values, proving
        // the packing convention is identical.
        auto manual_args = manual->arguments();
        float ma = 0.f;
        uint32_t mn = 0u;
        std::memcpy(&ma, manual->uniform(manual_args[2].uniform).data(), sizeof(ma));
        std::memcpy(&mn, manual->uniform(manual_args[3].uniform).data(), sizeof(mn));
        expect(ma == a);
        expect(mn == n);
    };

    "typed_kernel_dispatch_thread_count"_test = [] {
        // dispatch(thread_count_x, block) computes grid = ceil(tc / block.x).
        CudaKernelInvoke invoke{1u};
        invoke << 1u;
        auto cmd = std::move(invoke).dispatch(1000u, uint3{256u, 1u, 1u});
        expect(luisa::all(cmd->grid_dim() == uint3{4u, 1u, 1u}));
        expect(luisa::all(cmd->block_dim() == uint3{256u, 1u, 1u}));

        // Exact multiples must not add an extra block.
        CudaKernelInvoke invoke2{1u};
        invoke2 << 1u;
        auto cmd2 = std::move(invoke2).dispatch(1024u, uint3{256u, 1u, 1u});
        expect(luisa::all(cmd2->grid_dim() == uint3{4u, 1u, 1u}));
    };

    "cuda_kernel_operator_call"_test = [] {
        // The untyped CudaKernel operator() fold must produce the same
        // command as feeding CudaKernelInvoke directly.
        CudaKernel kernel{0xcafeu};
        auto cmd = kernel(1.5f, 3u).dispatch(uint3{1u}, uint3{64u});
        expect(cmd->cuda_function() == 0xcafeu);
        auto args = cmd->arguments();
        expect(args.size() == 2u);
        expect(args[0].tag == Argument::Tag::UNIFORM);
        expect(args[1].tag == Argument::Tag::UNIFORM);
        float f = 0.f;
        uint32_t u = 0u;
        std::memcpy(&f, cmd->uniform(args[0].uniform).data(), sizeof(f));
        std::memcpy(&u, cmd->uniform(args[1].uniform).data(), sizeof(u));
        expect(f == 1.5f);
        expect(u == 3u);
    };

    "typed_cuda_kernel_signature"_test = [] {
        // Typed kernel: usage is declared in the signature (CudaArg<T, U>, or
        // READ_WRITE for a bare resource type); call sites pass bare views,
        // and scalars pass by const reference.
        using Kernel = CudaKernelT<CudaArg<Buffer<float>, Usage::READ>,
                                   CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                                   float, uint32_t>;
        static_assert(Kernel::arg_count() == 4u);
        static_assert(std::is_same_v<cuda_arg_traits_t<Buffer<float>>::view_type,
                                     BufferView<float>>);
        static_assert(std::is_same_v<cuda_arg_traits_t<ByteBuffer>::view_type,
                                     ByteBufferView>);
        static_assert(std::is_same_v<cuda_arg_traits_t<Image<float>>::view_type,
                                     ImageView<float>>);
        static_assert(std::is_same_v<cuda_arg_traits_t<Volume<float>>::view_type,
                                     VolumeView<float>>);
        static_assert(std::is_same_v<cuda_arg_traits_t<BufferView<float>>::view_type,
                                     BufferView<float>>);
        // Bare resource types keep compiling and default to READ_WRITE.
        static_assert(cuda_arg_traits_t<Buffer<float>>::usage == Usage::READ_WRITE);
        static_assert(cuda_arg_traits_t<Buffer<float>>::is_resource);
        static_assert(cuda_arg_traits_t<CudaArg<Buffer<float>, Usage::READ>>::usage == Usage::READ);
        static_assert(cuda_arg_traits_t<CudaArg<Buffer<float>, Usage::WRITE>>::usage == Usage::WRITE);
        // Uniforms carry no resource type; they pass by const reference.
        static_assert(!cuda_arg_traits_t<float>::is_resource);
        static_assert(std::is_same_v<cuda_arg_traits_t<float>::arg_type, const float &>);
        static_assert(std::is_same_v<cuda_arg_traits_t<CudaArg<Buffer<float>, Usage::READ>>::arg_type,
                                     BufferView<float>>);

        // argument_usages() holds exactly the resource arguments' baked
        // usages, in argument order.
        static constexpr auto usages = Kernel::argument_usages();
        static_assert(usages.size() == 2u);
        static_assert(usages[0] == Usage::READ);
        static_assert(usages[1] == Usage::READ_WRITE);

        // ResourceArg accepts bare views (READ_WRITE default) and
        // usage-tagged views (untyped path).
        ResourceArg<BufferView<float>> bare{BufferView<float>{}};
        expect(bare.usage == Usage::READ_WRITE);
        ResourceArg<BufferView<float>> tagged{
            UsageArg<BufferView<float>>{BufferView<float>{}, Usage::READ}};
        expect(tagged.usage == Usage::READ);
    };

    "typed_kernel_baked_usage"_test = [] {
        // The usage baked into the typed signature (not the call site) must
        // reach the command's argument-usages array. Default-constructed
        // views suffice: only handle/usage packing is checked.
        using Kernel = CudaKernelT<CudaArg<Buffer<float>, Usage::READ>,
                                   CudaArg<Buffer<uint>, Usage::WRITE>,
                                   float>;
        Kernel kernel{0xba5eu};
        auto cmd = kernel(BufferView<float>{}, BufferView<uint>{}, 2.5f)
                       .dispatch(uint3{1u}, uint3{64u});
        expect(cmd->cuda_function() == 0xba5eu);

        // Recorded usages equal the baked ones, in resource-argument order.
        expect(cmd->argument_usages().size() == 2u);
        expect(cmd->argument_usages()[0] == Usage::READ);
        expect(cmd->argument_usages()[1] == Usage::WRITE);

        struct Visit {
            Argument::Tag tag;
            Usage usage;
        };
        luisa::vector<Visit> visited;
        cmd->traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                visited.emplace_back(Visit{Argument::Tag::BUFFER, usage});
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                visited.emplace_back(Visit{Argument::Tag::TEXTURE, usage});
            }
        });
        expect(visited.size() == 2u);
        expect(visited[0].tag == Argument::Tag::BUFFER);
        expect(visited[0].usage == Usage::READ);
        expect(visited[1].tag == Argument::Tag::BUFFER);
        expect(visited[1].usage == Usage::WRITE);

        // The uniform is packed by value after the resources.
        auto args = cmd->arguments();
        expect(args.size() == 3u);
        expect(args[2].tag == Argument::Tag::UNIFORM);
        float a = 0.f;
        std::memcpy(&a, cmd->uniform(args[2].uniform).data(), sizeof(a));
        expect(a == 2.5f);

        // A bare resource signature defaults to READ_WRITE (back-compat).
        using DefaultKernel = CudaKernelT<Buffer<float>>;
        DefaultKernel default_kernel{1u};
        auto default_cmd = default_kernel(BufferView<float>{})
                               .dispatch(uint3{1u}, uint3{64u});
        expect(default_cmd->argument_usages().size() == 1u);
        expect(default_cmd->argument_usages()[0] == Usage::READ_WRITE);
    };
}

