// Test for the "command owns its blob" design of the
// VK_NV_cuda_kernel_launch custom dispatch command. No device required.
//
// The contract under test is that CudaKernelLaunchCommand's on-wire argument
// region -- [Argument records][uniform payload], with uniform offsets relative
// to the end of the exact argument header -- is BYTE-IDENTICAL to the layout the
// previous (externally-owned) KernelLauncher produced, while the whole launch now
// costs one allocation instead of three:
//
//   (a) default launcher (no reserve())            -> growth + fix-up path
//   (b) reserve()-exact launcher (typed launches)  -> zero-growth fast path
//   (c) over-reserved / under-reserved growth      -> block growth + fix-up
//   (d) a moved command (construct -> std::move)   -> the relocation hazard
//
// (b) additionally asserts the fused layout itself (the blob starts at the
// object slot of the same allocation), which is the allocation-count evidence
// that holds without perturbing the allocator; the optional
// LC_VK_CUDA_TEST_COUNT_ALLOCATIONS build replaces global operator new/delete to count
// the allocations of a typed launch directly.
//
// Style mirrors src/tests/unit/ext/test_vk_cuda_launch_command.cpp.

#include "ut/ut.hpp"

#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/vk_cuda_interop.h>
#include <luisa/core/logging.h>
  #include <cstdlib>
  #include <cstring>

#ifdef LC_VK_CUDA_TEST_COUNT_ALLOCATIONS
  #include <atomic>
  #include <new>
#endif

  // Opt-in allocation accounting
// Opt-in allocation accounting (build with -DLC_VK_CUDA_TEST_COUNT_ALLOCATIONS): counts
// global operator new calls, which is what the fused block (and luisa::make_unique)
// use. luisa::vector / eastl containers allocate through the EASTL allocator and
// are only counted if that allocator routes through operator new, so this counter
// is corroborating evidence, not the primary gate -- the primary evidence is the
// pointer-containment assertion in "reserve_exact_packing_and_fused_layout", which
// proves the blob lives in the command's own allocation and therefore that no
// second allocation can exist for it.
#ifdef LC_VK_CUDA_TEST_COUNT_ALLOCATIONS
namespace {
inline std::atomic<size_t> g_new_calls{0u};
}// namespace

void *operator new(size_t n) {
    ++g_new_calls;
    auto *p = std::malloc(n == 0u ? 1u : n);
    if (p == nullptr) { LUISA_ERROR("operator new failed to allocate {} bytes", n); }
    return p;
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, size_t) noexcept { std::free(p); }
void *operator new[](size_t n) { return ::operator new(n); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }
void *operator new(size_t n, const std::nothrow_t &) noexcept {
    ++g_new_calls;
    return std::malloc(n == 0u ? 1u : n);
}
void operator delete(void *p, const std::nothrow_t &) noexcept { std::free(p); }
void *operator new[](size_t n, const std::nothrow_t &) noexcept { return ::operator new(n, std::nothrow); }
void operator delete[](void *p, const std::nothrow_t &) noexcept { ::operator delete(p); }
#endif

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::vk_cuda_interop;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto arg_size = sizeof(Argument);
constexpr auto header_of = [](size_t n) noexcept { return n * arg_size; };

  [[maybe_unused]] auto to_bytes(luisa::span<const Argument> records,
                                 luisa::span<const std::byte> payload) noexcept {
      // The expected on-wire region, described independently of the launcher: the
      // record array followed by the payload, gaps zeroed (as vector::resize did).
      luisa::vector<std::byte> out(records.size_bytes() + payload.size_bytes());
      if (!records.empty()) { std::memcpy(out.data(), records.data(), records.size_bytes()); }
      if (!payload.empty()) { std::memcpy(out.data() + records.size_bytes(), payload.data(), payload.size_bytes()); }
      return out;
  }

  [[maybe_unused]] bool blob_equals(const CudaKernelLaunchCommand &cmd,
                                    const luisa::vector<std::byte> &expected) noexcept {
      auto blob = cmd.argument_blob();
      return blob.size_bytes() == expected.size() &&
             (blob.empty() || std::memcmp(blob.data(), expected.data(), expected.size()) == 0);
  }

// The Params blob cuda_launch_kernel() packs: a 16-byte-aligned slot per
// argument (a buffer is an LCBuffer{ptr,size} binding), then the uint4 ls_kid
// trailer. Written here independently of the command's cache.
[[maybe_unused]] uint32_t expected_params_bytes(luisa::span<const Argument> args) noexcept {
    auto bytes = static_cast<size_t>(0u);
    for (auto &&arg : args) {
        auto n = arg.tag == Argument::Tag::UNIFORM ? arg.uniform.size : static_cast<size_t>(2u * sizeof(uint64_t));
        bytes = (bytes + 15u) & ~static_cast<size_t>(15u);
        bytes += n;
    }
    bytes = (bytes + 15u) & ~static_cast<size_t>(15u);
    return static_cast<uint32_t>(bytes + sizeof(uint4));
}

// The canonical 11-argument shape (radix one-sweep: 8 buffers + 3 by-value
// indices), the shape used to measure the recording allocation count.
template<typename Launcher>
void fill_radix_shape(Launcher &launcher) noexcept {
    launcher.add_buffer(0x1000u, 0u, 1024u, Usage::READ_WRITE)
        .add_buffer(0x1100u, 0u, 1024u, Usage::READ_WRITE)
        .add_buffer(0x1200u, 0u, 1024u, Usage::READ)
        .add_buffer(0x1300u, 0u, 1024u, Usage::WRITE)
        .add_uniform(3u)
        .add_buffer(0x1400u, 64u, 512u, Usage::READ)
        .add_buffer(0x1500u, 64u, 512u, Usage::WRITE)
        .add_uniform(7u)
        .add_buffer(0x1600u, 0u, 256u, Usage::READ_WRITE)
        .add_uniform(static_cast<uint64_t>(0x0123456789abcdefull))
        .add_buffer(0x1700u, 0u, 256u, Usage::READ_WRITE);
}

}// namespace

int main() {

          // (a) default launcher: growth + fix-up must still produce the reference bytes.
    "default_launcher_packing"_test = [] {
        KernelLauncher launcher;
        launcher.add_buffer(0x1000u, 32u, 256u, Usage::READ_WRITE)
            .add_uniform(2.5f)
            .add_texture(0x2000u, 1u, Usage::READ)
            .add_uniform(7u);
        auto cmd = std::move(launcher).build(1u, uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u}, 0u);

        auto args = cmd->arguments();
        expect(args.size() == 4u);
        const auto header = header_of(4u);
        expect(args[0].tag == Argument::Tag::BUFFER && args[0].buffer.handle == 0x1000u &&
               args[0].buffer.offset == 32u && args[0].buffer.size == 256u);
        expect(args[1].tag == Argument::Tag::UNIFORM);
        expect(args[2].tag == Argument::Tag::TEXTURE && args[2].texture.handle == 0x2000u &&
               args[2].texture.level == 1u);
        expect(args[3].tag == Argument::Tag::UNIFORM);
        expect(args[1].uniform.offset == header);
        expect(args[1].uniform.size == sizeof(float));
        expect(args[1].uniform.alignment == alignof(float));
        expect(args[3].uniform.offset == header + sizeof(float));

      luisa::vector<Argument> records(4u);
      // eastl vector(n) does not zero trivial storage; the expected bytes must
      // be fully defined (padding zero, like the launcher's records) before the
      // field-wise assignments below, or the memcmp compares uninitialized bytes.
      std::memset(records.data(), 0, records.size() * sizeof(Argument));
      records[0].tag = Argument::Tag::BUFFER;
        records[0].buffer = Argument::Buffer{0x1000u, 32u, 256u};
        records[1].tag = Argument::Tag::UNIFORM;
        records[1].uniform = Argument::Uniform{header, sizeof(float), alignof(float)};
        records[2].tag = Argument::Tag::TEXTURE;
                  // Field-wise on purpose: a whole-struct assignment would copy the
          // temporary's indeterminate trailing padding into the record, while
          // the launcher zeroes it (see KernelLauncher::add_texture).
          records[2].texture.handle = 0x2000u;
          records[2].texture.level = 1u;
        records[3].tag = Argument::Tag::UNIFORM;
        records[3].uniform = Argument::Uniform{header + sizeof(float), sizeof(uint32_t), alignof(uint32_t)};
        luisa::vector<std::byte> payload(sizeof(float) + sizeof(uint32_t));
        auto f = 2.5f;
        auto n = 7u;
        std::memcpy(payload.data(), &f, sizeof(f));
        std::memcpy(payload.data() + sizeof(f), &n, sizeof(n));
        expect(blob_equals(*cmd, to_bytes(records, payload)));// byte-identical on-wire region

        auto usages = cmd->argument_usages();
        expect(usages.size() == 2u && usages[0] == Usage::READ_WRITE && usages[1] == Usage::READ);
        expect(cmd->params_bytes() == expected_params_bytes(args));
        expect(cmd->argument_count() == 4u);
    };

    // alignment padding inside the payload is zeroed, not garbage (the fused
    // block is not value-initialized like a vector resize would be).
    "uniform_padding_is_zeroed"_test = [] {
        KernelLauncher launcher;
        launcher.add_uniform(uint8_t{0xabu}).add_uniform(uint64_t{0x0123456789abcdefull});
        auto cmd = std::move(launcher).build(1u, uint3{1u, 1u, 1u}, uint3{1u, 1u, 1u}, 0u);
        auto args = cmd->arguments();
        const auto header = header_of(2u);
        expect(args.size() == 2u);
        expect(args[0].uniform.offset == header && args[0].uniform.size == 1u);
        expect(args[1].uniform.offset == header + 8u && args[1].uniform.size == 8u);
        expect(cmd->argument_blob().size_bytes() == header + 16u);
        auto blob = cmd->argument_blob();
        // the 7 padding bytes between the two uniforms must be zero
        auto padding_zero = true;
        for (auto i = 0u; i < 7u; ++i) {
            padding_zero &= (blob[header + 1u + i] == std::byte{0});
        }
        expect(padding_zero);
        uint64_t big = 0u;
        std::memcpy(&big, cmd->uniform(args[1].uniform).data(), sizeof(big));
        expect(big == 0x0123456789abcdefull);
    };

    // (b) reserve()-exact: the typed-launch fast path. Same bytes as (a)'s
    // growth path for the identical shape, and the blob lives inside the
    // command's own allocation.
    "reserve_exact_packing_and_fused_layout"_test = [] {
        // reference: identical shape through the growth path
        KernelLauncher plain;
        fill_radix_shape(plain);
        auto reference = std::move(plain).build(0xbeefu, uint3{1024u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);

        KernelLauncher launcher;
        launcher.reserve(11u, 8u, 16u * 3u);// 11 records, 8 resources, 48 payload bytes
        fill_radix_shape(launcher);
        auto cmd = std::move(launcher).build(0xbeefu, uint3{1024u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);

        expect(cmd->argument_blob().size_bytes() == reference->argument_blob().size_bytes());
        expect(std::memcmp(cmd->argument_blob().data(), reference->argument_blob().data(),
                           cmd->argument_blob().size_bytes()) == 0);
        auto args = cmd->arguments();
        expect(args.size() == 11u);
        const auto header = header_of(11u);
        // header + 4 + 4 + (pad to 8) + 8
        expect(args[4].uniform.offset == header && args[4].uniform.size == 4u);
        expect(args[7].uniform.offset == header + 4u && args[7].uniform.size == 4u);
        expect(args[9].uniform.offset == header + 8u && args[9].uniform.size == 8u);
        expect(cmd->argument_blob().size_bytes() == header + 16u);
        expect(cmd->argument_usages().size() == 8u);
        expect(cmd->params_bytes() == 192u);// 8 * 16 + 3 uniforms in 16-byte slots + ls_kid
        expect(cmd->params_bytes() == expected_params_bytes(args));

        // The fused layout: the blob starts at the object slot of the very
        // allocation the object lives in => there is no second allocation for it.
        auto *object = reinterpret_cast<const std::byte *>(cmd.get());
        expect(cmd->argument_blob().data() == object + CudaKernelLaunchCommand::object_slot_bytes());
        expect(cmd->argument_blob().size_bytes() + CudaKernelLaunchCommand::object_slot_bytes() <=
               CudaKernelLaunchCommand::object_slot_bytes() + cmd->argument_blob().size_bytes());
        expect(reinterpret_cast<uintptr_t>(cmd->argument_usages().data()) >=
               reinterpret_cast<uintptr_t>(cmd->argument_blob().data() + cmd->argument_blob().size_bytes()));
        expect(CudaKernelLaunchCommand::object_slot_bytes() % CudaKernelLaunchCommand::fused_alignment == 0u);
        // The base class' storage is intentionally empty (shadowing contract):
        // reading through ShaderDispatchCommandBase faults instead of silently
        // returning nothing.
        auto &base = static_cast<const ShaderDispatchCommandBase &>(*cmd);
        expect(base.arguments().data() == nullptr);
        expect(base.arguments().size() == cmd->argument_count());
    };

    // (c) over/under-reserved: the block grows and build() fixes the offsets up;
    // the bytes must still match the plain growth path.
    "grown_header_packing_matches"_test = [] {
        KernelLauncher plain;
        fill_radix_shape(plain);
        auto reference = std::move(plain).build(0xfeedu, uint3{512u, 1u, 1u}, uint3{128u, 1u, 1u}, 0u);

        KernelLauncher launcher;
        launcher.reserve(1u, 1u, 0u);// deliberately too small on every axis
        fill_radix_shape(launcher);
        auto cmd = std::move(launcher).build(0xfeedu, uint3{512u, 1u, 1u}, uint3{128u, 1u, 1u}, 0u);

        expect(cmd->argument_blob().size_bytes() == reference->argument_blob().size_bytes());
        expect(std::memcmp(cmd->argument_blob().data(), reference->argument_blob().data(),
                           cmd->argument_blob().size_bytes()) == 0);
        expect(cmd->argument_usages().size() == reference->argument_usages().size());
        auto n = cmd->argument_usages().size();
        auto same_usages = n == reference->argument_usages().size();
        for (decltype(n) i = 0u; i < n; ++i) {
            same_usages &= (cmd->argument_usages()[i] == reference->argument_usages()[i]);
        }
        expect(same_usages);
    };

          // the reserve() contract, restated at byte level: a pre-sized header
    // must not shift a uniform offset.
    "reserve_preserves_uniform_offsets"_test = [] {
        KernelLauncher launcher;
        launcher.reserve(3u, 2u, 64u);
        launcher.add_buffer(0x1u, 0u, 8u, Usage::READ)
            .add_uniform(1.5f)
            .add_buffer(0x2u, 0u, 8u, Usage::WRITE);
        auto cmd = std::move(launcher).build(1u, uint3{1u, 1u, 1u}, uint3{32u, 1u, 1u}, 0u);
        auto args = cmd->arguments();
        const auto header = header_of(3u);
        expect(args.size() == 3u);
        expect(args[1].tag == Argument::Tag::UNIFORM && args[1].uniform.offset == header);
        float v = 0.f;
        std::memcpy(&v, cmd->uniform(args[1].uniform).data(), sizeof(v));
        expect(v == 1.5f);
        auto usages = cmd->argument_usages();
        expect(usages.size() == 2u && usages[0] == Usage::READ && usages[1] == Usage::WRITE);
    };

    // an empty launcher still builds a valid (zero-argument) command
    "empty_launcher"_test = [] {
        KernelLauncher launcher;
        auto cmd = std::move(launcher).build(0xfeedu, uint3{512u, 2u, 1u}, uint3{128u, 1u, 1u}, 1024u);
        expect(cmd->arguments().empty());
        expect(cmd->argument_blob().empty());
        expect(cmd->argument_usages().empty());
        // just the ls_kid trailer
        expect(cmd->params_bytes() == sizeof(uint4));
        expect(cmd->shared_mem_bytes() == 1024u);
        expect(luisa::all(cmd->grid_dim() == uint3{4u, 2u, 1u}));
        expect(luisa::all(cmd->dispatch_size() == uint3{512u, 2u, 1u}));
    };

    // more resource arguments than the launcher stages inline: the heap fallback
    // must not change anything observable.
    "usage_overflow_packing"_test = [] {
        auto build = [](bool seed_reserve) {
            KernelLauncher launcher;
            if (seed_reserve) { launcher.reserve(20u, 20u, 16u); }
            for (auto i = 0u; i < 20u; ++i) {
                launcher.add_buffer(0x100u + i, i, 16u, i % 2u == 0u ? Usage::READ : Usage::WRITE);
            }
            launcher.add_uniform(9u);
            return std::move(launcher).build(7u, uint3{256u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);
        };
        auto a = build(false);
        auto b = build(true);
        expect(a->argument_blob().size_bytes() == b->argument_blob().size_bytes());
        expect(std::memcmp(a->argument_blob().data(), b->argument_blob().data(),
                           a->argument_blob().size_bytes()) == 0);
        expect(a->argument_usages().size() == 20u && b->argument_usages().size() == 20u);
        auto same = true;
        for (auto i = 0u; i < 20u; ++i) {
            same &= (a->argument_usages()[i] == b->argument_usages()[i]);
        }
        expect(same);
    };

    // (d) THE MOVE HAZARD: the fused blob is self-referential, so a member-wise
    // move would dangle. The move constructor relocates it; the moved object must
    // encode exactly like a freshly built one, and the moved-from object must be
    // empty rather than dangling.
    "moved_command_relocates_blob"_test = [] {
        KernelLauncher launcher;
        fill_radix_shape(launcher);
        auto cmd = std::move(launcher).build(0xbeefu, uint3{1024u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);

        KernelLauncher reference_launcher;
        reference_launcher.reserve(11u, 8u, 48u);
        fill_radix_shape(reference_launcher);
        auto reference = std::move(reference_launcher).build(
            0xbeefu, uint3{1024u, 1u, 1u}, uint3{256u, 1u, 1u}, 0u);

        auto *source_object = reinterpret_cast<const std::byte *>(cmd.get());
        auto source_blob = cmd->argument_blob().data();
        expect(source_blob == source_object + CudaKernelLaunchCommand::object_slot_bytes());

        CudaKernelLaunchCommand moved{std::move(*cmd)};

        // the destination no longer reads the source's allocation
        expect(moved.argument_blob().data() !=
               reinterpret_cast<const std::byte *>(&moved) + CudaKernelLaunchCommand::object_slot_bytes());
        expect(moved.argument_blob().data() != source_blob);
        // ... but it still describes the identical launch
        expect(moved.cuda_function() == 0xbeefu);
        expect(moved.argument_count() == 11u);
        expect(moved.arguments().size() == 11u);
        expect(moved.argument_blob().size_bytes() == reference->argument_blob().size_bytes());
        expect(std::memcmp(moved.argument_blob().data(), reference->argument_blob().data(),
                           moved.argument_blob().size_bytes()) == 0);
        expect(moved.argument_usages().size() == 8u);
        auto args = moved.arguments();
        for (decltype(args.size()) i = 0u; i < args.size(); ++i) {
            expect(args[i].tag == reference->arguments()[i].tag);
            if (args[i].tag == Argument::Tag::UNIFORM) {
                expect(args[i].uniform.offset == reference->arguments()[i].uniform.offset);
                expect(args[i].uniform.size == reference->arguments()[i].uniform.size);
                auto lhs = moved.uniform(args[i].uniform);
                auto rhs = reference->uniform(reference->arguments()[i].uniform);
                expect(lhs.size_bytes() == rhs.size_bytes());
                expect(std::memcmp(lhs.data(), rhs.data(), lhs.size_bytes()) == 0);
            } else if (args[i].tag == Argument::Tag::BUFFER) {
                expect(args[i].buffer.handle == reference->arguments()[i].buffer.handle);
                expect(args[i].buffer.offset == reference->arguments()[i].buffer.offset);
                expect(args[i].buffer.size == reference->arguments()[i].buffer.size);
            }
        }
        expect(moved.params_bytes() == reference->params_bytes());
        expect(luisa::all(moved.dispatch_size() == uint3{1024u, 1u, 1u}));
        expect(moved.stream_tag() == StreamTag::COMPUTE);
        expect(moved.custom_cmd_uuid() == luisa::to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL));

        // the moved-from source is empty, not dangling
        expect(cmd->arguments().empty());
        expect(cmd->argument_blob().empty());
        expect(cmd->argument_usages().empty());

        // a moved command still traverses for the barrier / reorder passes
        struct Visit {
            Argument::Tag tag;
            uint64_t handle;
            Usage usage;
        };
        luisa::vector<Visit> visited;
        moved.traverse_arguments([&]<typename T>(T const &arg, Usage usage) noexcept {
            if constexpr (std::is_same_v<T, Argument::Buffer>) {
                visited.emplace_back(Visit{Argument::Tag::BUFFER, arg.handle, usage});
            } else if constexpr (std::is_same_v<T, Argument::Texture>) {
                visited.emplace_back(Visit{Argument::Tag::TEXTURE, arg.handle, usage});
            }
        });
        expect(visited.size() == 8u);
        expect(visited[0].handle == 0x1000u && visited[0].usage == Usage::READ_WRITE);
        expect(visited[3].usage == Usage::WRITE);
        expect(visited[7].handle == 0x1700u);

        // ... and a relocated command stays correct through a second move (its
        // blob is heap-owned now, its usages live in that buffer's tail).
        CudaKernelLaunchCommand twice{std::move(moved)};
        expect(twice.argument_count() == 11u);
        expect(twice.argument_blob().size_bytes() == reference->argument_blob().size_bytes());
        expect(std::memcmp(twice.argument_blob().data(), reference->argument_blob().data(),
                           twice.argument_blob().size_bytes()) == 0);
        expect(twice.argument_usages().size() == 8u);
        auto twice_usages_match = twice.argument_usages().size() == reference->argument_usages().size();
        for (size_t i = 0u; twice_usages_match && i < reference->argument_usages().size(); ++i) {
            twice_usages_match &= (twice.argument_usages()[i] == reference->argument_usages()[i]);
        }
        expect(twice_usages_match);
        expect(moved.arguments().empty());
        expect(twice.params_bytes() == reference->params_bytes());
    };

          // the externally-owned constructor keeps working and reads back
    // the caller's buffer through the same accessors
    "externally_owned_command"_test = [] {
        KernelLauncher launcher;
        launcher.add_buffer(0x1000u, 0u, 64u, Usage::READ).add_uniform(1.25f);
        auto built = std::move(launcher).build(3u, uint3{64u, 1u, 1u}, uint3{64u, 1u, 1u}, 0u);

        luisa::vector<std::byte> buffer;
        buffer.resize(built->argument_blob().size_bytes());
        std::memcpy(buffer.data(), built->argument_blob().data(), buffer.size());
        luisa::vector<Usage> usages{Usage::READ};
        CudaKernelLaunchCommand external{3u, uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u}, 0u, uint3{64u, 1u, 1u},
                                        std::move(buffer), 2u, std::move(usages)};

        expect(external.arguments().size() == 2u);
        expect(external.argument_blob().size_bytes() == built->argument_blob().size_bytes());
        expect(std::memcmp(external.argument_blob().data(), built->argument_blob().data(),
                           external.argument_blob().size_bytes()) == 0);
        expect(external.argument_usages().size() == 1u && external.argument_usages()[0] == Usage::READ);
        expect(external.params_bytes() == built->params_bytes());

        // and it moves the ordinary way (its buffer is already external storage)
        CudaKernelLaunchCommand moved_external{std::move(external)};
        expect(moved_external.arguments().size() == 2u);
        expect(std::memcmp(moved_external.argument_blob().data(), built->argument_blob().data(),
                           moved_external.argument_blob().size_bytes()) == 0);
        expect(moved_external.argument_usages().size() == 1u);
        expect(external.arguments().empty());
    };

    // the typed (CudaKernelT) and untyped (CudaKernel) launch paths, which always
    // reserve(), must pack exactly like KernelLauncher
    "typed_launch_packing"_test = [] {        using Kernel = CudaKernelT<CudaArg<Buffer<float>, Usage::READ>,
                                   CudaArg<Buffer<uint>, Usage::READ_WRITE>,
                                   CudaArg<ByteBuffer, Usage::WRITE>,
                                   uint32_t, float>;
      Kernel kernel{0xba5eu};
      // Default-constructed views carry the invalid-resource sentinel handle
      // (~0ull), NOT 0; the reference launcher must encode the very same values,
      // so name the views and take handle()/offset_bytes()/size_bytes() from
      // them rather than hard-coding zeros.
      BufferView<float> in0{};
      BufferView<uint> in1{};
      ByteBufferView in2{};
      auto cmd = kernel(in0, in1, in2, 7u, 2.5f)
                     .dispatch(uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u});

      KernelLauncher launcher;
      launcher.add_buffer(in0.handle(), in0.offset_bytes(), in0.size_bytes(), Usage::READ)
          .add_buffer(in1.handle(), in1.offset_bytes(), in1.size_bytes(), Usage::READ_WRITE)
          .add_buffer(in2.handle(), in2.offset_bytes(), in2.size_bytes(), Usage::WRITE)
          .add_uniform(7u)
          .add_uniform(2.5f);
        auto reference = std::move(launcher).build(0xba5eu, uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u}, 0u);

        expect(cmd->arguments().size() == 5u);
        expect(cmd->argument_blob().size_bytes() == reference->argument_blob().size_bytes());
        expect(std::memcmp(cmd->argument_blob().data(), reference->argument_blob().data(),
                           cmd->argument_blob().size_bytes()) == 0);
        expect(cmd->argument_usages().size() == 3u);
        expect(cmd->argument_usages()[0] == Usage::READ &&
               cmd->argument_usages()[1] == Usage::READ_WRITE &&
               cmd->argument_usages()[2] == Usage::WRITE);
        auto *object = reinterpret_cast<const std::byte *>(cmd.get());
        expect(cmd->argument_blob().data() == object + CudaKernelLaunchCommand::object_slot_bytes());

        CudaKernel untyped{0xcafeu};
        auto c2 = untyped(1.5f, 3u).dispatch(uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u});
        auto args = c2->arguments();
        expect(args.size() == 2u);
        float f = 0.f;
        uint32_t n = 0u;
        std::memcpy(&f, c2->uniform(args[0].uniform).data(), sizeof(f));
        std::memcpy(&n, c2->uniform(args[1].uniform).data(), sizeof(n));
        expect(f == 1.5f && n == 3u);
        expect(c2->argument_usages().empty());
    };

    // a command handed to a command list travels as unique_ptr<Command>: the
    // object itself is never moved there, and delete-through-Command frees the
    // fused block (the offset-0 property the fused layout depends on).
    "ownership_and_layout"_test = [] {
        KernelLauncher launcher;
        launcher.reserve(2u, 2u, 8u);
        launcher.add_buffer(0x1u, 0u, 4u, Usage::READ).add_buffer(0x2u, 0u, 4u, Usage::WRITE);
        auto cmd = std::move(launcher).build(9u, uint3{1u, 1u, 1u}, uint3{1u, 1u, 1u}, 0u);
        auto *derived = cmd.get();
        luisa::unique_ptr<Command> base_cmd{std::move(cmd)};
        auto *as_command = base_cmd.get();
        expect(as_command == static_cast<Command *>(derived));// offset 0 => global delete frees the block
        auto *as_cuda = static_cast<CudaKernelLaunchCommand *>(as_command);
        expect(as_cuda->arguments().size() == 2u);
        expect(as_cuda->argument_blob().data() ==
               reinterpret_cast<const std::byte *>(derived) + CudaKernelLaunchCommand::object_slot_bytes());
    };

#ifdef LC_VK_CUDA_TEST_COUNT_ALLOCATIONS
    "one_allocation_per_typed_launch"_test = [] {
        using K = CudaKernelT<CudaArg<Buffer<float>, Usage::READ>,
                              CudaArg<Buffer<uint>, Usage::WRITE>,
                              uint32_t>;
        K kernel{1u};
        // warm up so one-time allocator/container init is not counted
        static_cast<void>(kernel(BufferView<float>{}, BufferView<uint>{}, 3u)
                              .dispatch(uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u}));
        auto before = g_new_calls.load();
        {
            auto cmd = kernel(BufferView<float>{}, BufferView<uint>{}, 3u)
                           .dispatch(uint3{1u, 1u, 1u}, uint3{64u, 1u, 1u});
            expect(cmd->argument_blob().data() ==
                   reinterpret_cast<const std::byte *>(cmd.get()) +
                       CudaKernelLaunchCommand::object_slot_bytes());
        }
          // exactly one allocation for the whole launch: command + blob + usages.
        expect(g_new_calls.load() - before == 1u);
    };
#endif
}

