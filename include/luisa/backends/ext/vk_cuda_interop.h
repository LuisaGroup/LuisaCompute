#pragma once
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>
#include <luisa/vstl/meta_lib.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/cuda/cuda_config_ext.h>
namespace luisa::compute {
class VkCudaInterop;
namespace vk_cuda_interop {
struct Signal {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct Wait {
    VkCudaInterop *ext;
    uint64_t handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};

// Custom dispatch command launching a CUDA kernel (imported from the cuda
// backend through VkCudaInterop::create_cuda_kernel_shader) inside a Vulkan
// command buffer.
//
// Imported shaders are DSL kernels compiled by the cuda backend, whose ABI is
// a single by-value `struct Params` parameter: every kernel argument occupies
// a 16-byte-aligned slot in the blob (buffers as `LCBuffer { ptr, size_bytes
// }` bindings, uniforms as raw values), followed by an `ls_kid` uint4 trailer
// carrying {dispatch_size.xyz, kernel_id}. The vk backend builds this blob at
// encode time from the command's Argument array, so the command additionally
// records the exact thread count (`_dispatch_size`, used for `ls_kid`) next
// to the grid/block geometry.
//
// The command owns its blob: KernelLauncher::build() creates
// the command and its argument blob in ONE allocation: the launcher stages the
// blob in a block that already reserves room for the command object at its
// front, and build() placement-news the command there. Nothing is copied on
// the fast path (a launcher that called reserve() with the exact launch shape):
//
//      0                     object_slot_bytes()                              block end
//   +---------------------+----------------------+----------------+-------+--------------+
//   | command object slot | Argument records     | uniform payload| slack | usage tail   |
//   | (sizeof + padding)  | (count * 32 bytes)   | (used bytes)   | (opt.)| (usages*4 B) |
//   +---------------------+----------------------+----------------+-------+--------------+
//   \______________ one allocator block (_alloc_block: eastl default allocator / ::operator new by build), 16-aligned ___/
//   ^ this == block base (the Command subobject is at offset 0; asserted)
//
// The on-wire region is [Argument records][uniform payload] -- byte-identical
// to what an externally-owned command's argument buffer holds, at the same offsets (see
// argument_blob() and the packing self-check); the usage records live in a
// tail beyond that region and are only addressed through argument_usages().
//
// Storage & accessors -- CONTRACT: the argument blob never lives in the base
// class' luisa::vector (ShaderDispatchCommandBase is always given an empty
// buffer by this class), so arguments()/uniform() here are SHADOWING accessors
// that read _blob_data/_blob_bytes. Read a CudaKernelLaunchCommand's arguments
// through CudaKernelLaunchCommand, never through a ShaderDispatchCommandBase
// reference: the base sees an empty buffer with a null data pointer, so such a
// read faults instead of silently returning nothing (the vk encode path's
// argument-count assert makes the same misuse loud). Every in-tree consumer
// (lc::vk::cuda_launch_kernel, the barrier recorder and the command-reorder
// pass) uses the derived accessors or traverse_arguments().
//
// Move semantics -- the blob is self-referential (it lives inside this object's
// own allocation), so a member-wise move would leave the destination pointing
// into the source's block. The move constructor therefore RELOCATES the used
// trailing region into a heap buffer owned by the destination (one allocation
// plus a memcpy) and leaves the source empty and safe. Commands travel into
// command lists behind luisa::unique_ptr (CommandList::CommandContainer is a
// vector of unique_ptr<Command>), so the object itself is never moved on the
// production path; the relocation exists to keep the type MoveConstructible
// and provably correct, and is covered by unit tests.
class CudaKernelLaunchCommand final : public CustomDispatchCommand, public ShaderDispatchCommandBase {

public:
    // Alignment of the fused block and of the blob inside it. It is a multiple
    // of alignof(Argument) (8) and matches what the luisa::detail::allocator_allocate
    // aligned path guarantees (>=16 on MSVC/x64, GCC and Clang), so the blob
    // is properly aligned without any over-aligned allocation. Uniform values
    // keep their blob-relative alignment, exactly as in the plain single-blob
    // layout (the DSL Params packing re-aligns to 16 at encode time).
    static constexpr size_t fused_alignment = 16u;

    // Bytes of the fused block occupied by the object itself; the blob starts
    // here. (A member function body is a complete-class context, so sizeof of
    // the enclosing type is legal -- a static data member initializer would not
    // be.)
    [[nodiscard]] static constexpr size_t object_slot_bytes() noexcept {
        return (sizeof(CudaKernelLaunchCommand) + fused_alignment - 1u) & ~static_cast<size_t>(fused_alignment - 1u);
    }

private:
    uint3 _grid_dim;
    uint3 _block_dim;
    uint32_t _shared_mem_bytes;
    // Exact number of threads to launch (<= grid * block), for ls_kid.
    uint3 _dispatch_size;
    size_t _argument_count;
    // [Argument records][uniform payload]: the on-wire region. Points into this
    // object's own allocation (build() path) or into _owned_blob (the
    // externally-owned ctor and a relocated move).
    std::byte *_blob_data;
    size_t _blob_bytes;
    // Per-resource usages, in argument order: points into the usage tail of the
    // fused block, into _argument_usages, or into _owned_blob after a move.
    const Usage *_usage_data;
    size_t _usage_count;
    // Trailing storage belonging to this allocation (0 for externally-owned
    // storage). Only the move constructor reads it: it is what has to be copied
    // out of the source's block.
    size_t _trailing_bytes;
    // Size of the DSL kernel ABI blob the vk backend packs at encode time (see
    // params_bytes()); cached here so encoding resizes its scratch buffer once.
    uint32_t _params_bytes;
    // Externally-owned storage: the form where the argument buffer is handed
    // to the public ctor and the destination of a relocated move. Empty on the fused
    // fast path -- an empty luisa::vector performs no allocation.
    luisa::vector<std::byte> _owned_blob;
    luisa::vector<Usage> _argument_usages;

    // The command's own trailing region (blob + slack + usage tail), i.e. what
    // belongs to this allocation rather than to a separate buffer.
    [[nodiscard]] std::byte *_trailing_storage() noexcept {
        return reinterpret_cast<std::byte *>(this) + object_slot_bytes();
    }
    [[nodiscard]] const std::byte *_trailing_storage() const noexcept {
        return reinterpret_cast<const std::byte *>(this) + object_slot_bytes();
    }

    // The size of the by-value Params blob cuda_launch_kernel() packs for this
    // command: every argument takes a 16-byte-aligned slot (a buffer is an
    // LCBuffer {ptr, size_bytes} binding = 16 bytes, a uniform its raw size),
    // followed by the uint4 `ls_kid` trailer. Mirrors allocate_argument()'s
    // "align the running end, then append" recurrence exactly.
    [[nodiscard]] static uint32_t _compute_params_bytes(luisa::span<const Argument> args) noexcept {
        auto bytes = static_cast<size_t>(0u);
        for (auto &&arg : args) {
            auto arg_bytes = arg.tag == Argument::Tag::UNIFORM ? arg.uniform.size : static_cast<size_t>(2u * sizeof(uint64_t));
            bytes = luisa::align(bytes, static_cast<size_t>(16u)) + arg_bytes;
        }
        bytes = luisa::align(bytes, static_cast<size_t>(16u)) + sizeof(uint4);
        constexpr auto max_u32 = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
        // Saturates: the encode path re-checks the scratch buffer's size.
        return bytes > max_u32 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(bytes);
    }

    static void _assert_geometry(uint3 grid_dim, uint3 block_dim, uint3 dispatch_size) noexcept {
        LUISA_ASSERT(grid_dim.x > 0u && grid_dim.y > 0u && grid_dim.z > 0u,
                     "CUDA kernel launch grid dimension must be nonzero.");
        LUISA_ASSERT(block_dim.x > 0u && block_dim.y > 0u && block_dim.z > 0u,
                     "CUDA kernel launch block dimension must be nonzero.");
        LUISA_ASSERT(dispatch_size.x > 0u && dispatch_size.y > 0u && dispatch_size.z > 0u,
                     "CUDA kernel launch dispatch size must be nonzero.");
        auto fits = [](uint32_t d, uint32_t g, uint32_t b) noexcept {
            return static_cast<uint64_t>(d) <=
                   static_cast<uint64_t>(g) * static_cast<uint64_t>(b);
        };
        LUISA_ASSERT(fits(dispatch_size.x, grid_dim.x, block_dim.x) &&
                         fits(dispatch_size.y, grid_dim.y, block_dim.y) &&
                         fits(dispatch_size.z, grid_dim.z, block_dim.z),
                     "CUDA kernel launch dispatch size exceeds grid * block.");
    }

    template<typename Self, typename Visitor>
    static void traverse(Self &&self, Visitor &visitor) noexcept {
        auto usages = self.argument_usages();
        auto usage_index = 0u;
        auto next_usage = [usages, &usage_index]() noexcept {
            LUISA_ASSERT(usage_index < usages.size(),
                         "CUDA kernel launch command has fewer argument usages than resource arguments.");
            return usages[usage_index++];
        };
        for (auto &&arg : self.arguments()) {
            switch (arg.tag) {
                case Argument::Tag::BUFFER:
                    visitor.visit(arg.buffer, next_usage());
                    break;
                case Argument::Tag::TEXTURE:
                    visitor.visit(arg.texture, next_usage());
                    break;
                case Argument::Tag::BINDLESS_ARRAY:
                    visitor.visit(arg.bindless_array, next_usage());
                    break;
                case Argument::Tag::ACCEL:
                    visitor.visit(arg.accel, next_usage());
                    break;
                case Argument::Tag::UNIFORM: break;
                default: break;
            }
        }
        LUISA_ASSERT(usage_index == usages.size(),
                     "CUDA kernel launch command has {} argument usages for the resource arguments.",
                     usages.size());
    }

public:
    // Epoch-6 (externally-owned) form, kept verbatim for source compatibility:
    // the caller's argument buffer and usage list are stored by the command and
    // the accessors read them from there. A hand-made command therefore still
    // costs the caller's own allocations and nothing extra.
    CudaKernelLaunchCommand(uint64_t cuda_function_handle,
                            uint3 grid_dim, uint3 block_dim,
                            uint32_t shared_mem_bytes,
                            uint3 dispatch_size,
                            luisa::vector<std::byte> &&argument_buffer,
                            size_t argument_count,
                            luisa::vector<Usage> &&argument_usages) noexcept
        : ShaderDispatchCommandBase{cuda_function_handle,
                                    luisa::vector<std::byte>{},
                                    argument_count},
          _grid_dim{grid_dim},
          _block_dim{block_dim},
          _shared_mem_bytes{shared_mem_bytes},
          _dispatch_size{dispatch_size},
          _argument_count{argument_count},
          _blob_data{nullptr},
          _blob_bytes{0u},
          _usage_data{nullptr},
          _usage_count{0u},
          _trailing_bytes{0u},
          _params_bytes{0u},
          _owned_blob{std::move(argument_buffer)},
          _argument_usages{std::move(argument_usages)} {
        _assert_geometry(grid_dim, block_dim, dispatch_size);
        _blob_data = _owned_blob.data();
        _blob_bytes = _owned_blob.size();
        _usage_data = _argument_usages.empty() ? nullptr : _argument_usages.data();
        _usage_count = _argument_usages.size();
        _params_bytes = _compute_params_bytes(arguments());
    }

    CudaKernelLaunchCommand(CudaKernelLaunchCommand const &) = delete;
    CudaKernelLaunchCommand &operator=(CudaKernelLaunchCommand const &) = delete;
    CudaKernelLaunchCommand &operator=(CudaKernelLaunchCommand &&) = delete;

    // Move: see the class comment. A fused command's blob travels by value into
    // a heap buffer owned by the destination; an externally-owned command's
    // buffers travel by moving their vectors. The source is left empty and safe.
    CudaKernelLaunchCommand(CudaKernelLaunchCommand &&rhs) noexcept
        : CustomDispatchCommand(std::move(rhs)),
          ShaderDispatchCommandBase(std::move(rhs)),
          _grid_dim{rhs._grid_dim},
          _block_dim{rhs._block_dim},
          _shared_mem_bytes{rhs._shared_mem_bytes},
          _dispatch_size{rhs._dispatch_size},
          _argument_count{rhs._argument_count},
          _blob_data{nullptr},
          _blob_bytes{rhs._blob_bytes},
          _usage_data{nullptr},
          _usage_count{rhs._usage_count},
          _trailing_bytes{0u},
          _params_bytes{rhs._params_bytes},
          // A fused source keeps neither vector in use (both are empty), so
          // moving them is allocation-free; an external source's buffers travel
          // here and their offsets are re-based below.
          _owned_blob{std::move(rhs._owned_blob)},
          _argument_usages{std::move(rhs._argument_usages)} {
        if (rhs._trailing_bytes > 0u) {
            // Fused storage: the blob lives inside the source's allocation and
            // cannot travel with a member-wise move. Relocate
            // [records][payload][slack][usage tail] into a heap buffer owned by
            // *this, then rebase both pointers into it.
            auto tail_offset = rhs._usage_data != nullptr
                                   ? static_cast<size_t>(rhs._usage_data -
                                                         reinterpret_cast<const Usage *>(rhs._trailing_storage()))
                                   : static_cast<size_t>(0);
            LUISA_DEBUG_ASSERT(tail_offset + rhs._usage_count * sizeof(Usage) <= rhs._trailing_bytes,
                               "CUDA kernel launch command: relocated usage tail is out of range.");
            _owned_blob.resize(rhs._trailing_bytes);
            std::memcpy(_owned_blob.data(), rhs._trailing_storage(), rhs._trailing_bytes);
            _blob_data = _owned_blob.data();
            _usage_data = _usage_count > 0u
                              ? reinterpret_cast<const Usage *>(_owned_blob.data()) + tail_offset
                              : nullptr;
        } else {
            // External storage (_owned_blob holds [records][payload], plus the
            // tail when this command was itself relocated from a fused block
            // earlier, in which case _argument_usages is empty).
            _blob_data = _owned_blob.data();
            if (!_argument_usages.empty()) {
                _usage_data = _argument_usages.data();
            } else if (_usage_count > 0u) {
                // Relocated source: its usages live in the buffer that the init
                // list above just MOVED into *this. rhs._owned_blob is empty now
                // (data() == nullptr), so the offset must be taken against
                // *this*'s storage - the very block the source's pointers still
                // describe.
                auto offset = static_cast<size_t>(
                    rhs._usage_data -
                    reinterpret_cast<const Usage *>(_owned_blob.data()));
                _usage_data = reinterpret_cast<const Usage *>(_owned_blob.data()) + offset;
            } else {
                _usage_data = nullptr;
            }
        }
        // Leave the source empty: its trailing region is *its* allocation and
        // must not be read through *this, and the moved-from base buffer is gone.
        rhs._argument_count = 0u;
        rhs._blob_data = nullptr;
        rhs._blob_bytes = 0u;
        rhs._usage_data = nullptr;
        rhs._usage_count = 0u;
        rhs._trailing_bytes = 0u;
    }

public:
    [[nodiscard]] uint64_t custom_cmd_uuid() const noexcept override {
        return luisa::to_underlying(CustomCommandUUID::VK_CUDA_LAUNCH_KERNEL);
    }
    [[nodiscard]] StreamTag stream_tag() const noexcept override {
        return StreamTag::COMPUTE;
    }
    [[nodiscard]] uint64_t cuda_function() const noexcept { return handle(); }
    [[nodiscard]] uint3 grid_dim() const noexcept { return _grid_dim; }
    [[nodiscard]] uint3 block_dim() const noexcept { return _block_dim; }
    [[nodiscard]] uint32_t shared_mem_bytes() const noexcept { return _shared_mem_bytes; }
    // The exact number of threads the kernel is launched with (the `ls_kid`
    // trailer payload); equals grid * block for explicit-geometry launches.
    [[nodiscard]] uint3 dispatch_size() const noexcept { return _dispatch_size; }
    [[nodiscard]] size_t argument_count() const noexcept { return _argument_count; }
    // The per-resource usages, in argument order.
    [[nodiscard]] luisa::span<const Usage> argument_usages() const noexcept {
        return luisa::span<const Usage>{_usage_data, _usage_count};
    }
    // The on-wire argument region: [Argument records][uniform payload], i.e. the
    // exact bytes an externally-owned command's argument buffer carries. Provided so the
    // command ABI can be checked byte-for-byte; the vk encode path reads the
    // typed accessors below instead.
    [[nodiscard]] luisa::span<const std::byte> argument_blob() const noexcept {
        return luisa::span<const std::byte>{_blob_data, _blob_bytes};
    }
    // Shadowing accessors (see the class comment): they read the fused blob when
    // there is one, otherwise the externally-owned buffer.
    [[nodiscard]] luisa::span<const Argument> arguments() const noexcept {
        return luisa::span<const Argument>{reinterpret_cast<const Argument *>(_blob_data), _argument_count};
    }
    [[nodiscard]] luisa::span<Argument> arguments() noexcept {
        return luisa::span<Argument>{reinterpret_cast<Argument *>(_blob_data), _argument_count};
    }
    [[nodiscard]] luisa::span<const std::byte> uniform(const Argument::Uniform &u) const noexcept {
        return luisa::span<const std::byte>{_blob_data, _blob_bytes}.subspan(u.offset, u.size);
    }
    // The size of the by-value DSL Params blob cuda_launch_kernel() packs from
    // these arguments, so the encode path can allocate its scratch buffer once
    // instead of re-growing it per argument. Saturated at UINT32_MAX; encoding
    // still bounds-checks.
    [[nodiscard]] uint32_t params_bytes() const noexcept { return _params_bytes; }
    // The reorder budget counts threads, so report the exact dispatch size.
    [[nodiscard]] uint3 max_dispatch_size() const noexcept override { return _dispatch_size; }
    // The reorder pass tracks the module's accesses through its declared
    // argument usages: a declared READ is a read-only contract (dispatches
    // sharing the range merge into one barrier-free layer), a declared
    // WRITE is exclusive over its range. This holds for raw (hand-written)
    // function handles too - a kernel that writes through a pointer it
    // declared READ violates its declaration and must not be launched
    // with it.
    void traverse_arguments(ArgumentVisitor &visitor) const noexcept override {
        traverse(*this, visitor);
    }
    void traverse_arguments(MutableArgumentVisitor &visitor) noexcept override {
        traverse(*this, visitor);
    }
    // Un-hide the generic-lambda traverse_arguments adapters.
    using CustomDispatchCommand::traverse_arguments;

private:
    // The fused-storage form, created only by KernelLauncher::build(): every
    // byte is already in place inside `block`, which *this constructor
    // placement-news into (see KernelLauncher).
    struct FusedBlob {
        std::byte *blob_data;      // [Argument records][uniform payload]
        size_t blob_bytes;         // logical size of that region
        Usage *usages;             // the usage tail (may be null when empty)
        size_t usage_count;
        size_t argument_count;
        size_t trailing_bytes;     // block bytes after the object slot
    };

    CudaKernelLaunchCommand(uint64_t cuda_function_handle,
                            uint3 grid_dim, uint3 block_dim,
                            uint32_t shared_mem_bytes,
                            uint3 dispatch_size,
                            const FusedBlob &blob) noexcept
        : ShaderDispatchCommandBase{cuda_function_handle,
                                    luisa::vector<std::byte>{},
                                    blob.argument_count},
          _grid_dim{grid_dim},
          _block_dim{block_dim},
          _shared_mem_bytes{shared_mem_bytes},
          _dispatch_size{dispatch_size},
          _argument_count{blob.argument_count},
          _blob_data{blob.blob_data},
          _blob_bytes{blob.blob_bytes},
          _usage_data{blob.usages},
          _usage_count{blob.usage_count},
          _trailing_bytes{blob.trailing_bytes},
          _params_bytes{0u},
          _owned_blob{},
          _argument_usages{} {
        _assert_geometry(grid_dim, block_dim, dispatch_size);
        // The object must sit at the block base: CommandList owns commands as
        // unique_ptr<Command> and destroys through that base pointer; the
        // deleter runs the (virtual) destructor and then frees the exact pointer
        // handed back - eastl's default_delete via GetDefaultAllocator()->
        // deallocate (size hint ignored) in EASTL builds, the delete expression
        // (global operator delete) in system-STL builds - so it must equal the
        // block base. Command is the primary base of
        // CustomDispatchCommand, hence at offset 0, but the layout of a
        // multiply-inheriting class is not something to assume.
        LUISA_DEBUG_ASSERT(static_cast<const void *>(static_cast<const Command *>(this)) ==
                               static_cast<const void *>(this),
                           "CUDA kernel launch command: the Command subobject must be at offset 0 of its "
                           "fused allocation (the command list deletes through Command*).");
        LUISA_DEBUG_ASSERT(_blob_data == _trailing_storage(),
                           "CUDA kernel launch command: the fused blob must start at the object slot.");
        _params_bytes = _compute_params_bytes(arguments());
    }

    friend class KernelLauncher;
};

// Header-only builder collecting buffer/texture/uniform arguments and packing
// them into a CudaKernelLaunchCommand. Uniform values are copied into the
// command; resource arguments are referenced by handle and must outlive the
// dispatch.
//
// At encode time the vk backend converts the argument list into the DSL
// kernel ABI blob: buffers become `LCBuffer { ptr = VkDeviceAddress + offset,
// size_bytes = size }` slots, so the order of add_* calls must match the
// imported kernel's parameter order exactly.
//
// The builder stages the blob in the very allocation that becomes
// the command (see CudaKernelLaunchCommand), so building a launch costs ONE
// heap allocation total - argument records, uniform payload, usage list and
// command object together - with no second copy. reserve() sizes that block up
// front from the compile-time launch shape; a typed CudaKernelT / CudaKernel
// launch always calls it, so it hits the zero-growth, zero-fix-up, zero-copy
// path. A launcher that never calls reserve() keeps working: the block grows on
// demand (same doubling as before) and build() rotates the uniform payload down
// behind the exact argument header and patches the recorded offsets, producing
// byte-identical on-wire bytes either way - which is what the packing
// self-check in the vk-interop tests guards.
class KernelLauncher {

private:
    static constexpr size_t kBlobAlignment = CudaKernelLaunchCommand::fused_alignment;
    // Resource usages are staged here, so a launch allocates nothing for the
    // usage list. A typed kernel knows its exact resource count at compile time
    // (CudaKernelT::resource_arg_count()) and the untyped CudaKernel reserves
    // sizeof...(Args); 16 covers every observed launch shape (the radix one-
    // sweep kernel has 8). A hand-rolled launcher declaring more falls back to
    // `_heap_usages` - one extra allocation for that launch, nothing else
    // changes.
    static constexpr size_t kInlineUsages = 16u;

    // The fused staging block:
    //
    //   [ object slot | Argument records (_argument_capacity * 32B) |
    //     uniform payload (_uniform_bytes, room up to _payload_capacity_bytes) |
    //     slack | usage tail (_usage_tail_bytes) ]
    //
    // luisa::vector is deliberately not used: the command object must live in the
    // same allocation as the blob. The block comes from _alloc_block (see the
    // pairing comment above) and is freed either here (build() never called, via
    // the matching _free_block) or - once build() has placement-new'd the command
    // at its front - by the unique_ptr<Command> that owns the command, whose
    // deleter performs the paired deallocation. That is the same allocation/
    // deallocation pair today's luisa::make_unique<CudaKernelLaunch-Command>
    // relies on in each build (this class declares no class-specific operator
    // new/delete).
    std::byte *_block{nullptr};
    size_t _block_bytes{0u};
    size_t _argument_capacity{0u};    // Argument records that fit in the blob
    size_t _payload_capacity_bytes{0u};   // uniform payload room behind them (16-aligned)
    size_t _usage_tail_bytes{0u};     // tail capacity for the usage records (16-aligned)
    size_t _argument_count{0u};
    size_t _uniform_bytes{0u};        // payload bytes in use
    size_t _usage_count{0u};
    alignas(Usage) std::byte _inline_usages[kInlineUsages * sizeof(Usage)];
    luisa::vector<Usage> _heap_usages;    // overflow only; empty (no allocation) normally

    [[nodiscard]] static size_t _slot_bytes() noexcept { return CudaKernelLaunchCommand::object_slot_bytes(); }
    [[nodiscard]] std::byte *_blob() noexcept { return _block + _slot_bytes(); }
    [[nodiscard]] Argument *_argument_records() noexcept { return std::launder(reinterpret_cast<Argument *>(_blob())); }
    [[nodiscard]] size_t _header_bytes() const noexcept { return _argument_count * sizeof(Argument); }
    [[nodiscard]] size_t _capacity_bytes() const noexcept { return _argument_capacity * sizeof(Argument); }
    [[nodiscard]] static size_t _align_blob(size_t bytes) noexcept { return (bytes + kBlobAlignment - 1u) & ~(kBlobAlignment - 1u); }

    // The block's allocation must pair with the deallocation its OWNER performs.
    // The fused command is handed to a luisa::unique_ptr<...>, whose deleter is
    // build-configuration dependent:
    //   - EASTL builds (LUISA_USE_SYSTEM_STL undefined): eastl's default_delete
    //     runs ~T() and eastl::GetDefaultAllocator()->deallocate(p, size) (size
    //     is an ignored hint), so the block must come from the same allocator --
    //     luisa::detail::allocator_allocate, which is exactly that call.
    //   - System-STL builds: std::default_delete runs the delete EXPRESSION,
    //     i.e. the global operator delete paired with ::operator new (16-byte
    //     aligned on MSVC/x64, GCC and Clang; the block must stay blob-aligned).
    // Crossing the two would free a pointer through an allocator that never saw
    // it (undefined behavior, observed as a crash in the first command
    // destruction).
    [[nodiscard]] static void *_alloc_block(size_t bytes) noexcept {
#if defined(LUISA_USE_SYSTEM_STL)
        return ::operator new(bytes);
#else
        return luisa::detail::allocator_allocate(bytes, kBlobAlignment);
#endif
    }
    static void _free_block(void *p) noexcept {
#if defined(LUISA_USE_SYSTEM_STL)
        ::operator delete(p);
#else
        luisa::detail::allocator_deallocate(p, kBlobAlignment);
#endif
    }
    // Byte offset of the usage tail inside the block; 16-aligned because the
    // record region is a multiple of sizeof(Argument) (32) and the payload room
    // is kept 16-aligned.
    [[nodiscard]] size_t _tail_offset() const noexcept { return _slot_bytes() + _capacity_bytes() + _payload_capacity_bytes; }

    // Grow the staging block so it holds the current capacities
    // (_argument_capacity records, _payload_capacity_bytes of payload, the usage
    // tail), preserving `staged_bytes` of already-staged content at the same
    // offsets. Callers update the capacity members first and pass the length of
    // the content staged SO FAR (slot + old records + used payload), so a header
    // growth lands the old payload where the payload shift expects it.
    void _sync_block(size_t staged_bytes) noexcept {
        auto needed = _slot_bytes() + _capacity_bytes() + _payload_capacity_bytes + _usage_tail_bytes;
        if (needed <= _block_bytes) { return; }
        auto grown = _block_bytes > 0u && needed < 2u * _block_bytes ? 2u * _block_bytes : needed;
        auto capacity = _align_blob(grown);
        auto *block = static_cast<std::byte *>(_alloc_block(capacity));
        if (_block != nullptr) {
            std::memcpy(block, _block, staged_bytes < _block_bytes ? staged_bytes : _block_bytes);
            _free_block(_block);
        }
        _block = block;
        _block_bytes = capacity;
    }

    // Make room for `slots` argument records. Growing the header moves the
    // uniform payload up (every reference is an offset, so a block growth is
    // harmless) and patches the recorded uniform offsets. `exact` sizes the
    // header to `slots` so that a known launch shape needs no fix-up in build().
    void _reserve_argument_slots(size_t slots, bool exact) noexcept {
        if (slots <= _argument_capacity) { return; }
        auto new_capacity = exact ? slots : (slots > 2u * _argument_capacity ? slots : 2u * _argument_capacity);
        auto old_bytes = _capacity_bytes();
        auto new_bytes = new_capacity * sizeof(Argument);
        auto staged = _slot_bytes() + old_bytes + _uniform_bytes;
        _argument_capacity = new_capacity;
        _sync_block(staged);
        auto *base = _blob();
        std::memmove(base + new_bytes, base + old_bytes, _uniform_bytes);
        auto *args = _argument_records();
        for (auto i = 0u; i < _argument_count; ++i) {
            if (args[i].tag == Argument::Tag::UNIFORM) {
                args[i].uniform.offset += new_bytes - old_bytes;
            }
        }
    }

    [[nodiscard]] const Usage *_usages() const noexcept {
        return _heap_usages.empty() ? reinterpret_cast<const Usage *>(_inline_usages) : _heap_usages.data();
    }
    void _add_usage(Usage usage) noexcept {
        if (_usage_count < kInlineUsages && _heap_usages.empty()) {
            std::memcpy(_inline_usages + _usage_count * sizeof(Usage), &usage, sizeof(Usage));
            ++_usage_count;
            return;
        }
        if (_heap_usages.empty()) {
            // Overflow: migrate the inline staging, then keep growing on the heap.
            _heap_usages.reserve(_usage_count + 8u);
            for (auto i = 0u; i < _usage_count; ++i) {
                _heap_usages.emplace_back(_usages()[i]);
            }
        }
        _heap_usages.emplace_back(usage);
        ++_usage_count;
    }

    [[nodiscard]] Argument &_create_argument() noexcept {
        _reserve_argument_slots(_argument_count + 1u, false);
        auto &arg = _argument_records()[_argument_count++];
        // Byte-identity with the previous vector staging: luisa::vector::resize
        // zero-filled every grown byte, so an Argument's padding and the union
        // bytes beyond the assigned member were always 0 on the wire. `arg =
        // Argument{}` alone does NOT guarantee that: the implicit copy-assignment
        // copies the subobjects only, leaving padding/union tail indeterminate
        // (observed as stale payload bytes in the 12..28 range of a texture
        // record). Zero the whole 32-byte record explicitly, then callers assign
        // the tag and the union member they use.
        std::memset(&arg, 0, sizeof(Argument));
        return arg;
    }

    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand> _build(
        uint64_t cuda_function_handle,
        uint3 grid_dim, uint3 block_dim,
        uint32_t shared_mem_bytes, uint3 dispatch_size) && noexcept {
        // The tail must hold every usage actually collected, even if the
        // launcher over-grew past reserve().
        auto tail_bytes = _align_blob(_usage_count * sizeof(Usage));
        if (tail_bytes > _usage_tail_bytes) { _usage_tail_bytes = tail_bytes; }
        _sync_block(_slot_bytes() + _capacity_bytes() + _uniform_bytes);

        auto header_bytes = _header_bytes();
        auto *base = _blob();
        if (_argument_capacity != _argument_count) {
            // Unreserved launcher with growth slack: rotate the uniform payload
            // down behind the exact argument header and patch the recorded
            // offsets (the same fix-up as before, unchanged).
            auto capacity_bytes = _capacity_bytes();
            std::memmove(base + header_bytes, base + capacity_bytes, _uniform_bytes);
            auto *args = _argument_records();
            for (auto i = 0u; i < _argument_count; ++i) {
                if (args[i].tag == Argument::Tag::UNIFORM) {
                    args[i].uniform.offset -= capacity_bytes - header_bytes;
                }
            }
        }
        auto blob_bytes = header_bytes + _uniform_bytes;
        auto *usages = _usage_count > 0u
                           ? reinterpret_cast<Usage *>(_block + _tail_offset())
                           : nullptr;
        if (usages != nullptr) {
            std::memcpy(usages, _usages(), _usage_count * sizeof(Usage));
        }
        // Construct the command at the front of the block it was staged in: the
        // blob is already final, so nothing is copied here.
        CudaKernelLaunchCommand::FusedBlob blob{
            base, blob_bytes, usages, _usage_count, _argument_count,
            _block_bytes - _slot_bytes()};
        auto *cmd = new (static_cast<void *>(_block))
            CudaKernelLaunchCommand(cuda_function_handle, grid_dim, block_dim,
                                    shared_mem_bytes, dispatch_size, blob);
        // Ownership of the block has moved to the command (which is freed by
        // destruction through its unique_ptr<Command>, i.e. the deleter paired
        // with _alloc_block above).
        _block = nullptr;
        _block_bytes = 0u;
        _argument_capacity = 0u;
        _payload_capacity_bytes = 0u;
        _usage_tail_bytes = 0u;
        _argument_count = 0u;
        _uniform_bytes = 0u;
        _usage_count = 0u;
        return luisa::unique_ptr<CudaKernelLaunchCommand>{cmd};
    }

public:
    KernelLauncher() noexcept = default;
    KernelLauncher(KernelLauncher const &) = delete;
    KernelLauncher &operator=(KernelLauncher const &) = delete;
    // The launcher owns its staging block until build() turns it into the
    // command; a launcher that is destroyed without building frees it.
    ~KernelLauncher() noexcept {
        if (_block != nullptr) { _free_block(_block); }
    }
    KernelLauncher(KernelLauncher &&rhs) noexcept
        : _block{rhs._block}, _block_bytes{rhs._block_bytes},
          _argument_capacity{rhs._argument_capacity},
          _payload_capacity_bytes{rhs._payload_capacity_bytes},
          _usage_tail_bytes{rhs._usage_tail_bytes},
          _argument_count{rhs._argument_count},
          _uniform_bytes{rhs._uniform_bytes},
          _usage_count{rhs._usage_count},
          _heap_usages{std::move(rhs._heap_usages)} {
        std::memcpy(_inline_usages, rhs._inline_usages, sizeof(_inline_usages));
        rhs._block = nullptr;
        rhs._block_bytes = 0u;
    }
    KernelLauncher &operator=(KernelLauncher &&rhs) noexcept {
        if (this != &rhs) {
            if (_block != nullptr) { _free_block(_block); }
            _block = rhs._block;
            _block_bytes = rhs._block_bytes;
            _argument_capacity = rhs._argument_capacity;
            _payload_capacity_bytes = rhs._payload_capacity_bytes;
            _usage_tail_bytes = rhs._usage_tail_bytes;
            _argument_count = rhs._argument_count;
            _uniform_bytes = rhs._uniform_bytes;
            _usage_count = rhs._usage_count;
            _heap_usages = std::move(rhs._heap_usages);
            std::memcpy(_inline_usages, rhs._inline_usages, sizeof(_inline_usages));
            rhs._block = nullptr;
            rhs._block_bytes = 0u;
        }
        return *this;
    }

    // Pre-size the staging block for a known launch shape: exactly
    // `argument_count` argument records, `resource_count` resource usages and
    // at least `uniform_bytes` of by-value payload. Called with compile-time
    // counts by CudaKernelT / CudaKernel, so the launch performs a single
    // allocation (command + argument blob + usage tail together), with no
    // growth, no fix-up and no copy. A hand-rolled launcher that never calls
    // reserve() keeps working (its block grows on demand; build() then trims
    // the header and patches the offsets).
    void reserve(size_t argument_count, size_t resource_count, size_t uniform_bytes) noexcept {
        auto payload = _align_blob(uniform_bytes);
        if (payload > _payload_capacity_bytes) { _payload_capacity_bytes = payload; }
        auto tail = _align_blob(resource_count * sizeof(Usage));
        if (tail > _usage_tail_bytes) { _usage_tail_bytes = tail; }
        _reserve_argument_slots(argument_count, true);
        _sync_block(_slot_bytes() + _capacity_bytes() + _uniform_bytes);
        if (resource_count > kInlineUsages && _heap_usages.capacity() < resource_count) {
            _heap_usages.reserve(resource_count);
        }
    }

    // Buffer arguments arrive in the imported DSL kernel as LCBuffer bindings
    // (base device address + offset, size in bytes).
    KernelLauncher &add_buffer(uint64_t handle, size_t offset, size_t size, Usage usage) noexcept {
        auto &&arg = _create_argument();
        arg.tag = Argument::Tag::BUFFER;
        arg.buffer = Argument::Buffer{handle, offset, size};
        _add_usage(usage);
        return *this;
    }
    template<typename T>
    KernelLauncher &add_buffer(const BufferView<T> &view, Usage usage) noexcept {
        return add_buffer(view.handle(), view.offset_bytes(), view.size_bytes(), usage);
    }
    KernelLauncher &add_texture(uint64_t handle, uint32_t level, Usage usage) noexcept {
        auto &&arg = _create_argument();
        arg.tag = Argument::Tag::TEXTURE;
        // Field-wise on purpose: sizeof(Argument::Texture) is 16 (4 bytes of
        // trailing padding after `level`), and a whole-struct assignment may
        // copy the padding bytes of the source temporary, which are
        // indeterminate. _create_argument() zeroed the record, so writing only
        // the two members keeps the on-wire padding deterministically 0 on every
        // compiler.
        arg.texture.handle = handle;
        arg.texture.level = level;
        _add_usage(usage);
        return *this;
    }
    // Scalar/aggregate kernel parameters passed by value.
    KernelLauncher &add_uniform(const void *data, size_t size, size_t alignment) noexcept {
        LUISA_DEBUG_ASSERT(alignment > 0u && alignment <= 16u,
                           "Invalid uniform alignment {}.", alignment);
        // Reserve the argument record first: growing the header moves the
        // uniform payload, so the data offset is computed afterwards.
        auto index = _argument_count;
        static_cast<void>(_create_argument());
        auto local = luisa::align(_uniform_bytes, alignment);
        if (local + size > _payload_capacity_bytes) {
            auto grown = local + size;
            if (_payload_capacity_bytes > 0u && grown < 2u * _payload_capacity_bytes) {
                grown = 2u * _payload_capacity_bytes;
            }
            _payload_capacity_bytes = _align_blob(grown);
            _sync_block(_slot_bytes() + _capacity_bytes() + _uniform_bytes);
        }
        auto offset = _capacity_bytes() + local;
        if (local > _uniform_bytes) {
            // Keep the on-wire region fully initialized: luisa::vector's resize
            // used to zero-fill the alignment padding between the previous
            // payload end and this slot, so the fused block must do it
            // explicitly for the produced [records][payload] bytes to match the
            // previous layout byte-for-byte (not merely semantically).
            std::memset(_blob() + _capacity_bytes() + _uniform_bytes, 0, local - _uniform_bytes);
        }
        std::memcpy(_blob() + offset, data, size);
        auto *args = _argument_records();// re-read: the growth above may move it
        args[index].tag = Argument::Tag::UNIFORM;
        args[index].uniform = Argument::Uniform{offset, size, alignment};
        _uniform_bytes = local + size;
        return *this;
    }
    template<typename T>
    KernelLauncher &add_uniform(const T &value) noexcept {
        return add_uniform(&value, sizeof(T), alignof(T));
    }
    // Exact-thread-count launch: grid = ceil(thread_count / block_dim) and
    // the command records `thread_count` as the dispatch size (the `ls_kid`
    // trailer payload). This matches the cuda backend's launch semantics and
    // is the launch form for imported DSL kernels.
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand> build(
        uint64_t cuda_function_handle,
        uint3 thread_count, uint3 block_dim,
        uint32_t shared_mem_bytes = 0u) && noexcept {
        LUISA_ASSERT(block_dim.x > 0u && block_dim.y > 0u && block_dim.z > 0u,
                     "CUDA kernel launch block dimension must be nonzero.");
        auto ceil_div = [](uint32_t t, uint32_t b) noexcept {
            return (t + b - 1u) / b;
        };
        auto grid_dim = uint3{ceil_div(thread_count.x, block_dim.x),
                              ceil_div(thread_count.y, block_dim.y),
                              ceil_div(thread_count.z, block_dim.z)};
        return std::move(*this)._build(cuda_function_handle, grid_dim, block_dim,
                                       shared_mem_bytes, thread_count);
    }
};


// Attaches an explicit Usage to a resource argument. Although imported
// shaders are DSL-compiled, the cuda backend does not retain the argument
// position mapping needed to default usages at the vk call site; use the
// read()/write()/read_write() helpers, or pass a bare view (which defaults
// to READ_WRITE).
template<typename T>
struct UsageArg {
    T resource;
    Usage usage;
};

// Per-argument adapter carrying a view and its usage through the
// CudaKernelInvoke ratchet. Typed kernels (CudaKernelT) construct these
// internally from the usage baked into their signature; the untyped path
// accepts bare views (READ_WRITE) or usage-tagged views (read()/write()/
// read_write()).
template<typename View>
struct ResourceArg {
    View view;
    Usage usage;
    ResourceArg(View v) noexcept
        : view{v}, usage{Usage::READ_WRITE} {}
    ResourceArg(UsageArg<View> a) noexcept
        : view{a.resource}, usage{a.usage} {}
};

namespace detail {

// Resource types accepted by the typed CUDA kernel launch API. Everything
// else that is trivially copyable is packed as a by-value uniform.
template<typename T>
struct is_cuda_launch_resource : std::false_type {};
template<typename T>
struct is_cuda_launch_resource<BufferView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Buffer<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<ImageView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Image<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<VolumeView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_resource<Volume<T>> : std::true_type {};
template<>
struct is_cuda_launch_resource<ByteBuffer> : std::true_type {};
template<>
struct is_cuda_launch_resource<ByteBufferView> : std::true_type {};

template<typename T>
inline constexpr auto is_cuda_launch_resource_v =
    is_cuda_launch_resource<std::remove_cvref_t<T>>::value;

// Maps owning resources to their view types; views map to themselves.
template<typename T>
struct cuda_resource_view {
    using type = T;
};
template<typename T>
struct cuda_resource_view<Buffer<T>> {
    using type = BufferView<T>;
};
template<>
struct cuda_resource_view<ByteBuffer> {
    using type = ByteBufferView;
};
template<typename T>
struct cuda_resource_view<Image<T>> {
    using type = ImageView<T>;
};
template<typename T>
struct cuda_resource_view<Volume<T>> {
    using type = VolumeView<T>;
};
template<typename T>
using cuda_resource_view_t = typename cuda_resource_view<std::remove_cvref_t<T>>::type;

template<typename T>
struct is_cuda_launch_texture : std::false_type {};
template<typename T>
struct is_cuda_launch_texture<ImageView<T>> : std::true_type {};
template<typename T>
struct is_cuda_launch_texture<VolumeView<T>> : std::true_type {};
template<typename T>
inline constexpr auto is_cuda_launch_texture_v =
    is_cuda_launch_texture<std::remove_cvref_t<T>>::value;

template<typename T>
struct is_usage_arg : std::false_type {};
template<typename T>
struct is_usage_arg<UsageArg<T>> : std::true_type {};
template<typename T>
struct is_resource_arg : std::false_type {};
template<typename T>
struct is_resource_arg<ResourceArg<T>> : std::true_type {};

// True when the signature parameter is a resource type (owning or view).
template<typename T>
inline constexpr auto is_cuda_arg_resource_v =
    is_cuda_launch_resource_v<T> ||
    is_cuda_launch_resource_v<cuda_resource_view_t<T>>;

}// namespace detail

///@name Usage tagging helpers for typed CUDA kernel launches
///@{
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> read(BufferView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> write(BufferView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<BufferView<T>> read_write(BufferView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> read(ImageView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> write(ImageView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<ImageView<T>> read_write(ImageView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> read(VolumeView<T> view) noexcept { return {view, Usage::READ}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> write(VolumeView<T> view) noexcept { return {view, Usage::WRITE}; }
template<typename T>
[[nodiscard]] UsageArg<VolumeView<T>> read_write(VolumeView<T> view) noexcept { return {view, Usage::READ_WRITE}; }
[[nodiscard]] inline UsageArg<ByteBufferView> read(ByteBufferView view) noexcept { return {view, Usage::READ}; }
[[nodiscard]] inline UsageArg<ByteBufferView> write(ByteBufferView view) noexcept { return {view, Usage::WRITE}; }
[[nodiscard]] inline UsageArg<ByteBufferView> read_write(ByteBufferView view) noexcept { return {view, Usage::READ_WRITE}; }
///@}

// DSL-style ratchet over KernelLauncher: arguments are fold-encoded through
// operator<< (mirroring detail::ShaderInvokeBase), and the rvalue-qualified
// dispatch(...) builds the CudaKernelLaunchCommand. Carries the imported
// shader's compiled block size so the thread-count dispatch overloads can
// default the block dimension.
class CudaKernelInvoke {

private:
    KernelLauncher _launcher;
    uint64_t _function;
    uint3 _block_size;

    template<typename R>
    static void _encode_resource(KernelLauncher &launcher, R &&resource, Usage usage) noexcept {
        using Res = std::remove_cvref_t<R>;
        using View = detail::cuda_resource_view_t<Res>;
        static_assert(detail::is_cuda_launch_resource_v<Res>,
                      "CUDA kernel launch arguments must be buffers, images, "
                      "volumes, or trivially-copyable uniforms.");
        // Owning resources are converted to views; views pass through.
        auto view = [&]() noexcept -> View {
            if constexpr (std::is_same_v<Res, View>) {
                return resource;
            } else {
                return resource.view();
            }
        }();
        if constexpr (detail::is_cuda_launch_texture_v<View>) {
            launcher.add_texture(view.handle(), view.level(), usage);
        } else {
            launcher.add_buffer(view.handle(), view.offset_bytes(), view.size_bytes(), usage);
        }
    }

public:
    explicit CudaKernelInvoke(uint64_t cuda_function_handle,
                              uint3 block_size = uint3{0u, 0u, 0u}) noexcept
        : _function{cuda_function_handle}, _block_size{block_size} {}
    CudaKernelInvoke(CudaKernelInvoke const &) = delete;
    CudaKernelInvoke(CudaKernelInvoke &&) noexcept = default;

    // Pre-size the launch argument buffer from the (compile-time) launch shape;
    // see KernelLauncher::reserve.  Purely an optimization: the argument
    // encoding is identical without it.
    void reserve(size_t argument_count, size_t resource_count, size_t uniform_bytes) noexcept {
        _launcher.reserve(argument_count, resource_count, uniform_bytes);
    }

    // Usage-tagged resource arguments.
    template<typename R>
    CudaKernelInvoke &operator<<(UsageArg<R> arg) noexcept {
        _encode_resource(_launcher, arg.resource, arg.usage);
        return *this;
    }
    // Typed-kernel resource arguments (bare or usage-tagged).
    template<typename R>
    CudaKernelInvoke &operator<<(ResourceArg<R> arg) noexcept {
        _encode_resource(_launcher, arg.view, arg.usage);
        return *this;
    }
    // Bare resource arguments default to READ_WRITE (barrier-safe).
    template<typename R>
        requires detail::is_cuda_launch_resource_v<R>
    CudaKernelInvoke &operator<<(R &&resource) noexcept {
        _encode_resource(_launcher, std::forward<R>(resource), Usage::READ_WRITE);
        return *this;
    }
    // Trivially-copyable by-value uniforms.
    template<typename T>
        requires(!detail::is_cuda_launch_resource_v<T> &&
                 !detail::is_usage_arg<std::remove_cvref_t<T>>::value &&
                 !detail::is_resource_arg<std::remove_cvref_t<T>>::value &&
                 std::is_trivially_copyable_v<std::remove_cvref_t<T>>)
    CudaKernelInvoke &operator<<(T &&value) noexcept {
        _launcher.add_uniform(std::forward<T>(value));
        return *this;
    }

    // Explicit-geometry dispatch: grid and block dimensions are given
    // directly; the recorded dispatch size is grid * block (saturating,
    // padded), so imported DSL kernels guarding on dispatch_size() observe
    // the padded count. Prefer the thread-count overloads for imported DSL
    // kernels. Returns the same command type as KernelLauncher::build, so
    // `stream << ...` works unchanged.
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint3 grid_dim, uint3 block_dim, uint32_t shared_mem_bytes = 0u) && noexcept {
        auto saturate_mul = [](uint32_t a, uint32_t b) noexcept {
            auto v = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
            constexpr auto m = std::numeric_limits<uint32_t>::max();
            return static_cast<uint32_t>(v > m ? m : v);
        };
        auto thread_count = uint3{saturate_mul(grid_dim.x, block_dim.x),
                                  saturate_mul(grid_dim.y, block_dim.y),
                                  saturate_mul(grid_dim.z, block_dim.z)};
        return std::move(_launcher).build(_function, thread_count, block_dim, shared_mem_bytes);
    }
    // 1D exact-thread-count overload with explicit block dimension:
    // grid = ceil(thread_count_x / block_dim.x) and the command records the
    // exact thread count for the `ls_kid` trailer.
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint32_t thread_count_x, uint3 block_dim, uint32_t shared_mem_bytes = 0u) && noexcept {
        LUISA_ASSERT(block_dim.x > 0u, "CUDA kernel launch block dimension must be nonzero.");
        return std::move(_launcher).build(
            _function, uint3{thread_count_x, 1u, 1u}, block_dim, shared_mem_bytes);
    }
    // Exact-thread-count dispatch using the imported shader's compiled block
    // size (see VkCudaInterop::create_cuda_kernel).
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint3 thread_count) && noexcept {
        LUISA_ASSERT(_block_size.x > 0u && _block_size.y > 0u && _block_size.z > 0u,
                     "This CUDA kernel shader carries no compiled block size "
                     "(imported from a raw handle?); dispatch with an "
                     "explicit block dimension instead.");
        return std::move(_launcher).build(_function, thread_count, _block_size, 0u);
    }
    // Ergonomic 1D overload of the above.
    [[nodiscard]] luisa::unique_ptr<CudaKernelLaunchCommand>
    dispatch(uint32_t thread_count_x) && noexcept {
        return std::move(*this).dispatch(uint3{thread_count_x, 1u, 1u});
    }
};

// Untyped DSL-style CUDA kernel: `cuda_kernel(args...).dispatch(...)`.
// Usage must be supplied via read()/write()/read_write() wrappers, or bare
// views default to READ_WRITE.
class CudaKernel {

protected:
    uint64_t _handle;
    uint3 _block_size;

public:
    explicit CudaKernel(uint64_t cuda_function_handle,
                        uint3 block_size = uint3{0u, 0u, 0u}) noexcept
        : _handle{cuda_function_handle}, _block_size{block_size} {}

    template<typename... Args>
    [[nodiscard]] CudaKernelInvoke operator()(Args &&...args) const noexcept {
        CudaKernelInvoke invoke{_handle, _block_size};
        // One Argument record per call-site argument; `slots * 16` is a
        // conservative uniform hint (add_uniform grows if it is too small).
        invoke.reserve(sizeof...(Args), sizeof...(Args), 16u * sizeof...(Args));
        static_cast<void>((invoke << ... << std::forward<Args>(args)));
        return invoke;
    }
};

// Usage-carrying signature parameter for typed CUDA kernels. A kernel's
// per-argument read/write intent is a property of the kernel signature, not
// of an individual launch, so it is declared once when creating the typed
// kernel (create_cuda_kernel(...).kernel<CudaArg<Buffer<float>, Usage::READ>,
// ...>()) and baked into the CudaKernelT instance; dispatch sites then pass
// bare views only, mirroring the DSL where usage is inferred from the
// compiled kernel and never appears at the call site.
template<typename T, Usage U = Usage::READ_WRITE>
struct CudaArg {
    using resource_type = std::remove_cvref_t<T>;
    static constexpr auto usage = U;
};

// Traits mapping a declared typed-kernel signature parameter to its call-site
// argument type and baked usage:
// - CudaArg<T, U> (T an owning resource or view): call site passes the bare
//   view; usage is U.
// - bare resource type (Buffer<float>, ByteBuffer, Image<float>,
//   Volume<float>, BufferView<float>, ...): same, defaulting to READ_WRITE
//   (barrier-safe, matching the untyped path's bare-view behavior).
// - anything else (trivially-copyable scalars/aggregates): passed by const
//   reference as a by-value uniform; carries no usage.
template<typename T>
struct cuda_arg_traits {
    using type = std::remove_cvref_t<T>;
    static constexpr bool is_resource = detail::is_cuda_arg_resource_v<type>;
    static constexpr Usage usage = Usage::READ_WRITE;
    using view_type = std::conditional_t<is_resource,
                                         detail::cuda_resource_view_t<type>,
                                         void>;
    using arg_type = std::conditional_t<is_resource, view_type, const type &>;
};
template<typename T, Usage U>
struct cuda_arg_traits<CudaArg<T, U>> {
    using type = std::remove_cvref_t<T>;
    static_assert(detail::is_cuda_arg_resource_v<type>,
                  "CudaArg<T, U> requires a CUDA launch resource type "
                  "(buffer/image/volume/byte-buffer, owning or view).");
    static constexpr bool is_resource = true;
    static constexpr Usage usage = U;
    using view_type = detail::cuda_resource_view_t<type>;
    using arg_type = view_type;
};
template<typename T>
using cuda_arg_traits_t = cuda_arg_traits<std::remove_cvref_t<T>>;

// Typed DSL-style CUDA kernel mirroring Shader<dim, Args...>: the host-side
// argument list is checked against the declared signature, and each resource
// parameter's Usage is declared in the signature (via CudaArg<T, U>, or
// READ_WRITE for a bare resource type) and baked into the instance — dispatch
// sites pass bare views only. Declare by-value parameters as their scalar
// types. The imported module image is opaque, so this checks shape, not the
// device signature (same trust level as AOT Shader<dim, Args...> loaded from
// file); the declared usages are likewise trusted, not verified against the
// compiled kernel. Per-launch usage overrides are only available through the
// untyped CudaKernel / KernelLauncher path.
template<concepts::non_cvref... Args>
class CudaKernelT final : public CudaKernel {

private:
    template<typename Arg, typename A>
    static void _encode_arg(CudaKernelInvoke &invoke, A &&arg) noexcept {
        using traits = cuda_arg_traits_t<Arg>;
        if constexpr (traits::is_resource) {
            // Construct ResourceArg explicitly so overload resolution against
            // CudaKernelInvoke::operator<< is deterministic.
            invoke << ResourceArg<typename traits::view_type>{
                UsageArg<typename traits::view_type>{arg, traits::usage}};
        } else {
            invoke << arg;
        }
    }

    // Positional pairing of Args... with the call-site arguments is guaranteed
    // by parallel pack expansion; a plain comma fold would lose the per-Arg
    // association only if Args... and args... were not both expanded together.
    [[nodiscard]] CudaKernelInvoke _invoke(
        typename cuda_arg_traits_t<Args>::arg_type... args) const noexcept {
        CudaKernelInvoke invoke{this->_handle, this->_block_size};
        invoke.reserve(sizeof...(Args), resource_arg_count(), uniform_bytes_hint());
        (static_cast<void>(_encode_arg<Args>(invoke, args)), ...);
        return invoke;
    }

    // Compile-time launch shape used to pre-size the argument buffer: exactly
    // the number of resource (non-uniform) arguments, and an upper bound of the
    // by-value uniform payload (16-byte rounding per uniform plus its slot).
    [[nodiscard]] static constexpr size_t resource_arg_count() noexcept {
        return (size_t{0u} + ... +
                (cuda_arg_traits_t<Args>::is_resource ? size_t{1u} : size_t{0u}));
    }
    [[nodiscard]] static constexpr size_t uniform_bytes_hint() noexcept {
        auto bytes = size_t{0u};
        auto add = [&bytes]<typename Arg>() noexcept {
            if constexpr (!cuda_arg_traits_t<Arg>::is_resource) {
                bytes += ((sizeof(typename cuda_arg_traits_t<Arg>::type) + 15u) & ~size_t{15u}) + 16u;
            }
        };
        (add.template operator()<Args>(), ...);
        return bytes;
    }

public:
    using CudaKernel::CudaKernel;

    [[nodiscard]] static constexpr size_t arg_count() noexcept {
        return sizeof...(Args);
    }

    // The per-resource-argument usages baked into this instance, in argument
    // order (uniform parameters are skipped), matching the layout of
    // CudaKernelLaunchCommand::argument_usages().
    [[nodiscard]] static constexpr auto argument_usages() noexcept {
        std::array<Usage, (0u + ... + (cuda_arg_traits_t<Args>::is_resource ? 1u : 0u))> usages{};
        auto index = 0u;
        auto append = [&index, &usages]<typename Arg>() noexcept {
            if constexpr (cuda_arg_traits_t<Arg>::is_resource) {
                usages[index++] = cuda_arg_traits_t<Arg>::usage;
            }
        };
        (append.template operator()<Args>(), ...);
        return usages;
    }

    [[nodiscard]] CudaKernelInvoke operator()(typename cuda_arg_traits_t<Args>::arg_type... args) const noexcept {
        return _invoke(args...);
    }
};

// RAII owner of a CUDA kernel shader handle created through
// VkCudaInterop::create_cuda_kernel_shader; destroys it on destruction.
// Defined at the end of this header (needs the complete VkCudaInterop type).
class CudaShader;

}// namespace vk_cuda_interop

class VkCudaInterop : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "VkCudaInterop";

public:
    [[nodiscard]] virtual BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept = 0;
    // Reverse direction of create_interop_buffer: the allocation is owned by
    // CUDA instead of Vulkan. The implementation allocates device memory with
    // the CUDA driver (cuMemCreate/cuMemMap on the CUDA device paired with
    // this Vulkan device), exports it as an OS shareable handle, and imports
    // it into Vulkan (VkExternalMemoryBufferCreateInfo + vkAllocateMemory
    // import), so that the resulting buffer can be used by Vulkan commands -
    // including VK_NV_cuda_kernel_launch dispatches - while CUDA sees it as a
    // plain device allocation at *cuda_device_ptr (when cuda_device_ptr is not
    // null; the pointer can be wrapped on a CUDA backend device with
    // Device::import_external_buffer).
    // The allocation is rounded up to the CUDA allocation granularity (2 MiB on
    // current NVIDIA desktop parts), so a small buffer still costs a granule.
    // Fails closed: returns an invalid BufferCreationInfo (handle ==
    // invalid_resource_handle, so the Buffer built from it evaluates to false)
    // when the
    // platform or the driver cannot export/import CUDA-owned memory; the
    // caller must check the result. The returned buffer releases both the
    // Vulkan and the CUDA side when destroyed.
    [[nodiscard]] virtual BufferCreationInfo create_interop_buffer_from_cuda(
        const Type *element, size_t elem_count, uint64_t *cuda_device_ptr) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept = 0;
    virtual void vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;
    virtual void vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept = 0;

public:
    [[nodiscard]] virtual CUDADeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept = 0;
    virtual void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept = 0;
    [[nodiscard]] virtual /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept = 0;
    virtual void unmap(void *cuda_ptr, void *cuda_handle) noexcept = 0;
    [[nodiscard]] virtual int cuda_device_index() const noexcept = 0;
    [[nodiscard]] virtual DeviceInterface *device() noexcept = 0;

public:
    // VK_NV_cuda_kernel_launch support: true only when the backend was built
    // with CUDA interop and the device extension/feature could be enabled
    // (owned logical devices only).
    [[nodiscard]] virtual bool cuda_kernel_launch_supported() const noexcept = 0;
    // Imports a compute shader compiled by the *cuda backend* from a DSL
    // kernel — cuda_shader_handle is Shader::handle() of a shader compiled
    // with cuda_device.compile(kernel) on the same physical GPU (pair the
    // devices via cuda_device_index()). The imported shader is usable with
    // vk_cuda_interop::KernelLauncher/CudaKernelLaunchCommand; its launch ABI
    // is the DSL kernel ABI (a single by-value `struct Params` blob with
    // 16-byte-aligned argument slots, buffers as LCBuffer { ptr, size_bytes }
    // bindings, and an `ls_kid` uint4 trailer carrying the exact dispatch
    // size).
    // Returns 0 when CUDA kernel launch is unsupported or the shader is not
    // importable (OptiX ray-tracing shaders, shaders with bound arguments, or
    // shaders using device printing). The vk shader is self-contained after
    // import; the cuda shader may be destroyed once this call returns.
    [[nodiscard]] virtual uint64_t create_cuda_kernel_shader(uint64_t cuda_shader_handle) noexcept = 0;
    virtual void destroy_cuda_kernel_shader(uint64_t handle) noexcept = 0;

    // RAII wrapper around create_cuda_kernel_shader; the returned object
    // destroys the shader handle on destruction. Evaluates to false when
    // CUDA kernel launch is unsupported or the shader is not importable.
    // This overload does not record a block size, so the thread-count
    // dispatch overloads are unavailable; use explicit geometry or the
    // Shader-taking overload below.
    [[nodiscard]] vk_cuda_interop::CudaShader create_cuda_kernel(uint64_t cuda_shader_handle) noexcept;
    // Imports a shader compiled on a cuda device (cuda_device.compile(kernel))
    // and records its compiled block size, enabling the thread-count dispatch
    // overloads. Passing a shader not compiled by the cuda backend is a
    // contract violation: null/invalid shaders return an empty CudaShader,
    // but a wild handle is still UB (same trust level as other handle APIs).
    template<size_t N, typename... Args>
    [[nodiscard]] vk_cuda_interop::CudaShader create_cuda_kernel(const Shader<N, Args...> &shader) noexcept;

    vk_cuda_interop::Signal vk_signal(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Signal{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Wait vk_wait(TimelineEvent const &cuda_event, uint64_t fence_index) noexcept {
        return vk_cuda_interop::Wait{
            this,
            cuda_event.handle(),
            fence_index};
    }
    vk_cuda_interop::Signal vk_signal(Event const &cuda_event) noexcept {
        auto signal = cuda_event.signal();
        return vk_cuda_interop::Signal{
            this,
            signal.handle,
            signal.fence};
    }
    vk_cuda_interop::Wait vk_wait(Event const &cuda_event, uint64_t fence = std::numeric_limits<uint64_t>::max()) noexcept {
        auto wait = cuda_event.wait(fence);
        return vk_cuda_interop::Wait{
            this,
            wait.handle,
            wait.fence};
    }
    template<typename T>
    Buffer<T> create_buffer(size_t elem_count) noexcept {
        return Buffer<T>{device(), create_interop_buffer(Type::of<T>(), elem_count)};
    }
    ByteBuffer create_byte_buffer(size_t size_bytes) noexcept {
        return ByteBuffer{device(), create_interop_buffer(Type::of<void>(), size_bytes)};
    }
    // CUDA-owned counterparts of create_buffer/create_byte_buffer (see
    // create_interop_buffer_from_cuda). Invalid (empty) buffers on failure.
    template<typename T>
    Buffer<T> create_buffer_from_cuda(size_t elem_count, uint64_t *cuda_device_ptr = nullptr) noexcept {
        return Buffer<T>{device(), create_interop_buffer_from_cuda(Type::of<T>(), elem_count, cuda_device_ptr)};
    }
    ByteBuffer create_byte_buffer_from_cuda(size_t size_bytes, uint64_t *cuda_device_ptr = nullptr) noexcept {
        return ByteBuffer{device(), create_interop_buffer_from_cuda(Type::of<void>(), size_bytes, cuda_device_ptr)};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, width, height, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint2(width, height),
            mip_levels};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, size.x, size.y, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint width, uint height, uint volume, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, width, height, volume, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint3(width, height, volume),
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, size.x, size.y, size.z, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }

    virtual ~VkCudaInterop() noexcept = default;
};
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(vk_cuda_interop::Wait)
namespace vk_cuda_interop {
inline void Signal::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->vk_signal(handle, stream_handle, fence);
}
inline void Wait::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->vk_wait(handle, stream_handle, fence);
}

// RAII owner of a CUDA kernel shader handle created through
// VkCudaInterop::create_cuda_kernel_shader; destroys it on destruction.
// Optionally carries the source DSL kernel's compiled block size, which the
// thread-count dispatch overloads use as the default block dimension.
class CudaShader {

private:
    VkCudaInterop *_ext{nullptr};
    uint64_t _handle{0u};
    uint3 _block_size{0u, 0u, 0u};

    void _destroy() noexcept {
        if (_handle != 0u) {
            _ext->destroy_cuda_kernel_shader(_handle);
            _handle = 0u;
        }
    }

public:
    CudaShader() noexcept = default;
    CudaShader(VkCudaInterop &ext, uint64_t handle,
               uint3 block_size = uint3{0u, 0u, 0u}) noexcept
        : _ext{&ext}, _handle{handle}, _block_size{block_size} {}
    ~CudaShader() noexcept { _destroy(); }
    CudaShader(CudaShader const &) = delete;
    CudaShader &operator=(CudaShader const &) = delete;
    CudaShader(CudaShader &&rhs) noexcept
        : _ext{rhs._ext}, _handle{rhs._handle}, _block_size{rhs._block_size} {
        rhs._handle = 0u;
    }
    CudaShader &operator=(CudaShader &&rhs) noexcept {
        if (this != &rhs) {
            _destroy();
            _ext = rhs._ext;
            _handle = rhs._handle;
            _block_size = rhs._block_size;
            rhs._handle = 0u;
        }
        return *this;
    }
    explicit operator bool() const noexcept { return _handle != 0u; }
    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    // The source DSL kernel's compiled block size, or (0, 0, 0) when unknown
    // (raw-handle imports).
    [[nodiscard]] uint3 block_size() const noexcept { return _block_size; }
    uint64_t release() noexcept {
        auto handle = _handle;
        _handle = 0u;
        return handle;
    }
    // Typed DSL-style kernel bound to this shader's handle.
    template<concepts::non_cvref... KArgs>
    [[nodiscard]] CudaKernelT<KArgs...> kernel() const noexcept {
        return CudaKernelT<KArgs...>{_handle, _block_size};
    }
    // Untyped DSL-style kernel bound to this shader's handle.
    [[nodiscard]] CudaKernel kernel() const noexcept {
        return CudaKernel{_handle, _block_size};
    }
};

}// namespace vk_cuda_interop

inline vk_cuda_interop::CudaShader VkCudaInterop::create_cuda_kernel(uint64_t cuda_shader_handle) noexcept {
    return vk_cuda_interop::CudaShader{*this, create_cuda_kernel_shader(cuda_shader_handle)};
}
template<size_t N, typename... Args>
inline vk_cuda_interop::CudaShader VkCudaInterop::create_cuda_kernel(const Shader<N, Args...> &shader) noexcept {
    if (!shader) { return {}; }
    return vk_cuda_interop::CudaShader{*this, create_cuda_kernel_shader(shader.handle()), shader.block_size()};
}

}// namespace luisa::compute