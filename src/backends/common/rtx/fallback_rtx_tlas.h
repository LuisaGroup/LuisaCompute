// The top level of the fallback: an LBVH over instances, laid out as one
// *region* of the shared acceleration buffer, where every instance references a
// fallback BLAS (fallback_rtx_layout.h).
//
// This is the port of examples/compute/lbvh/tlas.{h,cpp}, with the two
// differences the ABI forces:
//
//   * the transform and the identity of an instance live in the *instance
//     record* (128 bytes, fallback_rtx_layout.h), which a shader may read and
//     write; the record is the authority, and the build only fills the slots the
//     caller modified;
//   * the tree cannot be built from the host: the world AABB of an instance is
//     the transformed root AABB of the BLAS region it references, which the host
//     does not know.  The build therefore refreshes the region's blas table from
//     the live instance records (a GPU kernel) and derives the instance AABBs
//     from that table, exactly like the hardware backends derive them from the
//     traversable handle of an instance.

#pragma once

#include "fallback_rtx.h"
#include "fallback_rtx_layout.h"
#include "fallback_rtx_storage.h"

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/bindless_array.h>

namespace lc::fallback_rtx {

// The host-side description of the BLAS one instance refers to.  The device
// resolves it, because it owns both handle spaces (the fallback BLAS handles and
// the shared storage); the builder only needs the directory entry for the blas
// table and the region *view* the TLAS' bindless heap registers.
struct FallbackTlasBlas {
    // The blas-directory entry of the BLAS (the builder's private lane).
    uint directory_entry{};
    // u4 offset of the BLAS region inside the shared acceleration buffer.
    uint region_base{};
    // Payload of that region, in u4: the length of the heap view.
    uint region_u4{};
};

// The host-side state of one fallback TLAS.
struct FallbackTlas {
    AccelOption option;
    // Where the tree lives; `region.base == 0u` until the first build.
    FallbackRtxRegion region;
    uint instance_count{};
    // Host copy of the instance table, eight uint4 per instance.  It is the
    // source of the modified slots (and of the first build's full upload), and
    // it is what the *identity* of an instance - the BLAS it refers to - is read
    // from: a shader may rewrite the transform and the property lanes, but never
    // the primitive an instance refers to.
    luisa::vector<uint4> records;
    // The blas directory *entry* every instance refers to; it is what the
    // record's reserved uint4 carries, and what the table refresh resolves
    // against the device directory.
    luisa::vector<uint32_t> directory_entry;
    // The bindless heap of this TLAS (fallback_rtx_layout.h): slot 0 is the null
    // slot, slot 1 the TLAS' own region and slot 2+i the BLAS region of instance
    // i.  It is a member, so it is created with the TLAS and released when the
    // TLAS is destroyed - the lifetime of the heap is the lifetime of the tree
    // that references it, and nothing outside has to remember to free it.
    BindlessArray accel_heap;
    // Heaps this TLAS outgrew.  A rebuild that needs more slots creates a new
    // `BindlessArray` rather than growing one (a bindless array has a fixed slot
    // count), and the old one is *retired*, not destroyed: a command that is
    // still in flight may read it, which is the same promise the storage's
    // growable buffers make.
    luisa::vector<BindlessArray> retired_heaps;
    bool built{false};
};

// Host staging of the upload commands a build records.  `BufferUploadCommand`
// keeps a raw host pointer and the command may still be in flight when the next
// build runs, so every build allocates its own blocks and never reuses or
// reallocates the storage of an earlier one.  It is one block per build, so the
// cost is a few bytes per instance per build and is bounded by the number of
// builds (documented in the same spirit as the retired device buffers).
template<typename T>
class FallbackRtxHostStage {

public:
    [[nodiscard]] luisa::span<T> allocate(size_t count) noexcept {
        auto block = luisa::make_unique<T[]>(count);
        auto *data = block.get();
        _blocks.emplace_back(std::move(block));
        return luisa::span<T>{data, count};
    }

private:
    luisa::vector<luisa::unique_ptr<T[]>> _blocks;
};

class FallbackTlasBuilder {

public:
    explicit FallbackTlasBuilder(FallbackRtxStorage &storage) noexcept;

    // Record the build of `tlas`.  `resolved[i]` describes the BLAS of
    // `modifications[i].primitive` (the caller resolves it, because it owns the
    // BLAS handles); it is only read for the modifications that carry
    // `Modification::flag_primitive`.  With `update_instance_buffer_only` the
    // instance records, the region's blas table and the bindless heap are
    // refreshed but the radix tree is left as it is.
    void build(CommandList &commands, FallbackTlas &tlas, uint instance_count,
               luisa::span<const AccelBuildCommand::Modification> modifications,
               luisa::span<const FallbackTlasBlas> resolved,
               bool update_instance_buffer_only) noexcept;

    // The instance buffer slot of one record, and its host copy.
    [[nodiscard]] static uint instance_u4_offset(const FallbackTlas &tlas,
                                                 uint index) noexcept {
        return tlas.region.instance_offset + index * instance_u4;
    }

private:
    // Make sure `tlas` owns a heap with room for `instance_count` instances (slot
    // 2 + i is the BLAS of instance i), creating one or retiring the heap it
    // outgrew.  Both the new heap and the retirement live in `tlas`, so the
    // lifetime of the heap is the lifetime of the tree (RAII).
    void ensure_heap(FallbackTlas &tlas, uint instance_count) noexcept;
    // Copy the modified records into the device instance buffer, at their own
    // byte offsets: only the modified slots are uploaded, and each one lands at
    // the byte offset its record already occupies.
    void upload_records(CommandList &commands, FallbackTlas &tlas,
                        luisa::span<const uint32_t> uploaded) noexcept;
    // The same for *every* record, which is what a TLAS that is built for the
    // first time (or that grew) needs: an instance the caller never mentioned
    // must be a defined zero record, and one upload of the whole slice is
    // cheaper than one command per instance.
    void upload_every_record(CommandList &commands, FallbackTlas &tlas) noexcept;

    FallbackRtxStorage *_storage{nullptr};
    // One block per build: the staged instance records.
    FallbackRtxHostStage<uint> _record_stage;
    // The whole table is cleared before it is filled, so an instance whose row
    // never gets written is a defined (null) record and not yesterday's tree.
    Shader1D<Buffer<uint4>, uint, uint> _table_clear_kernel;
    // table[row] = directory[entry], for every live instance (row = its
    // `blas_index`, entry = the BLAS it refers to): a refresh of the region's
    // blas table from the live instance records.  The record's private lane also
    // carries the *bindless slot* of the referenced region, which the refresh
    // copies into the record's metadata uint4 (fallback_rtx_layout.h).
    Shader1D<Buffer<uint4>, Buffer<uint4>, Buffer<uint4>, uint, uint, uint, uint>
        _table_refresh_kernel;
    // World-space AABB of every instance, from the root AABB of the BLAS its
    // table row points at.
    Shader1D<Buffer<uint4>, Buffer<uint4>, Buffer<FallbackRtxPrim>, Buffer<uint>,
             uint, uint, uint, uint, uint, uint>
        _prim_kernel;
};

}// namespace lc::fallback_rtx
