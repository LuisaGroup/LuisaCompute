// Top-level build of the fallback: the instance records, the region's blas
// table and the shared tree build.

#include "fallback_rtx_tlas.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>

#include <bit>

namespace lc::fallback_rtx {

namespace {

// `Modification::set_transform` stores the *rows* of the world matrix
// (affine[row * 4 + column] = m[column][row]), which is the row convention the
// layout header uses for the instance record and the examples' LBVH uses for its
// instances.  The fourth row is the affine one.
[[nodiscard]] float4x4 world_from_modification(const float affine[12]) noexcept {
    float4x4 m;
    for (auto row = 0u; row < 3u; row++) {
        for (auto column = 0u; column < 4u; column++) {
            m[column][row] = affine[row * 4u + column];
        }
    }
    m[0][3] = 0.0f;
    m[1][3] = 0.0f;
    m[2][3] = 0.0f;
    m[3][3] = 1.0f;
    return m;
}

[[nodiscard]] float4 matrix_row(const float4x4 &m, uint row) noexcept {
    return make_float4(m[0][row], m[1][row], m[2][row], m[3][row]);
}

[[nodiscard]] uint4 u4_from_float4(float4 v) noexcept {
    return make_uint4(luisa::bit_cast<uint>(v.x), luisa::bit_cast<uint>(v.y),
                      luisa::bit_cast<uint>(v.z), luisa::bit_cast<uint>(v.w));
}

// The instance buffer as raw words: an upload addresses a byte range, and the
// record of instance `index` is exactly the eight uint4 starting at its slot.
[[nodiscard]] BufferView<uint> instance_words(const Buffer<uint4> &instances,
                                              uint u4_offset, uint u4_count) noexcept {
    return instances.view(u4_offset, u4_count).as<uint>();
}

constexpr uint record_words = instance_u4 * 4u;

}// namespace

FallbackTlasBuilder::FallbackTlasBuilder(FallbackRtxStorage &storage) noexcept
    : _storage{&storage},

      // The table is cleared before it is filled: a row whose instance carries a
      // null (never written) record must be a defined zero, not yesterday's tree
      // that a traversal could still descend into.
      _table_clear_kernel{storage.device().compile(Kernel1D{
          [](BufferVar<uint4> accel, UInt table_base, UInt count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              $if (i < count) {
                  accel.write(table_base + i * blas_record_u4 + 0u,
                              make_uint4(0u, 0u, 0u, 0u));
                  accel.write(table_base + i * blas_record_u4 + 1u,
                              make_uint4(0u, 0u, 0u, 0u));
              };
          }})},

      // table[row] = directory[entry]: the row is read from the *live* instance
      // record (a shader may have rewritten it), and the entry plus the bindless
      // slot are the builder's private use of the record's reserved uint4 - the
      // BLAS an instance refers to and where its region sits in this TLAS' heap
      // are host bookkeeping, and no shader-side operation ever changes them.
      _table_refresh_kernel{storage.device().compile(Kernel1D{
          [](BufferVar<uint4> accel, BufferVar<uint4> instances, BufferVar<uint4> directory,
             UInt instance_base, UInt table_base, UInt count, UInt directory_count) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto first = instance_base + i * instance_u4;
                  auto row = instances.read(first + i_misc).x;
                  auto private_lanes = instances.read(first + i_misc + 1u);
                  auto entry = private_lanes.x * blas_record_u4;
                  // A record the caller never filled is a null record: its row
                  // stays clear, and the range checks keep a garbage entry from
                  // turning into an out-of-bounds device read.
                  $if (row < count) {
                      $if (entry + 1u < directory_count) {
                          accel.write(table_base + row * blas_record_u4 + 0u,
                                      directory.read(entry + 0u));
                          auto metadata = directory.read(entry + 1u);
                          // The traversal resolves the referenced BLAS through the
                          // TLAS' bindless heap, so the record has to carry the
                          // slot of that region (fallback_rtx_layout.h); the slot
                          // is host-known and travelled in the record's private
                          // lane.  A slot of 0 is the null slot, which a traverse
                          // skips exactly like a null `blas_base` used to be.
                          accel.write(table_base + row * blas_record_u4 + 1u,
                                      make_uint4(metadata.x, metadata.y,
                                                 private_lanes.y, metadata.w));
                      };
                  };
              };
          }})},

      // World-space AABB of every TLAS instance, from the root AABB of the BLAS
      // its table row points at (the example reads the BLAS root AABB directly,
      // because there both levels live in the same buffers and the host builds
      // the table).
      _prim_kernel{storage.device().compile(Kernel1D{
          [](BufferVar<uint4> accel, BufferVar<uint4> instances,
             BufferVar<FallbackRtxPrim> prims, BufferVar<uint> reduce, UInt instance_base,
             UInt prim_base, UInt table_base, UInt accel_count, UInt count,
             UInt reduce_offset) noexcept {
              set_block_size(sort_block_size);
              UInt i = dispatch_id().x;
              $if (i < count) {
                  auto first = instance_base + i * instance_u4;
                  auto row = instances.read(first + i_misc).x;
                  // The identity AABB of an instance whose row is not live: the
                  // structural validator reports it, and the tree stays well
                  // formed instead of reading a wild offset on the device.
                  auto lo = def(make_float3(1.0e30f));
                  auto hi = def(make_float3(-1.0e30f));
                  $if (row < count) {
                      auto node_base = accel.read(table_base + row * blas_record_u4).y;
                      $if (node_base + 1u < accel_count) {
                          auto root_lo = accel.read(node_base);
                          auto root_hi = accel.read(node_base + 1u);
                          auto blas_lo = make_float3(root_lo.x.bitcast<float>(),
                                                     root_lo.y.bitcast<float>(),
                                                     root_lo.z.bitcast<float>());
                          auto blas_hi = make_float3(root_hi.x.bitcast<float>(),
                                                     root_hi.y.bitcast<float>(),
                                                     root_hi.z.bitcast<float>());
                          auto row_0 = instances.read(first + i_to_world + 0u);
                          auto row_1 = instances.read(first + i_to_world + 1u);
                          auto row_2 = instances.read(first + i_to_world + 2u);
                          auto w0 = make_float4(row_0.x.bitcast<float>(), row_0.y.bitcast<float>(),
                                                row_0.z.bitcast<float>(), row_0.w.bitcast<float>());
                          auto w1 = make_float4(row_1.x.bitcast<float>(), row_1.y.bitcast<float>(),
                                                row_1.z.bitcast<float>(), row_1.w.bitcast<float>());
                          auto w2 = make_float4(row_2.x.bitcast<float>(), row_2.y.bitcast<float>(),
                                                row_2.z.bitcast<float>(), row_2.w.bitcast<float>());
                          lo = def(make_float3(1.0e30f));
                          hi = def(make_float3(-1.0e30f));
                          $for (c, 8u) {
                              auto corner = make_float3(select(blas_lo.x, blas_hi.x, (c & 1u) != 0u),
                                                        select(blas_lo.y, blas_hi.y, (c & 2u) != 0u),
                                                        select(blas_lo.z, blas_hi.z, (c & 4u) != 0u));
                              auto p = make_float4(corner, 1.0f);
                              auto world = make_float3(dot(p, w0), dot(p, w1), dot(p, w2));
                              lo = min(lo, world);
                              hi = max(hi, world);
                          };
                      };
                  };
                  Var<FallbackRtxPrim> prim;
                  // The instance index the leaf node carries travels in the spare
                  // fourth lane of the lower plane, bit-cast, which is what keeps
                  // the record at 32 bytes (`FallbackRtxPrim`); it is read back
                  // bit-exactly by `prim_id()`.
                  prim.lo = make_float4(lo, i.bitcast<float>());
                  prim.hi = make_float4(hi, 0.0f);
                  prims.write(prim_base + i, prim);
                  // the scene volume the Morton codes are normalized with is not
                  // known on the host either (see fallback_rtx_storage.h)
                  reduce.atomic(reduce_offset + reduce_min_key + 0u).fetch_min(orderable_key(lo.x));
                  reduce.atomic(reduce_offset + reduce_min_key + 1u).fetch_min(orderable_key(lo.y));
                  reduce.atomic(reduce_offset + reduce_min_key + 2u).fetch_min(orderable_key(lo.z));
                  reduce.atomic(reduce_offset + reduce_max_key + 0u).fetch_max(orderable_key(hi.x));
                  reduce.atomic(reduce_offset + reduce_max_key + 1u).fetch_max(orderable_key(hi.y));
                  reduce.atomic(reduce_offset + reduce_max_key + 2u).fetch_max(orderable_key(hi.z));
              };
          }})} {}

// ---------------------------------------------------------------------------
// Instance records
// ---------------------------------------------------------------------------

void FallbackTlasBuilder::upload_records(CommandList &commands, FallbackTlas &tlas,
                                         luisa::span<const uint32_t> uploaded) noexcept {
    if (uploaded.empty()) { return; }
    // One host block for the whole build, filled with the records in the order
    // they will be uploaded; the block is never reallocated (see the header).
    auto staged = _record_stage.allocate(uploaded.size() * record_words);
    auto *data = staged.data();
    for (auto k = 0u; k < uploaded.size(); k++) {
        auto record = &tlas.records[static_cast<size_t>(uploaded[k]) * instance_u4];
        for (auto w = 0u; w < record_words; w++) {
            data[k * record_words + w] = reinterpret_cast<const uint *>(record)[w];
        }
    }
    auto &instances = _storage->instances();
    // One copy per *run* of consecutive instance indices, not one copy per
    // instance: the records of a run are adjacent in `staged` (it is filled in
    // this same order) and their slots are adjacent in the instance buffer, so a
    // single copy covers the whole run.  A build that mentions every instance
    // used to record one 128-byte copy command per instance, and at ~1.7 us of
    // submission each that was 110 ms of a 65536-instance rebuild against the
    // 1.5 ms the tree itself costs.  `uploaded` is in the caller's order, which
    // is why this coalesces runs instead of sorting them: ascending indices - the
    // common full build - collapse to one copy, and an order that has no run at
    // all is never worse than the per-instance form.
    for (auto k = 0u; k < uploaded.size();) {
        auto run = 1u;
        while (k + run < uploaded.size() && uploaded[k + run] == uploaded[k] + run) { run++; }
        auto slot = tlas.region.instance_offset + uploaded[k] * instance_u4;
        commands << instance_words(instances, slot, run * instance_u4)
                        .copy_from(staged.subspan(k * record_words, run * record_words));
        k += run;
    }
}

void FallbackTlasBuilder::upload_every_record(CommandList &commands,
                                              FallbackTlas &tlas) noexcept {
    auto count = static_cast<size_t>(tlas.instance_count);
    if (count == 0u) { return; }
    auto staged = _record_stage.allocate(count * record_words);
    auto *data = staged.data();
    for (auto i = 0u; i < count; i++) {
        auto record = &tlas.records[i * instance_u4];
        for (auto w = 0u; w < record_words; w++) {
            data[i * record_words + w] = reinterpret_cast<const uint *>(record)[w];
        }
    }
    commands << instance_words(_storage->instances(), tlas.region.instance_offset,
                               tlas.instance_count * instance_u4)
                    .copy_from(staged);
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

void FallbackTlasBuilder::ensure_heap(FallbackTlas &tlas,
                                      uint instance_count) noexcept {
    // Slot 0 is the null slot, slot 1 the TLAS' own region and slot 2+i the
    // BLAS of instance i (fallback_rtx_layout.h).
    auto needed = static_cast<size_t>(instance_count) + heap_first_blas_slot;
    if (tlas.accel_heap && tlas.accel_heap.size() >= needed) { return; }
    if (tlas.accel_heap) {
        // A bindless array has a fixed slot count, so a TLAS that grew gets a new
        // one.  The heap it outgrew is *retired*, not destroyed: a command that
        // is still in flight may read it (the same promise `GrowableBuffer`
        // makes for the storage).
        tlas.retired_heaps.emplace_back(std::move(tlas.accel_heap));
    }
    tlas.accel_heap = _storage->device().create_bindless_array(needed);
}

void FallbackTlasBuilder::build(CommandList &commands, FallbackTlas &tlas,
                                uint instance_count,
                                luisa::span<const AccelBuildCommand::Modification> modifications,
                                luisa::span<const FallbackTlasBlas> resolved,
                                bool update_instance_buffer_only) noexcept {
    LUISA_ASSERT(instance_count > 0u, "The fallback RTX TLAS build needs at least one instance.");
    LUISA_ASSERT(modifications.size() == resolved.size(),
                 "The fallback RTX TLAS build got {} modifications and {} resolved "
                 "primitives.",
                 modifications.size(), resolved.size());
    // A TLAS that grows (or is built for the first time) gets a *fresh* append:
    // its region and its instance slice are never moved, so an offset a shader
    // descriptor was built with stays valid (see `FallbackRtxStorage`).  The
    // previous region is left in place, which is the deliberate cost of that
    // promise.
    auto fresh = !tlas.built || instance_count != tlas.instance_count;
    if (fresh) {
        tlas.region = _storage->plan_tlas(commands, instance_count);
        _storage->write_region_header(commands, tlas.region, region_flag_tlas);
        tlas.instance_count = instance_count;
        // A record the caller does not mention is a defined zero record, not
        // whatever the device memory held: the table and the Blas index of such
        // an instance are then reported by the validator instead of being read as
        // a random tree.
        tlas.records.assign(static_cast<size_t>(instance_count) * instance_u4,
                            make_uint4(0u, 0u, 0u, 0u));
        // The sentinel marks "this instance was never given a mesh": the table
        // refresh leaves its row clear for it (the entry is out of the directory),
        // `validate_accel` reports it, and a traversal skips a null table row
        // (fallback_rtx_layout.h) instead of descending into a tree that was never
        // built.
        tlas.directory_entry.assign(instance_count, invalid_offset);
        // The heap this TLAS resolves its regions through.  It is created here,
        // on the first build (the instance count is only known now), and it is a
        // member of `tlas`, so destroying the TLAS releases it (RAII).
        ensure_heap(tlas, instance_count);
    }

    // ---- apply the modifications to the host copy ---------------------------
    constexpr auto known_flags = AccelBuildCommand::Modification::flag_primitive |
                                 AccelBuildCommand::Modification::flag_transform |
                                 AccelBuildCommand::Modification::flag_opaque |
                                 AccelBuildCommand::Modification::flag_visibility |
                                 AccelBuildCommand::Modification::flag_user_id;
    luisa::vector<uint32_t> uploaded;
    uploaded.reserve(modifications.size());
    // Whether the heap gained an entry this build: it decides whether the update
    // command has to be recorded at all.
    auto heap_dirty = false;
    for (auto i = 0u; i < modifications.size(); i++) {
        auto &&m = modifications[i];
        if (m.index >= instance_count) {
            LUISA_ERROR("The fallback RTX TLAS build got a modification of instance {}, but the "
                        "acceleration structure holds {}.",
                        m.index, instance_count);
        }
        if ((m.flags & ~known_flags) != 0u) {
            LUISA_ERROR("The fallback RTX TLAS build got instance modification flags 0x{:08X}, "
                        "which this fallback does not implement (procedural primitives, curves "
                        "and motion instances fail closed).",
                        m.flags);
        }
        auto record = &tlas.records[static_cast<size_t>(m.index) * instance_u4];
        if ((m.flags & AccelBuildCommand::Modification::flag_transform) != 0u) {
            auto to_world = world_from_modification(m.affine);
            auto to_object = inverse(to_world);
            record[i_to_object + 0u] = u4_from_float4(matrix_row(to_object, 0u));
            record[i_to_object + 1u] = u4_from_float4(matrix_row(to_object, 1u));
            record[i_to_object + 2u] = u4_from_float4(matrix_row(to_object, 2u));
            record[i_to_world + 0u] = u4_from_float4(matrix_row(to_world, 0u));
            record[i_to_world + 1u] = u4_from_float4(matrix_row(to_world, 1u));
            record[i_to_world + 2u] = u4_from_float4(matrix_row(to_world, 2u));
        }
        auto flags = record[i_misc].w;
        if ((m.flags & AccelBuildCommand::Modification::flag_primitive) != 0u) {
            // The fallback only has triangle geometry, and this mirrors what the
            // hardware update kernel records for a triangle instance.
            flags |= instance_flag_disable_face_culling;
            tlas.directory_entry[m.index] = resolved[i].directory_entry;
            // Register the BLAS region in this TLAS' heap, at its slot
            // (fallback_rtx_layout.h).  The slot is host-known, and the table
            // refresh kernel copies it into the record's metadata lane, so it
            // travels with the record the same way the directory entry does.
            if (resolved[i].region_u4 != 0u) {
                tlas.accel_heap.emplace_on_update(
                    heap_first_blas_slot + m.index,
                    _storage->accel().view(resolved[i].region_base, resolved[i].region_u4));
                heap_dirty = true;
            }
        }
        if ((m.flags & AccelBuildCommand::Modification::flag_visibility) != 0u) {
            record[i_misc].y = m.vis_mask;
        }
        if ((m.flags & AccelBuildCommand::Modification::flag_user_id) != 0u) {
            record[i_misc].z = m.user_id;
        }
        if ((m.flags & AccelBuildCommand::Modification::flag_opaque) != 0u) {
            if ((flags & instance_flag_disable_face_culling) != 0u) {
                flags &= ~(instance_flag_disable_any_hit | instance_flag_enforce_any_hit);
                flags |= (m.flags & AccelBuildCommand::Modification::flag_opaque_on) != 0u ? instance_flag_disable_any_hit : instance_flag_enforce_any_hit;
            }
            // `opaque` is kept so that the flag survives a build; the fallback
            // has no any-hit shading, which the traversal header documents as a
            // limitation (fallback_rtx_layout.h).
            if ((m.flags & AccelBuildCommand::Modification::flag_opaque_on) != 0u) {
                flags |= instance_flag_opaque;
            } else {
                flags &= ~instance_flag_opaque;
            }
        }
        auto &misc = record[i_misc];
        // One table record per instance, and the instance points at *its own*
        // record, so a traversal may index the table with either the instance
        // index or `blas_index`; the invariant is restated in
        // fallback_rtx_layout.h and checked by `validate_blas` / `validate_accel`.
        misc.x = m.index;
        misc.w = flags;
        record[i_misc + 1u].x = tlas.directory_entry[m.index];
        // The heap slot of the referenced region.  A modification that carries a
        // primitive re-registers the region, so the record takes the slot the
        // heap entry was just written to (or 0 when there is no region); a
        // modification that does not keeps the slot an earlier build recorded -
        // the entry that build registered is still live, because a retired
        // acceleration buffer keeps its content.  Slot 0 is the null slot, and it
        // is what a traversal skips (fallback_rtx_layout.h).
        if ((static_cast<uint>(m.flags) & AccelBuildCommand::Modification::flag_primitive) != 0u) {
            record[i_misc + 1u].y = resolved[i].region_u4 != 0u ?
                                        heap_first_blas_slot + m.index :
                                        0u;
        }
        record[i_misc + 1u].z = 0u;
        record[i_misc + 1u].w = 0u;
        uploaded.push_back(m.index);
    }

    // ---- upload -------------------------------------------------------------
    // The first build of a TLAS has to initialize the whole instance buffer (an
    // instance the caller never mentions would otherwise be uninitialized device
    // memory); afterwards only the modified slots travel.
    if (fresh) {
        upload_every_record(commands, tlas);
    } else {
        upload_records(commands, tlas, luisa::span{uploaded});
    }

    // ---- the blas table, from the live instance records ---------------------
    auto &accel = _storage->accel();
    auto &instances = _storage->instances();
    auto &directory = _storage->blas_directory();
    // The heap: slot 1 is the TLAS' own region, so a traversal reads the tree
    // header through the heap exactly like it reads a BLAS region, and the ones
    // registered above are its BLAS regions.  The TLAS entry is (re)written on
    // every fresh build, because the shared acceleration buffer grows *by
    // reallocation*: the view has to name the buffer that is current now.
    if (fresh) {
        tlas.accel_heap.emplace_on_update(
            heap_tlas_slot,
            accel.view(tlas.region.base, tlas.region.region_u4()));
        heap_dirty = true;
    }
    // The update is a command of its own: the heap's slots (and the backend's
    // global bindless heap) are written by it, and it has to be ordered after
    // the growth copies `plan_tlas` may have recorded into `commands`.
    if (heap_dirty) {
        commands << tlas.accel_heap.update();
    }
    commands << _table_clear_kernel(accel, tlas.region.blas_table_base, tlas.region.blas_count)
                    .dispatch(tlas.region.blas_count);
    commands << _table_refresh_kernel(accel, instances, directory,
                                      tlas.region.instance_offset, tlas.region.blas_table_base,
                                      instance_count,
                                      static_cast<uint>(directory.size()))
                    .dispatch(instance_count);

    // ---- the tree -----------------------------------------------------------
    if (!update_instance_buffer_only) {
        _storage->reset_reduction(commands, tlas.region);
        commands << _prim_kernel(accel, instances, _storage->prims(), _storage->reduce(),
                                 tlas.region.instance_offset, tlas.region.prim_offset,
                                 tlas.region.blas_table_base,
                                 static_cast<uint>(accel.size()), instance_count,
                                 tlas.region.reduce_offset)
                        .dispatch(instance_count);
        _storage->build_tree(commands, tlas.region);
    }
    tlas.built = true;
}

}// namespace lc::fallback_rtx
