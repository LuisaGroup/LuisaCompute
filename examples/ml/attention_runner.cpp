// Multi-Head Latent Attention (MLA) Example -- GPU runtime driver.

#include "attention_config.h"
#include "attention_kernels.h"
#include "attention_runner.h"

#include <algorithm>
#include <numeric>
#include <random>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/sparse_buffer.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/sparse_command_list.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

namespace mla {

AttentionDeviceBuffers create_device_buffers(Device &device) {
    AttentionDeviceBuffers buffers;

    buffers.h_buf = device.create_buffer<float>(hidden_size);
    buffers.ckv_buf = device.create_buffer<float>(latent_size);
    buffers.krope_buf = device.create_buffer<float>(rope_size);
    buffers.q_buf = device.create_buffer<float>(qkv_size);
    buffers.k_buf = device.create_buffer<float>(qkv_size);
    buffers.v_buf = device.create_buffer<float>(qkv_size);
    buffers.o_buf = device.create_buffer<float>(qkv_size);

    buffers.wq_buf = device.create_buffer<float>(wq_size);
    buffers.wdkv_buf = device.create_buffer<float>(wdkv_size);
    buffers.wuk_buf = device.create_buffer<float>(wuk_size);
    buffers.wuv_buf = device.create_buffer<float>(wuv_size);
    buffers.wkr_buf = device.create_buffer<float>(wkr_size);

    // ByteBuffer for cooperative-vector access to weight/data buffers.
    buffers.wuv_byte_buf = device.create_byte_buffer(wuv_size * sizeof(float));
    buffers.wq_byte_buf = device.create_byte_buffer(wq_size * sizeof(float));
    buffers.wdkv_byte_buf = device.create_byte_buffer(wdkv_size * sizeof(float));
    buffers.wkr_byte_buf = device.create_byte_buffer(wkr_size * sizeof(float));
    buffers.h_byte_buf = device.create_byte_buffer(hidden_size * sizeof(float));
    buffers.q_byte_buf = device.create_byte_buffer(qkv_size * sizeof(float));
    buffers.ckv_byte_buf = device.create_byte_buffer(latent_size * sizeof(float));
    buffers.krope_byte_buf = device.create_byte_buffer(rope_size * sizeof(float));

    return buffers;
}

void upload_host_data(Stream &stream, AttentionDeviceBuffers &buffers, const AttentionHostData &host) {
    // Upload the float buffers once, then derive the byte-buffer aliases
    // device-side (halves upload traffic). All in one batch.
    CommandList upload = CommandList::create();
    upload << buffers.q_buf.copy_from(luisa::span{host.q})
           << buffers.k_buf.copy_from(luisa::span{host.k})
           << buffers.v_buf.copy_from(luisa::span{host.v})
           << buffers.h_buf.copy_from(luisa::span{host.h})
           << buffers.wq_buf.copy_from(luisa::span{host.wq})
           << buffers.wdkv_buf.copy_from(luisa::span{host.wdkv})
           << buffers.wuk_buf.copy_from(luisa::span{host.wuk})
           << buffers.wuv_buf.copy_from(luisa::span{host.wuv})
           << buffers.wkr_buf.copy_from(luisa::span{host.wkr})
           << buffers.wuv_byte_buf.copy_from(buffers.wuv_buf)
           << buffers.wq_byte_buf.copy_from(buffers.wq_buf)
           << buffers.wdkv_byte_buf.copy_from(buffers.wdkv_buf)
           << buffers.wkr_byte_buf.copy_from(buffers.wkr_buf)
           << buffers.h_byte_buf.copy_from(buffers.h_buf)
           << buffers.q_byte_buf.copy_from(buffers.q_buf);
    stream << upload.commit() << synchronize();
}

namespace {

void run_mla(Device &device, Stream &stream, AttentionDeviceBuffers &buffers,
               ShaderOption &opt, Clock &compile_clock, bool cooperative) {
    if (cooperative) {
        LUISA_INFO("Compiling MLA cooperative kernels ...");
    } else {
        LUISA_INFO("Compiling MLA kernels ...");
    }

    // Compile: use ternary to select the template instantiation.
    opt.name = cooperative ? "mla_project_q_coop" : "mla_project_q";
    auto project_q_shader = cooperative
        ? device.compile<1>(create_project_q_kernel<true>(), opt)
        : device.compile<1>(create_project_q_kernel<false>(), opt);

    opt.name = cooperative ? "mla_project_kv_coop" : "mla_project_kv";
    auto project_kv_shader = cooperative
        ? device.compile(create_project_kv_kernel<true>(), opt)
        : device.compile(create_project_kv_kernel<false>(), opt);

    opt.name = cooperative ? "mla_online_attention_coop" : "mla_online_attention";
    auto online_attention_shader = cooperative
        ? device.compile(create_online_attention_kernel<true>(), opt)
        : device.compile(create_online_attention_kernel<false>(), opt);

    double compile_ms = compile_clock.toc();
    if (cooperative) {
        LUISA_INFO("  MLA cooperative kernels compiled in {:.2f} ms", compile_ms);
    } else {
        LUISA_INFO("  MLA kernels compiled in {:.2f} ms", compile_ms);
    }

    // Warm-up dispatch (not measured).
    {
        CommandList warmup = CommandList::create();
        warmup << online_attention_shader(buffers.q_buf, buffers.ckv_buf, buffers.wuk_buf,
                                          buffers.krope_buf, buffers.wuv_buf, buffers.o_buf,
                                          buffers.q_byte_buf, buffers.ckv_byte_buf,
                                          buffers.krope_byte_buf, buffers.wuv_byte_buf)
                      .dispatch(batch * num_heads * seq_len);
        stream << warmup.commit() << synchronize();
    }

    if (cooperative) {
        LUISA_INFO("Dispatching MLA cooperative GPU kernels ...");
    } else {
        LUISA_INFO("Dispatching MLA GPU kernels ...");
    }

    Clock dispatch_clock;
    CommandList cmd_list = CommandList::create();

    // Project Q: unified signature always passes all params (ByteBuf ignored in fallback).
    cmd_list << project_q_shader(buffers.h_buf, buffers.q_buf, buffers.wq_buf, buffers.wq_byte_buf)
                    .dispatch(batch * seq_len * project_q_block_size)
             << project_kv_shader(buffers.h_buf, buffers.ckv_buf, buffers.krope_buf,
                                  buffers.wdkv_buf, buffers.wkr_buf,
                                  buffers.wdkv_byte_buf, buffers.wkr_byte_buf)
                    .dispatch(batch * seq_len * project_kv_block_size);

    // Refresh the byte-buffer aliases of the projected tensors device-side
    // so the cooperative-vector loads in the attention kernel see them.
    // (No-op in fallback path but harmless — the copy is a device-side alias.)
    if (cooperative) {
        cmd_list << buffers.q_byte_buf.copy_from(buffers.q_buf)
                 << buffers.ckv_byte_buf.copy_from(buffers.ckv_buf)
                 << buffers.krope_byte_buf.copy_from(buffers.krope_buf);
    }

    // Online attention: unified signature always passes all ByteBuf params.
    cmd_list << online_attention_shader(buffers.q_buf, buffers.ckv_buf, buffers.wuk_buf,
                                         buffers.krope_buf, buffers.wuv_buf, buffers.o_buf,
                                         buffers.q_byte_buf, buffers.ckv_byte_buf,
                                         buffers.krope_byte_buf, buffers.wuv_byte_buf)
                    .dispatch(batch * num_heads * seq_len);

    stream << cmd_list.commit() << synchronize();
    double dispatch_ms = dispatch_clock.toc();
    if (cooperative) {
        LUISA_INFO("  MLA cooperative GPU dispatch + sync: {:.2f} ms", dispatch_ms);
    } else {
        LUISA_INFO("  MLA GPU dispatch + sync: {:.2f} ms", dispatch_ms);
    }
}

void run_mha(Device &device, Stream &stream, AttentionDeviceBuffers &buffers, ShaderOption &opt, Clock &compile_clock) {
    LUISA_INFO("Compiling MHA kernels ...");

    opt.name = "mha_online_attention";
    auto mha_online_shader = device.compile(create_mha_online_attention_kernel(), opt);

    double compile_ms = compile_clock.toc();
    LUISA_INFO("  MHA kernels compiled in {:.2f} ms", compile_ms);

    LUISA_INFO("Dispatching MHA GPU kernels ...");
    Clock dispatch_clock;
    CommandList cmd_list = CommandList::create();
    cmd_list << mha_online_shader(buffers.q_buf, buffers.k_buf, buffers.v_buf, buffers.o_buf)
                    .dispatch(batch * num_heads * seq_len);
    stream << cmd_list.commit() << synchronize();
    double dispatch_ms = dispatch_clock.toc();
    LUISA_INFO("  MHA GPU dispatch + sync: {:.2f} ms", dispatch_ms);
}

}// namespace

void run_attention(Device &device, Stream &stream, AttentionDeviceBuffers &buffers,
                   bool use_mla, bool cooperative_vector) {
    ShaderOption opt{.enable_debug_info = false};
    Clock compile_clock;

    if (use_mla) {
        run_mla(device, stream, buffers, opt, compile_clock, cooperative_vector);
    } else {
        run_mha(device, stream, buffers, opt, compile_clock);
    }
}

bool run_paged_attention(Device &device, Stream &stream, AttentionDeviceBuffers &buffers) {
    // A paged_attention_block_size-thread block spans consecutive i of the
    // same (b, h), so the block-table read in the attention kernel is
    // block-uniform.
    static_assert(seq_len % paged_attention_block_size == 0u,
                  "seq_len must be a multiple of the block size");

    // -- Geometry discovery --------------------------------------------------
    // The sparse tile size is chosen by the device and only known after
    // creating a sparse buffer; probe it before sizing the real pools.
    auto probe = device.create_sparse_buffer<float>(1u);
    if (!probe) [[unlikely]] {
        LUISA_WARNING("Paged attention requires a sparse-buffer-capable backend "
                      "(vk/cuda/dx/hip); falling back to dense MHA.");
        return false;
    }
    const auto tile_bytes = probe.tile_size_bytes();
    const auto elems_per_page = static_cast<uint>(probe.tile_size());
    constexpr auto elems_per_token_page = num_heads * head_dim;// [h][d] slice per token
    const auto token_capacity = elems_per_page / elems_per_token_page;
    if (token_capacity == 0u) [[unlikely]] {
        LUISA_WARNING("Sparse tile ({} B) cannot hold one token of all heads ({} floats); "
                      "falling back to dense MHA.",
                      tile_bytes, elems_per_token_page);
        return false;
    }
    // tokens_per_page = vLLM block_size: the largest divisor of seq_len that
    // fits one tile, so every page is fully packed with whole tokens.
    const auto tokens_per_page = paged_largest_divisor(seq_len, token_capacity);
    const auto pages_per_seq = seq_len / tokens_per_page;
    const auto total_pages = batch * pages_per_seq;
    // The kernel stages sub-tiles of (sub_tile * head_dim) floats filled
    // cooperatively by one block; a degenerate geometry that cannot fill the
    // sub-tile with the block falls back to dense MHA instead of asserting.
    if (const auto sub_tile = paged_sub_tile(tokens_per_page);
        (sub_tile * head_dim) % paged_attention_block_size != 0u) [[unlikely]] {
        LUISA_WARNING("Paged geometry (tokens/page={}, sub-tile={}) is incompatible with "
                      "the {}-thread attention block; falling back to dense MHA.",
                      tokens_per_page, sub_tile, paged_attention_block_size);
        return false;
    }
    LUISA_INFO("Paged KV: tile={} B, tokens/page={} (vLLM block_size), pages/seq={}, "
               "total pages={}, layout [page][h][t][d]",
               tile_bytes, tokens_per_page, pages_per_seq, total_pages);
    if (pages_per_seq == 1u) {
        LUISA_INFO("  Note: device tile granularity forces single-page sequences; "
                   "the block table is still a shuffled permutation.");
    }
    if (const auto padding = elems_per_page - tokens_per_page * elems_per_token_page) {
        LUISA_INFO("  Internal fragmentation: {} unused floats per tile", padding);
    }

    // -- Sparse pools, heaps, block-table buffer -----------------------------
    // One SparseBufferHeap per physical page: the backend sparse-residency
    // registry forbids one heap backing multiple live ranges, so a heap
    // allocation IS a KV block here (the vLLM BlockPool block analog).
    // Heaps are declared before the buffers they back so they are destroyed
    // last (reverse declaration order), never outliving a mapping.
    const auto pool_elems = static_cast<size_t>(total_pages) * elems_per_page;
    luisa::vector<SparseBufferHeap> heaps_k(total_pages);
    luisa::vector<SparseBufferHeap> heaps_v(total_pages);
    for (uint p = 0u; p < total_pages; ++p) {
        heaps_k[p] = device.allocate_sparse_buffer_heap(tile_bytes);
        heaps_v[p] = device.allocate_sparse_buffer_heap(tile_bytes);
    }
    auto paged_k = device.create_sparse_buffer<float>(pool_elems);
    auto paged_v = device.create_sparse_buffer<float>(pool_elems);
    auto block_table_buf = device.create_buffer<uint>(total_pages);
    if (!paged_k || !paged_v || !block_table_buf ||
        !std::all_of(heaps_k.begin(), heaps_k.end(), [](auto &h) { return static_cast<bool>(h); }) ||
        !std::all_of(heaps_v.begin(), heaps_v.end(), [](auto &h) { return static_cast<bool>(h); })) [[unlikely]] {
        LUISA_WARNING("Failed to allocate sparse paged-KV resources; "
                      "falling back to dense MHA.");
        return false;
    }

    // -- Block table (the vLLM KVCacheManager/BlockPool analog) --------------
    // A shuffled permutation of [0, total_pages) so physical pages are
    // maximally scattered, proving the indirection path correct.
    luisa::vector<uint> block_table(total_pages);
    std::iota(block_table.begin(), block_table.end(), 0u);
    std::shuffle(block_table.begin(), block_table.end(), std::mt19937{42});
    // No host synchronize here: stream commands execute in FIFO order, so the
    // single synchronize after the sparse-map commit below also covers this
    // upload (batched submission, one host stall -- lc_optimize sec.8).
    stream << block_table_buf.copy_from(luisa::span{block_table});
    {
        luisa::string mapping;
        for (uint p = 0u; p < pages_per_seq; ++p) {
            if (p != 0u) { mapping.append(", "); }
            mapping.append(luisa::format("{}", block_table[p]));
        }
        LUISA_INFO("  Block table, seq 0: logical [0..{}) -> physical [{}]",
                   pages_per_seq, mapping);
    }

    // -- Block allocation: map one sparse tile per physical page -------------
    // Per-page map ops mirror vLLM's incremental per-block allocation.
    {
        SparseCommandList map;
        for (uint p = 0u; p < total_pages; ++p) {
            map << paged_k.map_tile(p, 1u, heaps_k[p])
                << paged_v.map_tile(p, 1u, heaps_v[p]);
        }
        // The map commit must complete before any kernel touches the tiles.
        stream << map.commit() << synchronize();
    }

    // -- Compile -------------------------------------------------------------
    LUISA_INFO("Compiling paged attention kernels ...");
    Clock compile_clock;
    ShaderOption opt{.enable_debug_info = false};
    opt.name = "paged_reshape_kv";
    auto reshape_shader = device.compile(create_reshape_kv_to_paged_kernel(), opt);
    opt.name = "paged_attention";
    auto paged_shader = device.compile(
        create_paged_attention_kernel(tokens_per_page, elems_per_page), opt);
    LUISA_INFO("  Paged attention kernels compiled in {:.2f} ms", compile_clock.toc());

    // -- Warm-up dispatch (not measured) -------------------------------------
    {
        CommandList warmup = CommandList::create();
        warmup << reshape_shader(buffers.k_buf, buffers.v_buf,
                                 paged_k.view(), paged_v.view(),
                                 block_table_buf,
                                 tokens_per_page, pages_per_seq, elems_per_page)
                      .dispatch(qkv_size)
               << paged_shader(buffers.q_buf, paged_k.view(), paged_v.view(),
                               buffers.o_buf, block_table_buf, pages_per_seq)
                      .dispatch(batch * num_heads * seq_len);
        stream << warmup.commit() << synchronize();
    }

    // -- Timed dispatch ------------------------------------------------------
    LUISA_INFO("Dispatching paged attention GPU kernels ...");
    Clock dispatch_clock;
    {
        CommandList cmd_list = CommandList::create();
        cmd_list << reshape_shader(buffers.k_buf, buffers.v_buf,
                                   paged_k.view(), paged_v.view(),
                                   block_table_buf,
                                   tokens_per_page, pages_per_seq, elems_per_page)
                        .dispatch(qkv_size)
                 << paged_shader(buffers.q_buf, paged_k.view(), paged_v.view(),
                                 buffers.o_buf, block_table_buf, pages_per_seq)
                        .dispatch(batch * num_heads * seq_len);
        stream << cmd_list.commit() << synchronize();
    }
    LUISA_INFO("  Paged GPU dispatch + sync: {:.2f} ms", dispatch_clock.toc());

    // -- Eviction / reuse demo (vLLM BlockPool free-queue analog) ------------
    const auto last_page = total_pages - 1u;
    {
        SparseCommandList evict;
        evict << paged_k.unmap_tile(last_page, 1u);
        stream << evict.commit() << synchronize();
    }
    LUISA_INFO("  KV block {} evicted (ref_cnt -> 0)", last_page);
    {
        SparseCommandList realloc_cmd;
        realloc_cmd << paged_k.map_tile(last_page, 1u, heaps_k[last_page]);
        stream << realloc_cmd.commit() << synchronize();
    }
    LUISA_INFO("  KV block {} reallocated from the block-pool free queue", last_page);

    // -- Explicit teardown ---------------------------------------------------
    // Vulkan requires no active mappings when a sparse resource or heap is
    // destroyed; unmap everything before the locals go out of scope.
    {
        SparseCommandList unmap;
        unmap << paged_k.unmap_tile(0u, total_pages)
              << paged_v.unmap_tile(0u, total_pages);
        stream << unmap.commit() << synchronize();
    }
    return true;
}

void download_output(Stream &stream, AttentionDeviceBuffers &buffers, luisa::vector<float> &output) {
    output.resize(qkv_size);
    Clock download_clock;
    stream << buffers.o_buf.copy_to(luisa::span{output}) << synchronize();
    double download_ms = download_clock.toc();
    LUISA_INFO("  Download results: {:.2f} ms", download_ms);
}

}// namespace mla
