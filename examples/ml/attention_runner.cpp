// Multi-Head Latent Attention (MLA) Example -- GPU runtime driver.

#include "attention_config.h"
#include "attention_kernels.h"
#include "attention_runner.h"
#include "paged_attention.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>
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
    // same (b, h), so the page-index read in the attention kernel is
    // block-uniform.
    static_assert(seq_len % paged_attention_block_size == 0u,
                  "seq_len must be a multiple of the block size");

    // The logical page count is derived inside prepare() from the discovered
    // geometry (pages_per_seq depends on the device tile size):
    // total_pages = batch * pages_per_seq.
    PagedAttention paged{device, stream};
    if (!paged.prepare(batch)) [[unlikely]] {
        LUISA_WARNING("Falling back to dense MHA.");
        return false;
    }

    // -- Warm-up dispatch (not measured) -------------------------------------
    {
        CommandList warmup = CommandList::create();
        paged.compute(warmup, buffers.q_buf, buffers.k_buf, buffers.v_buf, buffers.o_buf);
        stream << warmup.commit() << synchronize();
    }

    // -- Re-allocation demo ---------------------------------------------------
    // Grow the pool beyond its current capacity: a larger sparse buffer is
    // created, the heap pool is re-allocated into it (live pages copied), and
    // the old buffer + heaps are destroyed through CommandList::add_callback
    // so they outlive every GPU reference.
    LUISA_INFO("Re-allocating paged KV pool ({} -> {} pages) ...",
               paged.capacity(), paged.capacity() * 2u);
    if (!paged.reserve(paged.capacity() * 2u)) [[unlikely]] {
        LUISA_WARNING("Pool re-allocation failed; falling back to dense MHA.");
        return false;
    }

    // -- Timed dispatch ------------------------------------------------------
    LUISA_INFO("Dispatching paged attention GPU kernels ...");
    Clock dispatch_clock;
    {
        CommandList cmd_list = CommandList::create();
        paged.compute(cmd_list, buffers.q_buf, buffers.k_buf, buffers.v_buf, buffers.o_buf);
        stream << cmd_list.commit() << synchronize();
    }
    LUISA_INFO("  Paged GPU dispatch + sync: {:.2f} ms", dispatch_clock.toc());

    // -- Eviction / reuse demo (vLLM BlockPool free-queue analog) ------------
    // Runs after the verified dispatch: unmap -> remap of a live tile leaves
    // stale residency on the dx backend (verified output only depends on the
    // dispatch above), so the churn demo is presentation-only, like the
    // original flattened implementation.
    const auto last_page = paged.total_pages() - 1u;
    if (paged.evict_page(last_page)) {
        LUISA_INFO("  KV block {} evicted (ref_cnt -> 0)", last_page);
        if (paged.allocate_page(last_page)) {
            LUISA_INFO("  KV block {} reallocated from the block-pool free queue", last_page);
        }
    } else {
        LUISA_INFO("  KV block {} shares its tile with live pages; eviction skipped", last_page);
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
