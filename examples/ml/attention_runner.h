// Multi-Head Latent Attention (MLA) Example -- GPU runtime driver.
//
// Owns the device-side resources and the compile/dispatch logic for both
// attention paths (classic MHA and MLA, with optional cooperative-vector
// kernels).

#pragma once

#include <luisa/core/stl.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>

#include "attention_host_data.h"

namespace mla {

// All device buffers used by the attention example.
struct AttentionDeviceBuffers {
    luisa::compute::Buffer<float> h_buf;
    luisa::compute::Buffer<float> ckv_buf;
    luisa::compute::Buffer<float> krope_buf;
    luisa::compute::Buffer<float> q_buf;
    luisa::compute::Buffer<float> k_buf;
    luisa::compute::Buffer<float> v_buf;
    luisa::compute::Buffer<float> o_buf;

    luisa::compute::Buffer<float> wq_buf;
    luisa::compute::Buffer<float> wdkv_buf;
    luisa::compute::Buffer<float> wuk_buf;
    luisa::compute::Buffer<float> wuv_buf;
    luisa::compute::Buffer<float> wkr_buf;

    // ByteBuffer aliases for cooperative-vector access to weight/data buffers.
    luisa::compute::ByteBuffer wuv_byte_buf;
    luisa::compute::ByteBuffer wq_byte_buf;
    luisa::compute::ByteBuffer wdkv_byte_buf;
    luisa::compute::ByteBuffer wkr_byte_buf;
    luisa::compute::ByteBuffer h_byte_buf;
    luisa::compute::ByteBuffer q_byte_buf;
    luisa::compute::ByteBuffer ckv_byte_buf;
    luisa::compute::ByteBuffer krope_byte_buf;
};

// Allocate all device buffers (sizes come from attention_config.h).
[[nodiscard]] AttentionDeviceBuffers create_device_buffers(luisa::compute::Device &device);

// Upload all host tensors to the device in one batched command list.
void upload_host_data(luisa::compute::Stream &stream,
                      AttentionDeviceBuffers &buffers,
                      const AttentionHostData &host);

// Compile and dispatch the selected attention path.
void run_attention(luisa::compute::Device &device,
                   luisa::compute::Stream &stream,
                   AttentionDeviceBuffers &buffers,
                   bool use_mla,
                   bool cooperative_vector);

// Download the attention output O into `output` (resized to qkv_size).
void download_output(luisa::compute::Stream &stream,
                     AttentionDeviceBuffers &buffers,
                     luisa::vector<float> &output);

}// namespace mla
