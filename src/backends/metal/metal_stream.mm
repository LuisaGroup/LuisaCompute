//
// Created by Mike Smith on 2021/7/26.
//

#import <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

MetalStream::MetalStream(id<MTLDevice> device, uint max_command_buffers) noexcept
    : _handle{[device newCommandQueueWithMaxCommandBufferCount:max_command_buffers]},
      _upload_ring_buffer{device, ring_buffer_size, true},
      _download_ring_buffer{device, ring_buffer_size, false},
      _sem{max_command_buffers} {
    _command_buffer_desc = [[MTLCommandBufferDescriptor alloc] init];
    _command_buffer_desc.retainedReferences = YES;
    _command_buffer_desc.errorOptions = MTLCommandBufferErrorOptionNone;
}

MetalStream::~MetalStream() noexcept {
    synchronize();
    _handle = nullptr;
}

id<MTLCommandBuffer> MetalStream::command_buffer() noexcept {
    _sem.acquire();
    return [_handle commandBufferWithDescriptor:_command_buffer_desc];
}

void MetalStream::dispatch(id<MTLCommandBuffer> command_buffer) noexcept {
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      LUISA_VERBOSE_WITH_LOCATION(
          "Command buffer completed in {} ms.",
          (cb.GPUEndTime - cb.GPUStartTime) * 1000.0f);
      if (cb.error != nullptr) [[unlikely]] {
          LUISA_WARNING_WITH_LOCATION(
              "Error occurred when executing command buffer in stream: {}",
              [cb.error.description cStringUsingEncoding:NSUTF8StringEncoding]);
      }
      _sem.release();
    }];
    std::scoped_lock lock{_mutex};
    _last = command_buffer;
    [command_buffer commit];
}

void MetalStream::synchronize() noexcept {
    if (auto last = [this]() noexcept {
            std::scoped_lock lock{_mutex};
            id<MTLCommandBuffer> last_cmd = _last;
            _last = nullptr;
            return last_cmd;
        }();
        last != nullptr) { [last waitUntilCompleted]; }
}

}
