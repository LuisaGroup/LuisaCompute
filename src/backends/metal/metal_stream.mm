//
// Created by Mike Smith on 2021/7/26.
//

#import <backends/metal/metal_stream.h>

namespace luisa::compute::metal {

MetalStream::MetalStream(id<MTLDevice> device, uint max_command_buffers) noexcept
    : _handle{[device newCommandQueueWithMaxCommandBufferCount:max_command_buffers]},
      _upload_host_buffer_pool{device, host_buffer_size, true},
      _download_host_buffer_pool{device, host_buffer_size, false},
      _sem{dispatch_semaphore_create(max_command_buffers)} {}

MetalStream::~MetalStream() noexcept {
    synchronize();
    _handle = nullptr;
}

id<MTLCommandBuffer> MetalStream::command_buffer() noexcept {
    return [_handle commandBuffer];
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
      dispatch_semaphore_signal(_sem);
    }];
    dispatch_semaphore_wait(_sem, DISPATCH_TIME_FOREVER);
    _last = command_buffer;
    [command_buffer commit];
}

void MetalStream::synchronize() noexcept {
    if (auto last = [this]() noexcept {
            id<MTLCommandBuffer> last_cmd = _last;
            _last = nullptr;
            return last_cmd;
        }();
        last != nullptr) { [last waitUntilCompleted]; }
}

}
