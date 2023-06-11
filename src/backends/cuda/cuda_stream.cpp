//
// Created by Mike on 8/1/2021.
//

#include <cstdlib>
#include <mutex>

#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/logging.h>
#include "cuda_error.h"
#include "cuda_command_encoder.h"
#include "cuda_callback_context.h"
#include "cuda_device.h"
#include "cuda_stream.h"

namespace luisa::compute::cuda {

[[nodiscard]] auto cuda_stream_assign_uid() noexcept {
    static std::atomic_uint uid{0u};
    return uid.fetch_add(1u, std::memory_order_relaxed);
}

CUDAStream::CUDAStream(CUDADevice *device) noexcept
    : _device{device}, _upload_pool{64_M, true} {
    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
}

CUDAStream::~CUDAStream() noexcept {
    _indirect = nullptr;
    LUISA_CHECK_CUDA(cuStreamDestroy(_stream));
}

void CUDAStream::synchronize() noexcept {
    LUISA_CHECK_CUDA(cuStreamSynchronize(_stream));
}

void CUDAStream::callback(CUDAStream::CallbackContainer &&callbacks) noexcept {
    if (!callbacks.empty()) {
        {
            std::scoped_lock lock{_callback_mutex};
            _callback_lists.emplace(std::move(callbacks));
        }
        LUISA_CHECK_CUDA(cuLaunchHostFunc(
            _stream, [](void *p) noexcept {
                constexpr auto pop = [](auto stream) -> luisa::vector<CUDACallbackContext *> {
                    std::scoped_lock lock{stream->_callback_mutex};
                    if (stream->_callback_lists.empty()) [[unlikely]] {
                        LUISA_WARNING_WITH_LOCATION(
                            "Fetching stream callback from empty queue.");
                        return {};
                    }
                    auto callbacks = std::move(stream->_callback_lists.front());
                    stream->_callback_lists.pop();
                    return callbacks;
                };
                auto stream = static_cast<CUDAStream *>(p);
                auto callbacks = pop(stream);
                for (auto callback : callbacks) {
                    callback->recycle();
                }
            },
            this));
    }
}

void CUDAStream::signal(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuEventRecord(event, _stream));
}

void CUDAStream::wait(CUevent event) noexcept {
    LUISA_CHECK_CUDA(cuStreamWaitEvent(_stream, event, CU_EVENT_WAIT_DEFAULT));
}

void CUDAStream::set_name(luisa::string &&name) noexcept {
    if (std::scoped_lock lock{_indirect_creation_mutex};
        _indirect != nullptr) {
        _indirect->set_name(name);
    }
    std::scoped_lock lock{_name_mutex};
    nvtxNameCuStreamA(_stream, name.c_str());
    _name = std::move(name);
}

void CUDAStream::dispatch(CommandList &&command_list) noexcept {
    CUDACommandEncoder encoder{this};
    auto commands = command_list.steal_commands();
    auto callbacks = command_list.steal_callbacks();
    {
        std::scoped_lock lock{_dispatch_mutex};
        for (auto &cmd : commands) { cmd->accept(encoder); }
        encoder.commit(std::move(callbacks));
    }
}

CUDAIndirectDispatchStream &CUDAStream::indirect() noexcept {
    auto newly_created = false;
    if (std::scoped_lock lock{_indirect_creation_mutex};
        _indirect == nullptr) {
        _indirect = luisa::make_unique<CUDAIndirectDispatchStream>(this);
        newly_created = true;
    }
    if (newly_created) {
        auto name = [this] {
            std::scoped_lock lock{_name_mutex};
            return _name;
        }();
        if (!name.empty()) {
            _indirect->set_name(name);
        }
    }
    return *_indirect;
}

class CUDAIndirectDispatchStream::TaskContext {

private:
    CUDAIndirectDispatchStream *_self;
    CUDAIndirectDispatchStream::Task *_task;
    uint64_t _value;

private:
    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<TaskContext, true> pool;
        return pool;
    }

public:
    TaskContext(CUDAIndirectDispatchStream *self,
                CUDAIndirectDispatchStream::Task *task,
                uint64_t value) noexcept
        : _self{self}, _task{task}, _value{value} {}

    [[nodiscard]] auto self() const noexcept { return _self; }
    [[nodiscard]] auto task() const noexcept { return _task; }
    [[nodiscard]] auto value() const noexcept { return _value; }

    [[nodiscard]] static auto create(CUDAIndirectDispatchStream *self,
                                     CUDAIndirectDispatchStream::Task *task,
                                     uint64_t value) noexcept {
        return _pool().create(self, task, value);
    }

    void recycle() noexcept {
        _pool().destroy(this);
    }
};

CUDAIndirectDispatchStream::CUDAIndirectDispatchStream(CUDAStream *parent) noexcept
    : _parent{parent}, _event_value{0u}, _stop_requested{false} {

    LUISA_CHECK_CUDA(cuStreamCreate(&_stream, CU_STREAM_NON_BLOCKING));
    _event = parent->device()->create_timeline_event(_stream);

    _thread = std::thread{[this] {
        for (;;) {
            // wait for the next task
            auto ctx = [this]() noexcept -> TaskContext * {
                std::unique_lock lock{_queue_mutex};
                _cv.wait(lock, [this] { return _stop_requested || !_task_contexts.empty(); });
                if (_stop_requested) {
                    if (!_task_contexts.empty()) {
                        LUISA_WARNING_WITH_LOCATION(
                            "CUDA indirect dispatch stream stopped "
                            "with {} task(s) remaining. Did you forget "
                            "to synchronize the stream?",
                            _task_contexts.size());
                    }
                    return nullptr;
                }
                auto ctx = _task_contexts.front();
                _task_contexts.pop();
                return ctx;
            }();
            // stop if requested
            if (ctx == nullptr) { break; }
            _parent->device()->with_handle([&] {
                // wait for event
                LUISA_CHECK_CUDA(cuLaunchHostFunc(
                    _stream, [](void *p) noexcept {
                        LUISA_INFO("executing task");
                    },
                    nullptr));
                // execute the task
                ctx->task()->execute(_stream);
                // notify parent that the task is done
                LUISA_CHECK_CUDA(cuStreamWriteValue64(
                    _stream, _event, ctx->value(),
                    CU_STREAM_WRITE_VALUE_DEFAULT));
                ctx->recycle();
            });
        }
    }};
}

CUDAIndirectDispatchStream::~CUDAIndirectDispatchStream() noexcept {
    {
        std::scoped_lock lock{_queue_mutex};
        _stop_requested = true;
    }
    _cv.notify_one();
    _thread.join();
    _parent->device()->destroy_timeline_event(_event);
    LUISA_CHECK_CUDA(cuStreamDestroy(_stream));
}

void CUDAIndirectDispatchStream::enqueue(Task *command) noexcept {
    auto value = ++_event_value;
    auto ctx = TaskContext::create(this, command, value);
    LUISA_CHECK_CUDA(cuLaunchHostFunc(
        _parent->handle(), [](void *p) noexcept {
            auto ctx = static_cast<TaskContext *>(p);
            {
                std::scoped_lock lock{ctx->self()->_queue_mutex};
                ctx->self()->_task_contexts.emplace(ctx);
            }
            ctx->self()->_cv.notify_one();
        },
        ctx));
    // make the parent wait for the task to be done
    LUISA_CHECK_CUDA(cuStreamWaitValue64(
        _parent->handle(), _event, value,
        CU_STREAM_WAIT_VALUE_GEQ));
}

void CUDAIndirectDispatchStream::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        nvtxNameCuStreamA(_stream, nullptr);
    } else {
        auto tagged_name = luisa::format("{} (indirect dispatch)", name);
        nvtxNameCuStreamA(_stream, tagged_name.c_str());
    }
}

}// namespace luisa::compute::cuda

