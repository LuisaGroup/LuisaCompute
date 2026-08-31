#include "fallback_command_queue.h"

#include <luisa/core/stl/vector.h>
#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/logging.h>

#ifdef LUISA_FALLBACK_USE_AKARI_THREAD_POOL

#include <optional>
#ifdef LUISA_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace luisa::compute::fallback {

struct Thread {

#ifdef LUISA_PLATFORM_WINDOWS
    HANDLE handle{};
#else
    pthread_t handle{};
#endif
    luisa::move_only_function<void()> f;

    // Large generated kernels can require more than 4 MiB of stack when the
    // fallback compiler deliberately uses its bounded compile-time O0 path.
    // Stack pages are committed on demand, so reserving 16 MiB keeps ordinary
    // kernels cheap while allowing those oversized kernels to execute safely.
    static constexpr auto STACK_SIZE = 16_M;

    template<class F>
    explicit Thread(uint tid [[maybe_unused]], F &&f) noexcept : f{std::forward<F>(f)} {

#ifdef LUISA_PLATFORM_WINDOWS
        handle = CreateThread(nullptr, STACK_SIZE, [](void *arg) -> DWORD {
            auto *f = static_cast<luisa::move_only_function<void()> *>(arg);
            (*f)();
            return 0; }, &this->f, 0, nullptr);
        if (handle == nullptr) {
            LUISA_ERROR_WITH_LOCATION("Failed to create thread");
        }
        auto thread_group = tid / 64;
        GROUP_AFFINITY affinity{};
        affinity.Group = static_cast<WORD>(thread_group);
        affinity.Mask = 1ull << (tid % 64);
        if (SetThreadGroupAffinity(handle, &affinity, nullptr) == 0) {
            LUISA_WARNING_WITH_LOCATION("Failed to pin thread to group");
        }
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, STACK_SIZE);
#ifdef LUISA_PLATFORM_APPLE
        pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        if (pthread_create(&handle, &attr, [](void *arg) noexcept -> void * {
            auto *f = static_cast<luisa::move_only_function<void()> *>(arg);
            (*f)();
            return nullptr; }, &this->f) != 0) {
            LUISA_ERROR_WITH_LOCATION("Failed to create thread");
        }
        pthread_attr_destroy(&attr);
#if !defined(LUISA_PLATFORM_APPLE)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid, &cpuset);
        if (pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) != 0) {
            LUISA_WARNING_WITH_LOCATION("Failed to create thread");
        }
#endif
#endif
    }
    Thread(const Thread &) = delete;
    Thread(Thread &&) = delete;
    Thread &operator=(const Thread &) = delete;
    Thread &operator=(Thread &&) = delete;

    ~Thread() noexcept {
#ifdef LUISA_PLATFORM_WINDOWS
        WaitForSingleObject(handle, INFINITE);
        CloseHandle(handle);
#else
        pthread_join(handle, nullptr);
#endif
    }
};

class AkrThreadPool {

public:
    struct alignas(64) ParallelFor {
        uint count = 0u;
        uint block_size = 0u;
        luisa::move_only_function<void(uint)> *task = nullptr;
    };

private:
    luisa::vector<luisa::unique_ptr<Thread>> _threads;
    std::mutex _task_mutex, _submit_mutex;
    std::condition_variable _has_work, _work_done;
    ParallelFor _parallel_for = {};

#ifdef __cpp_lib_hardware_interference_size
    static constexpr auto cache_line_size = std::hardware_destructive_interference_size;
#else
    static constexpr auto cache_line_size = 64u;
#endif
    alignas(cache_line_size) std::atomic_uint32_t _item_count = 0u;
    alignas(cache_line_size) std::atomic_uint64_t _task_generation = 0u;

    std::atomic_uint32_t _thread_working;
    std::atomic_bool _stopped = false;

public:
    explicit AkrThreadPool(size_t n_threads) noexcept {
        for (size_t tid = 0; tid < n_threads; tid++) {
            _threads.emplace_back(std::move(luisa::make_unique<Thread>(tid, [this] {
                auto last_task_generation = 0;
                while (!_stopped.load(std::memory_order_seq_cst)) {
                    std::unique_lock lock{_task_mutex};
                    _has_work.wait(lock, [this, &last_task_generation] { return (last_task_generation != _task_generation.load() && _parallel_for.task != nullptr) || _stopped.load(std::memory_order_seq_cst); });
                    if (_stopped.load(std::memory_order_seq_cst)) { return; }
                    last_task_generation = _task_generation.load(std::memory_order_seq_cst);
                    auto [count, block_size, task] = _parallel_for;
                    lock.unlock();
                    while (_item_count < count) {
                        auto i = _item_count.fetch_add(block_size, std::memory_order_seq_cst);
                        for (uint j = i; j < std::min<uint>(i + block_size, count); j++) {
                            (*task)(j);
                        }
                    }
                    if (_thread_working.fetch_sub(1, std::memory_order_seq_cst) == 1) {
                        _work_done.notify_all();
                    }
                }
            })));
        }
    }
    ~AkrThreadPool() noexcept {
        _stopped.store(true, std::memory_order_seq_cst);
        _has_work.notify_all();
        _threads.clear();
    }
    void parallel_for(uint count, luisa::move_only_function<void(uint)> &&task) noexcept {
        std::scoped_lock _lk{_submit_mutex};
        std::unique_lock lock{_task_mutex};
        _thread_working.store(_threads.size(), std::memory_order_seq_cst);
        _parallel_for = {count, 1u, &task};
        _item_count.store(0, std::memory_order_seq_cst);
        _task_generation.fetch_add(1, std::memory_order_seq_cst);
        _has_work.notify_all();
        _work_done.wait(lock, [this] { return _thread_working.load(std::memory_order_seq_cst) == 0; });
        _parallel_for = {};
    }
};

}// namespace luisa::compute::fallback

#endif

namespace luisa::compute::fallback {

inline void FallbackCommandQueue::_run_dispatch_loop() noexcept {
    // wait and fetch tasks
    for (;;) {
        auto task = [this] {
            std::unique_lock lock{_mutex};
            _cv.wait(lock, [this] { return !_tasks.empty(); });
            auto task = std::move(_tasks.front());
            _tasks.pop();
            return task;
        }();
        // we use a null task to signal the end of the dispatcher
        if (!task) { break; }
        // good to go
        task();
        // count the finish of a task
        _total_finish_count.fetch_add(1u);
    }
#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
    if (_dispatch_queue != nullptr) {
        dispatch_release(_dispatch_queue);
    }
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
    _worker_pool.reset();
#endif
}

inline void FallbackCommandQueue::_wait_for_task_queue_available() const noexcept {
    if (_in_flight_limit == 0u) { return; }
    auto last_enqueue_count = _total_enqueue_count.load();
    while (_total_finish_count.load() + _in_flight_limit <= last_enqueue_count) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(50us);
    }
}

inline void FallbackCommandQueue::_enqueue_task_no_wait(luisa::move_only_function<void()> &&task) noexcept {
    _total_enqueue_count.fetch_add(1u);
    {
        std::scoped_lock lock{_mutex};
        _tasks.push(std::move(task));
    }
    _cv.notify_one();
}

FallbackCommandQueue::FallbackCommandQueue(size_t in_flight_limit, size_t num_threads [[maybe_unused]]) noexcept
    : _in_flight_limit{in_flight_limit}, _worker_count{num_threads} {
    if (_worker_count == 0u) {
        _worker_count = std::thread::hardware_concurrency();
    }
    _dispatcher = std::thread{[this] { _run_dispatch_loop(); }};
}

FallbackCommandQueue::~FallbackCommandQueue() noexcept {
    _enqueue_task_no_wait({});// signal the end of the dispatcher
    _dispatcher.join();
}

void FallbackCommandQueue::synchronize() noexcept {
    auto finished = false;
    _enqueue_task_no_wait([this, &finished] {
        {
            std::scoped_lock lock{_synchronize_mutex};
            finished = true;
        }
        _synchronize_cv.notify_one();
    });
    std::unique_lock lock{_synchronize_mutex};
    _synchronize_cv.wait(lock, [&finished] { return finished; });
}

void FallbackCommandQueue::enqueue(luisa::move_only_function<void()> &&task) noexcept {
    _wait_for_task_queue_available();
    _enqueue_task_no_wait(std::move(task));
}

void FallbackCommandQueue::enqueue_parallel(uint n, luisa::move_only_function<void(uint)> &&task) noexcept {
    static auto LUISA_SINGLE_THREADING = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_SINGLE_THREADING");
        return env != nullptr && env == "1"sv;
    }();
    enqueue([this, n, task = std::move(task)]() mutable noexcept {
        // for debugging
        if (LUISA_SINGLE_THREADING) {
            for (auto i = 0u; i < n; ++i) { task(i); }
            return;
        }
#if defined(LUISA_FALLBACK_USE_DISPATCH_QUEUE)
        if (_dispatch_queue == nullptr) {
#ifdef LUISA_PLATFORM_APPLE
            _dispatch_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
#else
            _dispatch_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
#endif
            dispatch_retain(_dispatch_queue);
        }
        dispatch_apply_f(n, _dispatch_queue, &task, [](void *context, size_t idx) noexcept {
            auto task = static_cast<luisa::move_only_function<void(uint)> *>(context);
            (*task)(static_cast<uint>(idx));
        });
#elif defined(LUISA_FALLBACK_USE_PPL)
        concurrency::parallel_for(0u, n, task);
#elif defined(LUISA_FALLBACK_USE_TBB)
        tbb::parallel_for(0u, n, task);
#elif defined(LUISA_FALLBACK_USE_AKARI_THREAD_POOL)
        if (_worker_pool == nullptr) {
            _worker_pool = luisa::make_unique<AkrThreadPool>(_worker_count);
        }
        _worker_pool->parallel_for(n, std::move(task));
#endif
    });
}

}// namespace luisa::compute::fallback
