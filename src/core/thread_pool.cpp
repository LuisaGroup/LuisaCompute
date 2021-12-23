//
// Created by Mike Smith on 2021/12/23.
//

#include <sstream>
#include <core/logging.h>
#include <core/thread_pool.h>

namespace luisa {

ThreadPool::ThreadPool(size_t num_threads) noexcept
    : _synchronize_barrier{[&num_threads] {
          if (num_threads == 0u) { num_threads = std::thread::hardware_concurrency(); }
          return static_cast<std::ptrdiff_t>(num_threads /* worker threads */ + 1u /* main thread */);
      }()},
      _dispatch_barrier{static_cast<std::ptrdiff_t>(num_threads)},
      _should_stop{false} {
    if (num_threads + 1u > barrier_type::max()) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Too many threads: {} (max = {}).",
            num_threads, barrier_type::max() - 1u);
    }
    _threads.reserve(num_threads);
    for (auto i = 0u; i < num_threads; i++) {
        _threads.emplace_back(std::thread{[this] {
            for (;;) {
                std::unique_lock lock{_mutex};
                _cv.wait(lock, [this] { return !_tasks.empty() || _should_stop; });
                if (_should_stop) [[unlikely]] { break; }
                auto task = std::move(_tasks.front());
                _tasks.pop();
                lock.unlock();
                task();
            }
        }});
    }
    LUISA_INFO(
        "Created thread pool with {} thread{}.",
        num_threads, num_threads == 1u ? "" : "s");
}

#define LUISA_THREAD_POOL_CHECK_NOT_IN_WORKER_THREAD(f) \
    auto thread_id = std::this_thread::get_id();        \
    for (auto &&t : _threads) {                         \
        if (t.get_id() == thread_id) [[unlikely]] {     \
            std::ostringstream oss;                     \
            oss << thread_id;                           \
            LUISA_ERROR_WITH_LOCATION(                  \
                "Invoking ThreadPool::" #f "() "        \
                "from worker thread {}.",               \
                oss.str());                             \
        }                                               \
    }

void ThreadPool::barrier() noexcept {
    LUISA_THREAD_POOL_CHECK_NOT_IN_WORKER_THREAD(barrier)
    _dispatch_all([this] { _dispatch_barrier.arrive_and_wait(); });
}

void ThreadPool::synchronize() noexcept {
    LUISA_THREAD_POOL_CHECK_NOT_IN_WORKER_THREAD(synchronize)
    _dispatch_all([this] { _synchronize_barrier.arrive_and_wait(); });
    _synchronize_barrier.arrive_and_wait();
}

#undef LUISA_THREAD_POOL_CHECK_NOT_IN_WORKER_THREAD

void ThreadPool::_dispatch(std::function<void()> task) noexcept {
    {
        std::scoped_lock lock{_mutex};
        _tasks.emplace(std::move(task));
    }
    _cv.notify_one();
}

void ThreadPool::_dispatch_all(std::function<void()> task) noexcept {
    {
        std::scoped_lock lock{_mutex};
        for (auto i = 0u; i < _threads.size() - 1u; i++) { _tasks.emplace(task); }
        _tasks.emplace(std::move(task));
    }
    _cv.notify_all();
}

ThreadPool::~ThreadPool() noexcept {
    {
        std::scoped_lock lock{_mutex};
        _should_stop = true;
    }
    _cv.notify_all();
    for (auto &&t : _threads) { t.join(); }
}

ThreadPool &ThreadPool::global() noexcept {
    static ThreadPool pool;
    return pool;
}

}// namespace luisa
