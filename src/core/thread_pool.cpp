//
// Created by Mike Smith on 2021/12/23.
//

#include <sstream>
#include <core/logging.h>
#include <core/thread_pool.h>

namespace luisa {

namespace detail {

[[nodiscard]] static inline auto &is_worker_thread() noexcept {
    static thread_local auto is_worker = false;
    return is_worker;
}

static inline void check_not_in_worker_thread(std::string_view f) noexcept {
    if (is_worker_thread()) [[unlikely]] {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        LUISA_ERROR_WITH_LOCATION(
            "Invoking ThreadPool::{}() "
            "from worker thread {}.",
            f, oss.str());
    }
}

}// namespace detail

// reference: https://github.com/yohhoy/yamc/blob/master/include/yamc_barrier.hpp
class Barrier {

private:
    uint _n;
    uint _counter;
    uint _phase;
    std::condition_variable _cv;
    std::mutex _mutex;

public:
    explicit Barrier(uint n) noexcept
        : _n{n}, _counter{n}, _phase{0u} {}
    void arrive_and_wait() noexcept {
        std::unique_lock lock{_mutex};
       auto arrive_phase = _phase;
        if (--_counter == 0u) {
            _counter = _n;
            _phase++;
            _cv.notify_all();
        }
        while (_phase <= arrive_phase) {
            _cv.wait(lock);
        }
    }
};

ThreadPool::ThreadPool(size_t num_threads) noexcept : _should_stop{false} {
    if (num_threads == 0u) {
        num_threads = std::max(
            std::thread::hardware_concurrency(), 1u);
    }
    _dispatch_barrier = luisa::make_unique<Barrier>(num_threads);
    _synchronize_barrier = luisa::make_unique<Barrier>(num_threads + 1u /* main thread */);
    _threads.reserve(num_threads);
    for (auto i = 0u; i < num_threads; i++) {
        _threads.emplace_back(std::thread{[this] {
            detail::is_worker_thread() = true;
            for (;;) {
                std::unique_lock lock{_mutex};
                _cv.wait(lock, [this] { return !_tasks.empty() || _should_stop; });
                if (_should_stop && _tasks.empty()) [[unlikely]] { break; }
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

void ThreadPool::barrier() noexcept {
    detail::check_not_in_worker_thread("barrier");
    _dispatch_all([this] { _dispatch_barrier->arrive_and_wait(); });
}

void ThreadPool::synchronize() noexcept {
    detail::check_not_in_worker_thread("synchronize");
    _dispatch_all([this] { _synchronize_barrier->arrive_and_wait(); });
    _synchronize_barrier->arrive_and_wait();
}

void ThreadPool::_dispatch(luisa::function<void()> task) noexcept {
    {
        std::scoped_lock lock{_mutex};
        _tasks.emplace(std::move(task));
    }
    _cv.notify_one();
}

void ThreadPool::_dispatch_all(luisa::function<void()> task, size_t max_threads) noexcept {
    {
        std::scoped_lock lock{_mutex};
        for (auto i = 0u; i < std::min(_threads.size(), max_threads) - 1u; i++) {
            _tasks.emplace(task);
        }
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
