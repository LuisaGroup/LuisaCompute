//
// Created by Mike Smith on 2021/11/20.
//

#include <mutex>

#include <core/mathematics.h>
#include <network/thread_pool.h>

namespace luisa::compute {

namespace detail {

class ThreadPoolImpl {
    
    enum struct Token {
        WORK_1D,
        WORK_2D,
        EXIT
    };

private:
    std::vector<std::thread> _workers;
    uint2 _dispatch_size{};
    uint2 _grid_size{};
    uint2 _block_size{};
    ThreadPool::Task1D _work_1d;
    ThreadPool::Task2D _work_2d;
    std::mutex _dispatch_mutex;
    std::mutex _finish_mutex;
    std::condition_variable _dispatch_cv;
    std::condition_variable _finish_cv;
    uint32_t _dispatch_count{0u};
    uint32_t _finish_count{0u};
    uint64_t _dispatch_fence{0ul};
    Token _token{};
    std::atomic_uint _curr_block{0u};

private:
    void _notify_finish() noexcept {
        {
            std::scoped_lock lock{_finish_mutex};
            _finish_count++;
        }
        _finish_cv.notify_one();
    }

    void _process_work_1d() noexcept {
        for (auto block = _curr_block++; block < _grid_size.x; block = _curr_block++) {
            for (auto i = 0u; i < _block_size.x; i++) {
                auto thread_id = block * _block_size.x + i;
                if (thread_id >= _dispatch_size.x) { break; }
                _work_1d(thread_id);
            }
        }
    }

    void _process_work_2d() noexcept {
        auto block_count = _grid_size.x * _grid_size.y;
        for (auto block = _curr_block++; block < block_count; block = _curr_block++) {
            auto block_x = block % _grid_size.x;
            auto block_y = block / _grid_size.x;
            for (auto y = 0u; y < _block_size.y; y++) {
                auto thread_y = block_y * _block_size.y + y;
                if (thread_y >= _dispatch_size.y) { break; }
                for (auto x = 0u; x < _block_size.x; x++) {
                    auto thread_x = block_x * _block_size.x + x;
                    if (thread_x >= _dispatch_size.x) { break; }
                    _work_2d(uint2(thread_x, thread_y));
                }
            }
        }
    }

    [[nodiscard]] auto _all_dispatched() noexcept {
        std::scoped_lock lock{_dispatch_mutex};
        return _dispatch_count == _workers.size();
    }

    void _synchronize() noexcept {
        std::unique_lock lock{_finish_mutex};
        _finish_cv.wait(lock, [this] { return _finish_count == _workers.size(); });
    }

    void _dispatch() noexcept {
        _dispatch_count = 0u;
        _finish_count = 0u;
        _dispatch_fence++;
        _curr_block = 0u;
        _grid_size = (_dispatch_size + _block_size - 1u) / _block_size;
        do {
            _dispatch_cv.notify_one();
        } while (!_all_dispatched());
    }

public:
    explicit ThreadPoolImpl(size_t num_workers) noexcept {
        _workers.reserve(num_workers);
        for (auto i = 0u; i < num_workers; i++) {
            _workers.emplace_back([this] {
                for (auto fence = 0ull;; fence++) {

                    // wait for work
                    std::unique_lock lock{_dispatch_mutex};
                    _dispatch_cv.wait(lock, [this, fence] { return _dispatch_fence > fence; });
                    _dispatch_count++;
                    lock.unlock();

                    // process work
                    switch (_token) {
                        case Token::WORK_1D:
                            _process_work_1d();
                            _notify_finish();
                            break;
                        case Token::WORK_2D:
                            _process_work_2d();
                            _notify_finish();
                            break;
                        case Token::EXIT:
                            return;
                    }
                }
            });
        }
    }

    ~ThreadPoolImpl() noexcept {
        _token = Token::EXIT;
        _dispatch();
        for (auto &&w : _workers) { w.join(); }
    }

    void dispatch(ThreadPool::Task1D task, uint32_t dispatch_size, uint32_t block_size) noexcept {
        _token = Token::WORK_1D;
        _work_1d = std::move(task);
        _dispatch_size = max(uint2{dispatch_size, 1u}, uint2{1u});
        _block_size = max(uint2{block_size, 1u}, uint2{1u});
        _dispatch();
        _synchronize();
        _work_1d = {};
    }

    void dispatch(ThreadPool::Task2D task, uint2 dispatch_size, uint2 block_size) noexcept {
        _token = Token::WORK_2D;
        _work_2d = std::move(task);
        _dispatch_size = max(dispatch_size, uint2{1u});
        _block_size = max(block_size, uint2{1u});
        _dispatch();
        _synchronize();
        _work_2d = {};
    }
};

}

ThreadPool::ThreadPool(size_t num_workers) noexcept
    : _thread_pool{std::make_unique<detail::ThreadPoolImpl>(
          num_workers == 0u
              ? std::thread::hardware_concurrency()
              : num_workers)} {}

ThreadPool::~ThreadPool() noexcept = default;

void ThreadPool::dispatch_1d(uint32_t dispatch_size, uint32_t block_size, Task1D task) noexcept {
    _thread_pool->dispatch(std::move(task), dispatch_size, block_size);
}

void ThreadPool::dispatch_2d(uint2 dispatch_size, uint2 block_size, Task2D task) noexcept {
    _thread_pool->dispatch(std::move(task), dispatch_size, block_size);
}

}
