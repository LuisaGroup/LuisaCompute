#pragma once
#include <luisa/core/fiber.h>
#include <luisa/vstl/common.h>
namespace lc::dx {
struct STDCondVar {
    std::condition_variable cv;
    std::mutex mtx;
};

struct FiberCondVar {
    luisa::fiber::condition_variable cv;
    luisa::fiber::mutex mtx;
};

struct CondVar {
    union {
        vstd::Storage<STDCondVar> std_condvar;
        vstd::Storage<FiberCondVar> fiber_condvar;
    };
    bool use_std;
    auto std_ptr() { return reinterpret_cast<STDCondVar *>(std_condvar.c); }
    auto fiber_ptr() { return reinterpret_cast<FiberCondVar *>(fiber_condvar.c); }
    CondVar(bool use_std)
        : use_std(use_std) {
        if (use_std) {
            new (std::launder(std_ptr())) STDCondVar();
        } else {
            new (std::launder(fiber_ptr())) FiberCondVar();
        }
    }
    template<typename Predicate>
        requires std::is_invocable_r_v<bool, Predicate>
    void wait(Predicate &&pred) {
        if (use_std) {
            auto ptr = std_ptr();
            std::unique_lock lck{ptr->mtx};
            ptr->cv.wait(lck, pred);
        } else {
            auto ptr = fiber_ptr();
            luisa::fiber::lock lck(ptr->mtx);
            ptr->cv.wait(lck, pred);
        }
    }
    void notify_one() {
        if (use_std) {
            auto ptr = std_ptr();
            ptr->cv.notify_one();
        } else {
            auto ptr = fiber_ptr();
            ptr->cv.notify_one();
        }
    }
    void notify_all() {
        if (use_std) {
            auto ptr = std_ptr();
            ptr->cv.notify_all();
        } else {
            auto ptr = fiber_ptr();
            ptr->cv.notify_all();
        }
    }
    void lock() {
        if (use_std) {
            std_ptr()->mtx.lock();
        } else {
            fiber_ptr()->mtx.lock();
        }
    }
    void try_lock() {
        if (use_std) {
            std_ptr()->mtx.try_lock();
        } else {
            fiber_ptr()->mtx.try_lock();
        }
    }
    void unlock() {
        if (use_std) {
            std_ptr()->mtx.unlock();
        } else {
            fiber_ptr()->mtx.unlock();
        }
    }
    ~CondVar() {
        if (use_std) {
            vstd::destruct(std_ptr());
        } else {
            vstd::destruct(fiber_ptr());
        }
    }
};

struct Thread {
    union {
        vstd::Storage<std::thread> std_thread;
        vstd::Storage<luisa::fiber::event> fiber_evt;
    };
    auto std_ptr() { return reinterpret_cast<std::thread *>(std_thread.c); }
    auto fiber_ptr() { return reinterpret_cast<luisa::fiber::event *>(fiber_evt.c); }
    bool use_std;
    template<typename Func>
        requires std::is_invocable_v<Func>
    Thread(Func &&func, bool use_std) : use_std(use_std) {
        if (use_std) {
            new (std::launder(std_ptr())) std::thread(std::forward<Func>(func));
        } else {
            new (std::launder(fiber_ptr())) luisa::fiber::event();
            luisa::fiber::schedule(
                [func = std::forward<Func>(func),
                 evt = *fiber_ptr()]() mutable {
                    func();
                    evt.signal();
                });
        }
    }
    void join() {
        if (use_std) {
            std_ptr()->join();
        } else {
            fiber_ptr()->wait();
        }
    }
    ~Thread() {
        if (use_std) {
            vstd::destruct(std_ptr());
        } else {
            vstd::destruct(fiber_ptr());
        }
    }
};
}// namespace lc::dx