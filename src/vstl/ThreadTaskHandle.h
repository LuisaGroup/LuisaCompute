#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
#include <EASTL/shared_ptr.h>
#include <span>
class ThreadPool;
class VENGINE_DLL_COMMON ThreadTaskHandle {
    friend class ThreadPool;

public:
    enum class TaskState : uint8_t {
        Waiting,
        Executed,
        Working,
        Finished
    };

private:
    struct VENGINE_DLL_COMMON TaskData {
        std::atomic_uint8_t state;
        vstd::function<void()> func;
        //ThreadPool Set
        /* std::mutex mtx;
		std::condition_variable cv;*/
        int64 refCount = 1;
        using Locker = std::pair<std::mutex, std::condition_variable>;
        vstd::StackObject<Locker> mainThreadLocker;
        vstd::spin_mutex lockerMtx;
        vstd::vector<eastl::shared_ptr<TaskData>> dependedJobs;
        vstd::vector<eastl::shared_ptr<TaskData>> dependingJob;
        std::atomic_size_t dependCount = 0;
        bool isWaitable;
        TaskData(bool waitable);
        TaskData(vstd::function<void()> &&func, bool waitable);
        ~TaskData();
        Locker *GetThreadLocker();
        void ReleaseThreadLocker();
    };

    bool isArray;
    union {
        vstd::StackObject<eastl::shared_ptr<TaskData>> taskFlag;
        vstd::StackObject<vstd::vector<eastl::shared_ptr<TaskData>>> taskFlags;
    };
    ThreadPool *pool;
    ThreadTaskHandle(
        ThreadPool *pool,
        bool waitable);
    ThreadTaskHandle(
        ThreadPool *pool,
        vstd::function<void()> &&func,
        bool waitable);
    ThreadTaskHandle(
        ThreadPool *pool,
        vstd::function<void(size_t)> &&func,
        size_t parallelCount,
        size_t threadCount,
        bool waitable);
    ThreadTaskHandle(
        ThreadPool *pool,
        vstd::function<void(size_t, size_t)> &&func,
        size_t parallelCount,
        size_t threadCount,
        bool waitable);
    template<typename H>
    void TAddDepend(H &&) const;

public:
    ~ThreadTaskHandle();
    ThreadTaskHandle(ThreadTaskHandle const &v);
    void AddDepend(ThreadTaskHandle const &handle) const;
    void AddDepend(std::span<ThreadTaskHandle const> handles) const;
    void AddDepend(std::initializer_list<ThreadTaskHandle const> handles) const {
        AddDepend(std::span<ThreadTaskHandle const>(handles.begin(), handles.end()));
    }
    ThreadTaskHandle(ThreadTaskHandle &&v);
    void operator=(ThreadTaskHandle const &v) {
        this->~ThreadTaskHandle();
        new (this) ThreadTaskHandle(v);
    }
    void operator=(ThreadTaskHandle &&v) {
        this->~ThreadTaskHandle();
        new (this) ThreadTaskHandle(std::move(v));
    }
    void Complete() const;
    bool IsComplete() const;
    void Execute() const;
};