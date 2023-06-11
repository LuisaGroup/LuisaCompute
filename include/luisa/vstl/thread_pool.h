#pragma once
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/lockfree_array_queue.h>

namespace vstd {
class DeferredThreadBarrier;
class ThreadBarrier;
class ThreadPool;
namespace tpool_detail {
struct NodeAlloc;
struct LC_VSTL_API Node {
    std::atomic_size_t joinedSize;
    ThreadPool *pool;
    NodeAlloc *worker;
    ThreadBarrier *barrier;
    std::atomic_size_t ref;
    bool executing = true;
    union {
        StackObject<function<void()>> singleFunc;
        StackObject<function<void(size_t)>> parallelFunc;
    };
    spin_mutex mtx;
    fixed_vector<Node *, 4> afterWork;
    std::atomic_size_t parallelIdx;
    std::atomic_size_t finishedIdx = 0;
    size_t parallelCount;
    size_t queueCount;
    Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void()> &&func);
    Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void(size_t)> &&func, size_t count, size_t queueCount);
    ~Node();
    void execute();
    void deref();
    void run_after(Node *node);
    void run_all_after(Node *node);
    size_t EnqueueNode(Node *node);
};
struct LC_VSTL_API NodeAlloc {
    spin_mutex allocMtx;
    Pool<Node, true> pool;
    NodeAlloc();
};
struct LC_VSTL_API Event : public IOperatorNewBase {
    friend class ::vstd::ThreadPool;
    friend class ::vstd::ThreadBarrier;
    friend class ::vstd::DeferredThreadBarrier;
    bool Valid() const { return node; }
    ~Event() {
        if (node) node->deref();
    }
    Event(Event const &v) {
        node = v.node;
        if (node) ++node->ref;
    }
    Event(Event &&v) {
        node = v.node;
        v.node = nullptr;
    }
    Event &operator=(Event const &v) {
        if (&v == this) return *this;
        if (node) node->deref();
        node = v.node;
        if (node) ++node->ref;
        return *this;
    }
    Event &operator=(Event &&v) {
        if (&v == this) return *this;
        if (node) node->deref();
        node = v.node;
        v.node = nullptr;
        return *this;
    }
    Event() : node(nullptr) {}
    Event then(function<void()> &&func);
    Event then(function<void(size_t)> &&func, size_t count);
    static Event after_self(function<void()> &&func);
    static Event after_self(function<void(size_t)> &&func, size_t count);

private:
    Node *node;
    Event(Node *node);
};
struct LC_VSTL_API WorkerThread {
    std::thread thd;
    NodeAlloc alloc;
    Node *tempNode{nullptr};
    WorkerThread(ThreadPool *tp);
};
}// namespace tpool_detail
using ThreadEvent = tpool_detail::Event;
class LC_VSTL_API ThreadPool : public IOperatorNewBase {
    friend class ThreadBarrier;
    friend class DeferredThreadBarrier;
    friend class tpool_detail::WorkerThread;
    friend class tpool_detail::Event;
    friend class tpool_detail::Node;

public:
    using Node = tpool_detail::Node;
    static ThreadEvent CurrentEvent();

private:
    tpool_detail::WorkerThread *threads;
    size_t threadCount;
    tpool_detail::NodeAlloc defaultNodeAlloc;
    LockFreeArrayQueue<Node *> globalQueue;
    std::mutex mtx;
    std::condition_variable cv;

    bool enabled = true;
    void thread_run(tpool_detail::WorkerThread *worker);
    void thread_run();
    template<typename... Args>
    Node *alloc_node(Args &&...args);
    void notify_worker(size_t i);

public:
    ThreadPool(size_t threadCount = std::thread::hardware_concurrency());
    ThreadPool(ThreadPool const &) = delete;
    ThreadPool(ThreadPool &&) = delete;
    ~ThreadPool();
};
class LC_VSTL_API ThreadBarrier : public IOperatorNewBase {
    friend class ThreadPool;
    friend class DeferredThreadBarrier;
    friend class tpool_detail::Node;
    std::atomic_size_t barrierCount = 0;
    std::mutex barrierMtx;
    std::condition_variable barrierCv;
    void addref();
    void notify();
    using Node = tpool_detail::Node;

public:
    ThreadPool *pool;
    void wait();
    ThreadEvent execute(function<void()> &&func);
    ThreadEvent execute(function<void()> &&func, span<ThreadEvent const> depend);
    ThreadEvent execute(function<void()> &&func, std::initializer_list<ThreadEvent> depend) {
        return execute(std::move(func), span<ThreadEvent const>(depend.begin(), depend.end()));
    }
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount);
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend);
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount, std::initializer_list<ThreadEvent> depend) {
        return execute(std::move(func), threadCount, span<ThreadEvent const>(depend.begin(), depend.end()));
    }
    ThreadBarrier(ThreadPool *pool)
        : pool(pool) {}
    ThreadBarrier() : ThreadBarrier(nullptr) {}
    ThreadBarrier(ThreadPool &pool)
        : ThreadBarrier(&pool) {}
    ThreadBarrier(ThreadBarrier &&) = delete;
    ThreadBarrier(ThreadBarrier const &) = delete;
    ~ThreadBarrier();
};
class LC_VSTL_API DeferredThreadBarrier : public IOperatorNewBase {
    using Node = tpool_detail::Node;
    LockFreeArrayQueue<Node *> nodes;

public:
    ThreadBarrier barrier;

    DeferredThreadBarrier(ThreadPool *pool);
    DeferredThreadBarrier() : DeferredThreadBarrier(nullptr) {}
    ~DeferredThreadBarrier();
    DeferredThreadBarrier(DeferredThreadBarrier &&) = delete;
    DeferredThreadBarrier(DeferredThreadBarrier const &) = delete;
    DeferredThreadBarrier(ThreadPool &pool) : DeferredThreadBarrier(&pool) {}
    ThreadEvent execute(function<void()> &&func);
    ThreadEvent execute(function<void()> &&func, span<ThreadEvent const> depend);
    ThreadEvent execute(function<void()> &&func, std::initializer_list<ThreadEvent> depend) {
        return execute(std::move(func), span<ThreadEvent const>(depend.begin(), depend.end()));
    }
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount);
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend);
    ThreadEvent execute(function<void(size_t)> &&func, size_t threadCount, std::initializer_list<ThreadEvent> depend) {
        return execute(std::move(func), threadCount, span<ThreadEvent const>(depend.begin(), depend.end()));
    }
    void submit();
    void wait();
};
}// namespace vstd
