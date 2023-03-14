#include "thread_pool.h"
namespace vstd {
namespace tpool_detail {
static thread_local WorkerThread *workerInst = nullptr;
static thread_local ThreadBarrier *curBarrier = nullptr;
static thread_local Node *tlocalNode = nullptr;
NodeAlloc::NodeAlloc()
    : pool(64, true) {}
WorkerThread::WorkerThread(ThreadPool *tp)
    : thd([tp, this] { tp->thread_run(this); }) {}
#ifndef NDEBUG
static std::atomic_int64_t nodeCount = 0;
struct MemoryLeakDetact {
    ~MemoryLeakDetact() {
        assert(nodeCount == 0);
    }
};
static MemoryLeakDetact leakDetactor;
#endif

Node::~Node() {
#ifndef NDEBUG
    --nodeCount;
#endif
    for (auto &&i : afterWork) {
        i->deref();
    }
    if (parallelCount == 0)
        singleFunc.destroy();
    else
        parallelFunc.destroy();
}
Node::Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void()> &&func)
    : worker(worker), parallelCount(0), ref(ref), pool(pool), joinedSize(joinedSize), barrier(barrier) {
    barrier->addref();
#ifndef NDEBUG
    ++nodeCount;
#endif
    singleFunc.create(std::move(func));
}
Node::Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void(size_t)> &&func, size_t count, size_t queueCount)
    : worker(worker), parallelCount(count), parallelIdx(0), ref(ref), pool(pool), queueCount(queueCount), joinedSize(joinedSize), barrier(barrier) {
    barrier->addref();
#ifndef NDEBUG
    ++nodeCount;
#endif
    parallelFunc.create(std::move(func));
}

void Node::deref() {
    if (--ref == 0) {
        worker->pool.destroy_lock(worker->allocMtx, this);
    }
}

void Node::execute() {
    curBarrier = barrier;
    auto then = [&] {
        barrier->notify();
        mtx.lock();
        executing = false;
        mtx.unlock();
        size_t notifyCount = 0;
        for (auto i : afterWork) {
            if (--i->joinedSize == 0)
                notifyCount += EnqueueNode(i);
            i->deref();
        }
        afterWork.clear();
        if (notifyCount == 0) return;
        pool->mtx.lock();
        pool->mtx.unlock();
        if (workerInst) {
            notifyCount--;
        }
        pool->notify_worker(notifyCount);
    };
    if (parallelCount == 0) {
        tlocalNode = this;
        (*singleFunc)();
        tlocalNode = nullptr;
        then();
    } else {
        while (true) {
            auto i = parallelIdx++;
            if (i >= parallelCount)
                break;

            tlocalNode = this;
            (*parallelFunc)(i);
            tlocalNode = nullptr;
            if (++finishedIdx == parallelCount) {
                then();
            }
        }
    }
}

Event::Event(Node *node)
    : node(node) {
}

Event Event::then(function<void()> &&func) {
    auto newNode = node->pool->alloc_node(node->barrier, 1, 2, std::move(func));
    node->run_after(newNode);
    return {newNode};
}
Event Event::then(function<void(size_t)> &&func, size_t count) {
    count = std::max<size_t>(1, count);
    auto queueCount = std::min<size_t>(count, node->pool->threadCount);
    auto newNode = node->pool->alloc_node(node->barrier, 1, 2, std::move(func), count, queueCount);
    node->run_all_after(newNode);
    return {newNode};
}
Event Event::after_self(function<void()> &&func) {
    auto node = tlocalNode;
    auto newNode = node->pool->alloc_node(node->barrier, 1, 2, std::move(func));
    node->run_after(newNode);
    return {newNode};
}
Event Event::after_self(function<void(size_t)> &&func, size_t count) {
    auto node = tlocalNode;
    count = std::max<size_t>(1, count);
    auto queueCount = std::min<size_t>(count, node->pool->threadCount);
    auto newNode = node->pool->alloc_node(node->barrier, 1, 2, std::move(func), count, queueCount);
    node->run_all_after(newNode);
    return {newNode};
}
size_t Node::EnqueueNode(Node *i) {
    if (i->parallelCount == 0) {
        ++i->ref;
        if (!workerInst->tempNode) {
            workerInst->tempNode = i;
        } else {
            pool->globalQueue.push(i);
        }
        return 1;
    } else {
        i->ref += i->queueCount;
        if (!workerInst->tempNode) {
            workerInst->tempNode = i;
        } else {
            pool->globalQueue.push(i);
        }
        for (auto cc : range(1, i->queueCount)) {
            pool->globalQueue.push(i);
        }
        return i->queueCount;
    }
}
void Node::run_after(Node *node) {
    {
        std::lock_guard lck(mtx);
        if (executing) {
            afterWork.emplace_back(node);
            return;
        }
    }
    if (--node->joinedSize == 0) {
        pool->globalQueue.push(node);
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->notify_worker(1);
    } else {
        node->deref();
    }
}
void Node::run_all_after(Node *node) {
    {
        std::lock_guard lck(mtx);
        if (executing) {
            afterWork.emplace_back(node);
            return;
        }
    }
    if (--node->joinedSize == 0) {
        auto count = node->queueCount;
        node->ref += count - 1;
        for (auto cc : range(count)) {
            pool->globalQueue.push(node);
        }
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->notify_worker(count);
    } else {
        node->deref();
    }
}

}// namespace tpool_detail

ThreadEvent ThreadBarrier::execute(function<void()> &&func) {
    auto newNode = pool->alloc_node(this, 0, 2, std::move(func));
    pool->globalQueue.push(newNode);
    pool->mtx.lock();
    pool->mtx.unlock();
    pool->notify_worker(1);
    return {newNode};
}
ThreadEvent ThreadBarrier::execute(function<void()> &&func, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return execute(std::move(func));
    }
    auto newNode = pool->alloc_node(this, depend.size(), depend.size() + 1, std::move(func));
    for (auto &&i : depend) {
        i.node->run_after(newNode);
    }
    return {newNode};
}

ThreadEvent ThreadBarrier::execute(function<void(size_t)> &&func, size_t threadCount) {
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->alloc_node(this, 0, queueCount + 1, std::move(func), threadCount, queueCount);
    for (auto i : range(queueCount)) {
        pool->globalQueue.push(newNode);
    }
    pool->mtx.lock();
    pool->mtx.unlock();
    pool->notify_worker(queueCount);
    return {newNode};
}
ThreadEvent ThreadBarrier::execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return execute(std::move(func), threadCount);
    }
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->alloc_node(this, depend.size(), depend.size() + 1, std::move(func), threadCount, queueCount);
    for (auto &&i : depend) {
        i.node->run_all_after(newNode);
    }
    return {newNode};
}

template<typename... Args>
tpool_detail::Node *ThreadPool::alloc_node(Args &&...args) {
    tpool_detail::NodeAlloc *alloc;
    if (tpool_detail::workerInst) {
        alloc = &tpool_detail::workerInst->alloc;
    } else {
        alloc = &defaultNodeAlloc;
    }
    auto ptr = alloc->pool.create_lock(alloc->allocMtx, alloc, this, std::forward<Args>(args)...);
    return ptr;
}

void ThreadPool::notify_worker(size_t i) {
    if (i < threadCount) {
        for (auto v : range(i)) {
            cv.notify_one();
        }
    } else {
        cv.notify_all();
    }
}
ThreadPool::ThreadPool(size_t threadCount) {
    threadCount = std::max<size_t>(threadCount, 1);
    this->threadCount = threadCount;
    auto localThreads = reinterpret_cast<tpool_detail::WorkerThread *>(vengine_malloc(threadCount * sizeof(tpool_detail::WorkerThread)));
    for (auto &i : vstd::ptr_range(localThreads, threadCount)) {
        new (&i) tpool_detail::WorkerThread(this);
    }
	threads = std::launder(localThreads);
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock lck(mtx);
        enabled = false;
    }
    cv.notify_all();
    for (auto &i : vstd::ptr_range(threads, threadCount)) {
		i.thd.join();
	}
    vengine_free(threads);
}
void ThreadPool::thread_run(tpool_detail::WorkerThread *worker) {
    tpool_detail::workerInst = worker;
    auto disp = scope_exit([] { tpool_detail::workerInst = nullptr; });
    while (enabled) {
        thread_run();
    }
}
void ThreadPool::thread_run() {
    auto ExecuteTmp = [&] {
        while (tpool_detail::workerInst->tempNode) {
            auto tmp = tpool_detail::workerInst->tempNode;
            tpool_detail::workerInst->tempNode = nullptr;
            tmp->execute();
            tmp->deref();
        }
    };
    while (auto task = globalQueue.pop()) {
        (*task)->execute();
        (*task)->deref();
        ExecuteTmp();
    }
    std::unique_lock lck(mtx);
    while (globalQueue.length() == 0 && enabled) {
        cv.wait(lck);
    }
}

ThreadEvent ThreadPool::CurrentEvent() {
    return ThreadEvent{tpool_detail::tlocalNode};
}
void ThreadBarrier::notify() {
    if (--barrierCount == 0) {
        barrierMtx.lock();
        barrierMtx.unlock();
        barrierCv.notify_all();
    }
}
void ThreadBarrier::wait() {
    if (barrierCount.load(std::memory_order_relaxed) == 0) return;
    std::unique_lock lck(barrierMtx);
    while (barrierCount > 0) {
        barrierCv.wait(lck);
    }
}
void ThreadBarrier::addref() {
    barrierCount++;
}
ThreadBarrier::~ThreadBarrier() {
    wait();
}
DeferredThreadBarrier::DeferredThreadBarrier(ThreadPool *pool)
    : barrier(pool) {}
DeferredThreadBarrier::~DeferredThreadBarrier() {
    submit();
    wait();
}
ThreadEvent DeferredThreadBarrier::execute(function<void()> &&func) {
    auto pool = barrier.pool;
    auto newNode = pool->alloc_node(&barrier, 0, 2, std::move(func));
    nodes.push(newNode);
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::execute(function<void()> &&func, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return execute(std::move(func));
    }
    auto pool = barrier.pool;
    auto newNode = pool->alloc_node(&barrier, depend.size(), depend.size() + 1, std::move(func));
    for (auto &&i : depend) {
        i.node->run_after(newNode);
    }
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::execute(function<void(size_t)> &&func, size_t threadCount) {
    auto pool = barrier.pool;
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->alloc_node(&barrier, 0, queueCount + 1, std::move(func), threadCount, queueCount);
    nodes.push(newNode);
    /*	for (auto i : range(queueCount)) {
	}*/
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return execute(std::move(func), threadCount);
    }
    threadCount = std::max<size_t>(1, threadCount);
    auto pool = barrier.pool;
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->alloc_node(&barrier, depend.size(), depend.size() + 1, std::move(func), threadCount, queueCount);
    for (auto &&i : depend) {
        i.node->run_all_after(newNode);
    }
    return {newNode};
}
void DeferredThreadBarrier::submit() {
    size_t count = 0;
    auto pool = barrier.pool;
    while (auto vOpt = nodes.pop()) {
        auto v = *vOpt;
        if (v->parallelCount == 0) {
            ++count;
            pool->globalQueue.push(v);
        } else {
            count += v->queueCount;
            for (auto i : range(count)) {
                pool->globalQueue.push(v);
            }
        }
    }
    if (count > 0) {
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->notify_worker(count);
    }
}
void DeferredThreadBarrier::wait() {
    barrier.wait();
}

}// namespace vstd