#include "thread_pool.h"
namespace vstd {
namespace tpool_detail {
static thread_local WorkerThread *workerInst = nullptr;
static thread_local ThreadBarrier *curBarrier = nullptr;
static thread_local Node *tlocalNode = nullptr;
NodeAlloc::NodeAlloc()
    : pool(64, true) {}
WorkerThread::WorkerThread(ThreadPool *tp)
    : thd([tp, this] { tp->ThreadRun(this); }) {}
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
        i->Deref();
    }
    if (parallelCount == 0)
        singleFunc.destroy();
    else
        parallelFunc.destroy();
}
Node::Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void()> &&func)
    : worker(worker), parallelCount(0), ref(ref), pool(pool), joinedSize(joinedSize), barrier(barrier) {
    barrier->AddRef();
#ifndef NDEBUG
    ++nodeCount;
#endif
    singleFunc.create(std::move(func));
}
Node::Node(NodeAlloc *worker, ThreadPool *pool, ThreadBarrier *barrier, size_t joinedSize, size_t ref, function<void(size_t)> &&func, size_t count, size_t queueCount)
    : worker(worker), parallelCount(count), parallelIdx(0), ref(ref), pool(pool), queueCount(queueCount), joinedSize(joinedSize), barrier(barrier) {
    barrier->AddRef();
#ifndef NDEBUG
    ++nodeCount;
#endif
    parallelFunc.create(std::move(func));
}

void Node::Deref() {
    if (--ref == 0) {
        worker->pool.Delete_Lock(worker->allocMtx, this);
    }
}

void Node::Execute() {
    curBarrier = barrier;
    auto Then = [&] {
        barrier->Notify();
        mtx.lock();
        executing = false;
        mtx.unlock();
        size_t notifyCount = 0;
        for (auto i : afterWork) {
            if (--i->joinedSize == 0)
                notifyCount += EnqueueNode(i);
            i->Deref();
        }
        afterWork.clear();
        if (notifyCount == 0) return;
        pool->mtx.lock();
        pool->mtx.unlock();
        if (workerInst) {
            notifyCount--;
        }
        pool->NotifyWorker(notifyCount);
    };
    if (parallelCount == 0) {
        tlocalNode = this;
        (*singleFunc)();
        tlocalNode = nullptr;
        Then();
    } else {
        while (true) {
            auto i = parallelIdx++;
            if (i >= parallelCount)
                break;

            tlocalNode = this;
            (*parallelFunc)(i);
            tlocalNode = nullptr;
            if (++finishedIdx == parallelCount) {
                Then();
            }
        }
    }
}

Event::Event(Node *node)
    : node(node) {
}

Event Event::Then(function<void()> &&func) {
    auto newNode = node->pool->AllocNode(node->barrier, 1, 2, std::move(func));
    node->RunAfter(newNode);
    return {newNode};
}
Event Event::Then(function<void(size_t)> &&func, size_t count) {
    count = std::max<size_t>(1, count);
    auto queueCount = std::min<size_t>(count, node->pool->threadCount);
    auto newNode = node->pool->AllocNode(node->barrier, 1, 2, std::move(func), count, queueCount);
    node->RunAllAfter(newNode);
    return {newNode};
}
Event Event::AfterSelf(function<void()> &&func) {
    auto node = tlocalNode;
    auto newNode = node->pool->AllocNode(node->barrier, 1, 2, std::move(func));
    node->RunAfter(newNode);
    return {newNode};
}
Event Event::AfterSelf(function<void(size_t)> &&func, size_t count) {
    auto node = tlocalNode;
    count = std::max<size_t>(1, count);
    auto queueCount = std::min<size_t>(count, node->pool->threadCount);
    auto newNode = node->pool->AllocNode(node->barrier, 1, 2, std::move(func), count, queueCount);
    node->RunAllAfter(newNode);
    return {newNode};
}
size_t Node::EnqueueNode(Node *i) {
    if (i->parallelCount == 0) {
        ++i->ref;
        if (!workerInst->tempNode) {
            workerInst->tempNode = i;
        } else {
            pool->globalQueue.Push(i);
        }
        return 1;
    } else {
        i->ref += i->queueCount;
        if (!workerInst->tempNode) {
            workerInst->tempNode = i;
        } else {
            pool->globalQueue.Push(i);
        }
        for (auto cc : range(1, i->queueCount)) {
            pool->globalQueue.Push(i);
        }
        return i->queueCount;
    }
}
void Node::RunAfter(Node *node) {
    {
        std::lock_guard lck(mtx);
        if (executing) {
            afterWork.emplace_back(node);
            return;
        }
    }
    if (--node->joinedSize == 0) {
        pool->globalQueue.Push(node);
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->NotifyWorker(1);
    } else {
        node->Deref();
    }
}
void Node::RunAllAfter(Node *node) {
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
            pool->globalQueue.Push(node);
        }
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->NotifyWorker(count);
    } else {
        node->Deref();
    }
}

}// namespace tpool_detail

ThreadEvent ThreadBarrier::Execute(function<void()> &&func) {
    auto newNode = pool->AllocNode(this, 0, 2, std::move(func));
    pool->globalQueue.Push(newNode);
    pool->mtx.lock();
    pool->mtx.unlock();
    pool->NotifyWorker(1);
    return {newNode};
}
ThreadEvent ThreadBarrier::Execute(function<void()> &&func, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return Execute(std::move(func));
    }
    auto newNode = pool->AllocNode(this, depend.size(), depend.size() + 1, std::move(func));
    for (auto &&i : depend) {
        i.node->RunAfter(newNode);
    }
    return {newNode};
}

ThreadEvent ThreadBarrier::Execute(function<void(size_t)> &&func, size_t threadCount) {
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->AllocNode(this, 0, queueCount + 1, std::move(func), threadCount, queueCount);
    for (auto i : range(queueCount)) {
        pool->globalQueue.Push(newNode);
    }
    pool->mtx.lock();
    pool->mtx.unlock();
    pool->NotifyWorker(queueCount);
    return {newNode};
}
ThreadEvent ThreadBarrier::Execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return Execute(std::move(func), threadCount);
    }
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->AllocNode(this, depend.size(), depend.size() + 1, std::move(func), threadCount, queueCount);
    for (auto &&i : depend) {
        i.node->RunAllAfter(newNode);
    }
    return {newNode};
}

template<typename... Args>
tpool_detail::Node *ThreadPool::AllocNode(Args &&...args) {
    tpool_detail::NodeAlloc *alloc;
    if (tpool_detail::workerInst) {
        alloc = &tpool_detail::workerInst->alloc;
    } else {
        alloc = &defaultNodeAlloc;
    }
    auto ptr = alloc->pool.New_Lock(alloc->allocMtx, alloc, this, std::forward<Args>(args)...);
    return ptr;
}

void ThreadPool::NotifyWorker(size_t i) {
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
void ThreadPool::ThreadRun(tpool_detail::WorkerThread *worker) {
    tpool_detail::workerInst = worker;
    auto disp = scope_exit([] { tpool_detail::workerInst = nullptr; });
    while (enabled) {
        ThreadRun();
    }
}
void ThreadPool::ThreadRun() {
    auto ExecuteTmp = [&] {
        while (tpool_detail::workerInst->tempNode) {
            auto tmp = tpool_detail::workerInst->tempNode;
            tpool_detail::workerInst->tempNode = nullptr;
            tmp->Execute();
            tmp->Deref();
        }
    };
    while (auto task = globalQueue.Pop()) {
        (*task)->Execute();
        (*task)->Deref();
        ExecuteTmp();
    }
    std::unique_lock lck(mtx);
    while (globalQueue.Length() == 0 && enabled) {
        cv.wait(lck);
    }
}

ThreadEvent ThreadPool::CurrentEvent() {
    return ThreadEvent{tpool_detail::tlocalNode};
}
void ThreadBarrier::Notify() {
    if (--barrierCount == 0) {
        barrierMtx.lock();
        barrierMtx.unlock();
        barrierCv.notify_all();
    }
}
void ThreadBarrier::Wait() {
    if (barrierCount.load(std::memory_order_relaxed) == 0) return;
    std::unique_lock lck(barrierMtx);
    while (barrierCount > 0) {
        barrierCv.wait(lck);
    }
}
void ThreadBarrier::AddRef() {
    barrierCount++;
}
ThreadBarrier::~ThreadBarrier() {
    Wait();
}
DeferredThreadBarrier::DeferredThreadBarrier(ThreadPool *pool)
    : barrier(pool) {}
DeferredThreadBarrier::~DeferredThreadBarrier() {
    Submit();
    Wait();
}
ThreadEvent DeferredThreadBarrier::Execute(function<void()> &&func) {
    auto pool = barrier.pool;
    auto newNode = pool->AllocNode(&barrier, 0, 2, std::move(func));
    nodes.Push(newNode);
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::Execute(function<void()> &&func, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return Execute(std::move(func));
    }
    auto pool = barrier.pool;
    auto newNode = pool->AllocNode(&barrier, depend.size(), depend.size() + 1, std::move(func));
    for (auto &&i : depend) {
        i.node->RunAfter(newNode);
    }
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::Execute(function<void(size_t)> &&func, size_t threadCount) {
    auto pool = barrier.pool;
    threadCount = std::max<size_t>(1, threadCount);
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->AllocNode(&barrier, 0, queueCount + 1, std::move(func), threadCount, queueCount);
    nodes.Push(newNode);
    /*	for (auto i : range(queueCount)) {
	}*/
    return {newNode};
}
ThreadEvent DeferredThreadBarrier::Execute(function<void(size_t)> &&func, size_t threadCount, span<ThreadEvent const> depend) {
    if (depend.empty()) {
        return Execute(std::move(func), threadCount);
    }
    threadCount = std::max<size_t>(1, threadCount);
    auto pool = barrier.pool;
    auto queueCount = std::min<size_t>(threadCount, pool->threadCount);
    auto newNode = pool->AllocNode(&barrier, depend.size(), depend.size() + 1, std::move(func), threadCount, queueCount);
    for (auto &&i : depend) {
        i.node->RunAllAfter(newNode);
    }
    return {newNode};
}
void DeferredThreadBarrier::Submit() {
    size_t count = 0;
    auto pool = barrier.pool;
    while (auto vOpt = nodes.Pop()) {
        auto v = *vOpt;
        if (v->parallelCount == 0) {
            ++count;
            pool->globalQueue.Push(v);
        } else {
            count += v->queueCount;
            for (auto i : range(count)) {
                pool->globalQueue.Push(v);
            }
        }
    }
    if (count > 0) {
        pool->mtx.lock();
        pool->mtx.unlock();
        pool->NotifyWorker(count);
    }
}
void DeferredThreadBarrier::Wait() {
    barrier.Wait();
}

}// namespace vstd