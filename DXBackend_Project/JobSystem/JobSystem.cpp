#include "JobSystem.h"
#include <thread>
#include <condition_variable>
#include <atomic>
#include "JobBucket.h"
#include "JobNode.h"
#include "../Common/Memory.h"
void JobSystem::UpdateNewBucket() {
START:
	if (currentBucketPos >= buckets.size()) {
		{
			lockGuard lck(mainThreadWaitMutex);
			mainThreadFinished = true;
			mainThreadWaitCV.notify_all();
		}
		return;
	}
	JobBucket* bucket = buckets[currentBucketPos];
	if (bucket->jobNodesVec.empty() || bucket->sys != this) {
		currentBucketPos++;
		goto START;
	}
	bucketMissionCount = bucket->allJobNodes.size();
	for (auto node : bucket->jobNodesVec) {
		if (node->targetDepending == 0) {
			executingNode.Push(node);
		}
	}
	bucket->jobNodesVec.clear();
	bucket->allJobNodes.clear();
	currentBucketPos++;
	uint32_t size = executingNode.Length();
	if (executingNode.Length() < mThreadCount) {
		lockGuard lck(threadMtx);
		for (int64_t i = 0; i < executingNode.Length(); ++i) {
			cv.notify_one();
		}
	} else {
		lockGuard lck(threadMtx);
		cv.notify_all();
	}
}
class JobThreadRunnable {
public:
	JobSystem* sys;
	/*bool* JobSystemInitialized;
	std::condition_variable* cv;
	ConcurrentQueue<JobNode*>* executingNode;
	std::atomic<int32_t>* bucketMissionCount;*/
	void operator()() {
		{
			std::unique_lock<std::mutex> lck(sys->threadMtx);
			while (!sys->jobSystemStart && sys->JobSystemInitialized) {
				sys->cv.wait(lck);
			}
		}
		int32_t value = (int32_t)-1;
	MAIN_THREAD_LOOP : {
		JobNode* node = nullptr;
		while (sys->executingNode.Pop(&node)) {
		START_LOOP:
			JobNode* nextNode = node->Execute(sys->executingNode, sys->cv);
			sys->jobNodePool.Delete(node);
			value = --sys->bucketMissionCount;
			if (nextNode != nullptr) {
				node = nextNode;
				goto START_LOOP;
			}
			if (value == 0) {
				sys->UpdateNewBucket();
			}
		}
		std::unique_lock<std::mutex> lck(sys->threadMtx);
		while (sys->JobSystemInitialized) {
			sys->cv.wait(lck);
			goto MAIN_THREAD_LOOP;
		}
	}
	}
};
JobBucket* JobSystem::GetJobBucket() {
	if (releasedBuckets.empty()) {
		JobBucket* bucket = new JobBucket(this);
		usedBuckets.push_back(bucket);
		return bucket;
	} else {
		auto ite = releasedBuckets.end() - 1;
		JobBucket* cur = *ite;
		cur->jobNodesVec.clear();
		releasedBuckets.erase(ite);
		return cur;
	}
}
void JobSystem::ReleaseJobBucket(JobBucket* node) {
	node->jobNodesVec.clear();
	releasedBuckets.push_back(node);
}
JobSystem::JobSystem(uint32_t threadCount) noexcept
	: executingNode(100),
	  mainThreadFinished(true),
	  jobNodePool(50) {
	/*allocatedMemory[0].reserve(50);
	allocatedMemory[1].reserve(50);*/
	mThreadCount = threadCount;
	usedBuckets.reserve(20);
	releasedBuckets.reserve(20);
	allThreads.resize(threadCount);
	for (uint32_t i = 0; i < threadCount; ++i) {
		JobThreadRunnable j;
		j.sys = this;
		allThreads[i] = new std::thread(j);
	}
}
JobSystem::~JobSystem() noexcept {
	JobSystemInitialized = false;
	{
		lockGuard lck(threadMtx);
		cv.notify_all();
	}
	for (uint32_t i = 0; i < allThreads.size(); ++i) {
		allThreads[i]->join();
		delete allThreads[i];
	}
	for (auto ite = usedBuckets.begin(); ite != usedBuckets.end(); ++ite) {
		delete *ite;
	}
}
/*
void* JobSystem::AllocFuncMemory(uint64_t size)
{
	void* ptr = vengine_malloc(size);
	allocatedMemory[allocatorSwitcher].push_back(ptr);
	return ptr;
}
void JobSystem::FreeAllMemory()
{
	for (uint i = 0; i < allocatedMemory[allocatorSwitcher].size(); ++i)
	{
		vengine_free(allocatedMemory[allocatorSwitcher][i]);
	}
	allocatedMemory[allocatorSwitcher].clear();
}
*/
void JobSystem::ExecuteBucket(JobBucket** bucket, uint32_t bucketCount) {
	jobNodePool.UpdateSwitcher();
	currentBucketPos = 0;
	buckets.resize(bucketCount);
	memcpy(buckets.data(), bucket, sizeof(JobBucket*) * bucketCount);
	mainThreadFinished = false;
	jobSystemStart = true;
	UpdateNewBucket();
	//FreeAllMemory();
}
void JobSystem::ExecuteBucket(JobBucket* bucket, uint32_t bucketCount) {
	jobNodePool.UpdateSwitcher();
	currentBucketPos = 0;
	buckets.resize(bucketCount);
	for (uint32_t i = 0; i < bucketCount; ++i) {
		buckets[i] = bucket + i;
	}
	mainThreadFinished = false;
	jobSystemStart = true;
	UpdateNewBucket();
	//FreeAllMemory();
}
void JobSystem::Wait() {
	std::unique_lock<std::mutex> lck(mainThreadWaitMutex);
	while (!mainThreadFinished) {
		mainThreadWaitCV.wait(lck);
	}
}
