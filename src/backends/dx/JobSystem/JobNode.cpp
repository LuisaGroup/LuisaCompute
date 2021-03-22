#include "JobNode.h"
#include "JobSystem.h"
#include "JobBucket.h"
JobNode::~JobNode()
{
	Dispose();
	if (dependedEventInitialized)
		dependingEvent.Delete();
}
JobNode* JobNode::Execute(LockFreeArrayQueue<JobNode*>& taskList, std::condition_variable& cv) {
	switch (runnableState)
	{
	case RunnableType::SingleTask:
		(*runnable)();
		break;
	case RunnableType::Parallel:
	{
		StackObject<Runnable<void(uint)>>& parameterRunnable = reinterpret_cast<StackObject<Runnable<void(uint)>>&>(runnable);
		for(uint i = parallelStart; i < parallelEnd; ++i)
			(*parameterRunnable)(i);
	}
	break;
	}
	auto ite = dependingEvent->begin();
	JobNode* nextNode = nullptr;
	while (ite != dependingEvent->end())
	{
		JobNode* node = *ite;
		uint32_t dependingCount = --node->targetDepending;
		if (dependingCount == 0)
		{
			nextNode = node;
			++ite;
			break;
		}
		++ite;
	}
	for (; ite != dependingEvent->end(); ++ite)
	{
		JobNode* node = *ite;
		uint32_t dependingCount = --node->targetDepending;
		if (dependingCount == 0)
		{
			taskList.Push(node);
			{
				lockGuard lck(*threadMtx);
				cv.notify_one();
			}
		}
	}
	return nextNode;
}
/*
void JobNode::Precede(JobNode* depending)
{
	depending->targetDepending++;
	dependingEvent->emplace_back(depending);
	if (depending->vecIndex >= 0)
	{
		JobBucket* bucket = depending->bucket;
		auto lastIte = bucket->jobNodesVec.end() - 1;
		auto indIte = bucket->jobNodesVec.begin() + depending->vecIndex;
		*indIte = *lastIte;
		(*indIte)->vecIndex = depending->vecIndex;
		bucket->jobNodesVec.erase(lastIte);
		depending->vecIndex = -1;
	}
}
*/
void JobNode::Create(JobBucket* bucket, Runnable<void()>&& runnable, JobSystem* sys, JobHandle const* dependedJobs, uint dependCount){
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	for (uint i = 0; i < dependCount; ++i) {
		targetDepending += dependedJobs->Count();
	}
	this->threadMtx = threadMtx;
	this->runnable.New(std::move(runnable));
	runnableState = RunnableType::SingleTask;
	for (uint i = 0; i < dependCount; ++i)
	{
		auto&& handle = dependedJobs[i];
		for (uint s = handle.start; s <= handle.end; ++s)
		{
			bucket->allJobNodes[s]->dependingEvent->push_back(this);
		}
	}
}
void JobNode::CreateParallel(JobBucket* bucket, Runnable<void()>&& runnable, uint parallelStart, uint parallelEnd, JobSystem* sys, JobHandle const* dependedJobs, uint dependCount) {
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	for (uint i = 0; i < dependCount; ++i) {
		targetDepending += dependedJobs->Count();
	}
	this->threadMtx = threadMtx;
	this->runnable.New(std::move(runnable));
	runnableState = RunnableType::Parallel;
	this->parallelStart = parallelStart;
	this->parallelEnd = parallelEnd;
	for (uint i = 0; i < dependCount; ++i)
	{
		auto&& handle = dependedJobs[i];
		for (uint s = handle.start; s <= handle.end; ++s)
		{
			bucket->allJobNodes[s]->dependingEvent->push_back(this);
		}
	}
}
void JobNode::CreateEmpty(JobBucket* bucket, JobSystem* sys, JobHandle const* dependedJobs, uint dependCount)
{
	std::mutex* threadMtx = &sys->threadMtx;
	targetDepending = 0;
	for (uint i = 0; i < dependCount; ++i) {
		targetDepending += dependedJobs->Count();
	}
	this->threadMtx = threadMtx;
	runnableState = RunnableType::UnAvaliable;
	for (uint i = 0; i < dependCount; ++i)
	{
		auto&& handle = dependedJobs[i];
		for (uint s = handle.start; s <= handle.end; ++s)
		{
			bucket->allJobNodes[s]->dependingEvent->push_back(this);
		}
	}
}
void JobNode::Reset()
{
	if (!dependedEventInitialized)
	{
		dependedEventInitialized = true;
		dependingEvent.New();
		dependingEvent->reserve(8);
	}
}
void JobNode::Dispose()
{
	switch (runnableState)
	{
	case RunnableType::SingleTask:
		runnable.Delete();
		break;
	case RunnableType::Parallel:
	{
		StackObject<Runnable<void(uint)>>& parameterRunnable = reinterpret_cast<StackObject<Runnable<void(uint)>>&>(runnable);
		parameterRunnable.Delete();
	}
	break;
	}
	runnableState = RunnableType::UnAvaliable;
	if (dependedEventInitialized)
	{
		dependingEvent->clear();
	}
}
