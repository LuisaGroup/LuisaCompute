#pragma once
#include <Common/TypeWiper.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <Common/DynamicDLL.h>
#include <vstl/Memory.h>
class   TaskThread
{
private:
	std::thread thd;
	std::mutex mtx;
	std::mutex mainThreadMtx;
	std::condition_variable cv;
	std::condition_variable mainThreadCV;
	std::atomic_bool enabled;
	std::atomic_bool runNext;
	std::atomic_bool mainThreadLocked;
	void(*funcData)(void*);
	void* funcBody;
	static void RunThread(TaskThread* ptr);
	template <typename T>
	inline static void Run(void* ptr)
	{
		(*(T*)ptr)();
	}
public:
	VSTL_OVERRIDE_OPERATOR_NEW
	TaskThread();
	void ExecuteNext();
	void Complete();
	bool IsCompleted() const { return !mainThreadLocked; }
	~TaskThread();
	template <typename T>
	void SetFunctor(T& func)
	{
		using Type = typename std::remove_const_t<T>;
		funcData = Run<Type>;
		funcBody = &((Type&)func);
	}
};
