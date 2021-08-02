#pragma once
#include <core/vstl/vector.h>
#include <Common/DynamicDLL.h>
class  IEnumerator
{
private:
	struct  Executor
	{
		alignas(size_t) char c[sizeof(size_t)];
		void(*constructPtr)(void*, void*) = nullptr;
		bool(*funcPtr)(void*) = nullptr;
		void(*disposeFunc)(void*) = nullptr;
		template <typename Func>
		Executor(const Func& f)
		{
			new (c)Func(f);
			funcPtr = [](void* ptr)->bool
			{
				Func* f = (Func*)ptr;
				return (*f)();
			};
			disposeFunc = [](void* ptr)->void
			{
				Func* f = (Func*)ptr;
				f->~Func();
			};
			constructPtr = [](void* ptr, void* arg)->void
			{
				Func* f = (Func*)arg;
				new(ptr)Func(*f);
			};

		}

		Executor(const Executor& exe);
		~Executor();
		bool operator()();
		void operator=(const Executor& exe);
	};
protected:
	vstd::vector<Executor> executors;
	uint32_t startIndex = 0;
public:
	virtual ~IEnumerator() {}
	void Reset() { startIndex = 0; }
	template <typename Func>
	void AddTask(const Func& func)
	{
		executors.emplace_back(func);
	}
	bool ExecuteOne();
};