#pragma once
#include <vstl/common.h>
#include <vstl/functional.h>
#include <vstl/ObjectPtr.h>
#include <vstl/ThreadPool.h>
namespace vstd {
template<typename Ret>
class ExecuteNode {
	template<typename T>
	friend class ExecuteNode;

private:
	struct Value {
		StackObject<Ret> data;
		spin_mutex mtx;
		function<void(Value*)> nextFunc;
		bool isComplete = false;
		~value() {
			data.destroy();
		}
	};
	ObjectPtr<Value> value;
	mutable optional<ThreadTaskHandle> handle;
	ThreadPool* tPool;
	template<typename Func, typename Arg>
	static void Execute(ObjectPtr<Value> const& value, Func&& func, Arg&& arg) {
		Value* ptr = value;
		ptr->data.create(func(std::forward<Arg>(arg)));
		{
			std::lock_guard lck(ptr->mtx);
			ptr->isComplete = true;
		}
		if (ptr->nextFunc) {
			ptr->nextFunc(ptr);
		}
	}
	ExecuteNode(
		ThreadPool* tPool)
		: value(MakeObjectPtr(new value())), tPool(tPool) {
	}
	ExecuteNode(
		ThreadPool* tPool,
		ThreadTaskHandle const& handle)
		: value(MakeObjectPtr(new value())), tPool(tPool), handle(handle) {
	}
	ExecuteNode(
		ThreadPool* tPool,
		ThreadTaskHandle& handle)
		: value(MakeObjectPtr(new value())), tPool(tPool), handle(handle) {
	}
	ExecuteNode(
		ThreadPool* tPool,
		ThreadTaskHandle&& handle)
		: value(MakeObjectPtr(new value())), tPool(tPool), handle(std::move(handle)) {
	}

public:
	template<typename Func>
	ExecuteNode(
		ThreadPool* tPool,
		Func&& func)
		: ExecuteNode(tPool) {
		handle.create(tPool->GetTask(
			[value = this->value,
			 func = std::forward<Func>(func)]() mutable {
				Value* ptr = value;
				ptr->data.create(func());
				{
					std::lock_guard lck(ptr->mtx);
					ptr->isComplete = true;
				}
				if (ptr->nextFunc) {
					ptr->nextFunc(ptr);
				}
			},
			true));
		handle->Execute();
	}
	Ret const& operator*() const& {
		if (handle) {
			handle->Complete();
			handle.destroy();
		}
		return *value->data;
	}
	Ret&& operator*() && {
		if (handle) {
			handle->Complete();
			handle.destroy();
		}
		return std::move(*value->data);
	}
	ExecuteNode(ExecuteNode const&) = default;
	ExecuteNode(ExecuteNode&&) = default;
	template<typename Func>
	decltype(auto) operator<<(Func&& func) const& {
		using RetType = FuncRetType<std::remove_cvref_t<Func>>;
		Value* ptr = value;
		std::lock_guard lck(ptr->mtx);
		if (ptr->isComplete) {
			return ExecuteNode<RetType>(
				tPool,
				[value = this->value,
				 func = std::forward<Func>(func)]() mutable {
					return func(std::move(*(value->data)));
				});
		} else {
			ExecuteNode<RetType> retExe(tPool, *handle);
			ptr->nextFunc =
				[func = std::forward<Func>(func),
				 retExeValue = retExe.value](auto ptr) mutable {
					ExecuteNode<RetType>::Execute(retExeValue, std::forward<Func>(func), *(ptr->data));
				};
			return retExe;
		}
	}
	template<typename Func>
	decltype(auto) operator<<(Func&& func) && {
		using RetType = FuncRetType<std::remove_cvref_t<Func>>;
		Value* ptr = value;
		std::lock_guard lck(ptr->mtx);
		if (ptr->isComplete) {
			return ExecuteNode<RetType>(
				tPool,
				[value = std::move(this->value),
				 func = std::forward<Func>(func)]() mutable {
					return func(std::move(*(value->data)));
				});
		} else {
			ExecuteNode<RetType> retExe(tPool, std::move(*handle));
			ptr->nextFunc =
				[func = std::forward<Func>(func),
				 retExeValue = retExe.value](auto ptr) mutable {
					ExecuteNode<RetType>::Execute(retExeValue, std::forward<Func>(func), std::move(*(ptr->data)));
				};
			return retExe;
		}
	}
};

class Executor {
	ThreadPool tPool;

public:
	Executor(size_t threadCount) : tPool(threadCount) {}
	template<typename Func>
	decltype(auto) operator<<(Func&& func) {
		using RetType = FuncRetType<std::remove_cvref_t<Func>>;
		return ExecuteNode<RetType>(&tPool, std::forward<Func>(func));
	}
};
}// namespace vstd