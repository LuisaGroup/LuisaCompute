#pragma once
#include <Common/Runnable.h>
#include <Common/Common.h>
#include <JobSystem/JobInclude.h>

template<typename T>
class ResourceName;
class GraphNodeData;
template<typename... Types>
class TupleCall;

class RuntimeType {
	friend class GraphNodeData;
	template<typename T>
	friend class ResourceName;
	template<typename... Types>
	friend class TupleCall;

	Type tp;
	/*Runnable<void*()> resourceGetter;
	Runnable<void(void*)> resourceDisposer;
	*/
	void* ptr = nullptr;
	Runnable<void(void*)> disposer;
	JobHandle handle;
	Runnable<JobHandle()> getJobHandle;//The job to generate self
	void Dispose();

public:
	~RuntimeType();
};

class GraphGlobalData {
	friend class GraphNodeData;
	template<typename T>
	friend class ResourceName;
	using Map = HashMap<vengine::string_view, RuntimeType>;
	Map runtimeType;
	Pool<GraphNodeData, true, true> pool;
	vengine::vector<GraphNodeData*> data;

public:
	GraphNodeData* GetNode();
	GraphGlobalData(
		size_t nodeProbablySize = 256);
	JobBucket* bucket;
	void DisposeCompile();
	~GraphGlobalData();
};

class GraphNodeData {
	template<typename T>
	friend class ResourceName;
	template<typename... Types>
	friend class TupleCall;

	GraphGlobalData* glbData;
	///////////// Precompile
	RuntimeType* retValue;
	vengine::vector<RuntimeType*> vec;
	///////////// Result
	Runnable<void()> executeFunc;
	template<typename Ret>
	void ProcessRetValue(vengine::string_view retValueName) {
		auto retValueIte = glbData->runtimeType.Find(retValueName);
		if (!retValueIte) {
			retValueIte = glbData->runtimeType.Emplace(retValueName);
		}
		retValue = &retValueIte.Value();
		auto&& va = *retValue;
		va.tp = typeid(Ret);
		if constexpr (!std::is_same_v<Ret, void>) {
			va.disposer = [](void* pp) {
				if constexpr (!std::is_pointer_v<Ret>) {
					reinterpret_cast<Ret*>(pp)->~Ret();
					vengine_free(pp);
				}
			};
		}
		va.getJobHandle = [&]() {
			auto&& handle = retValue->handle;
			if (handle) return handle;
			vengine::vector<JobHandle> dependJobs(vec.size());
			for (size_t i = 0; i < vec.size(); ++i) {
				dependJobs[i] = vec[i]->getJobHandle();
			}
			handle = glbData->bucket->GetTask(dependJobs.data(), dependJobs.size(), std::move(executeFunc));
			return handle;
		};
	}

public:
	GraphNodeData(
		GraphGlobalData* glb) : glbData(glb) {}
	~GraphNodeData();
	void Execute() const;
};

template<typename T>
class ResourceName {
	template<typename... Types>
	friend class TupleCall;
	vengine::string_view str;
	size_t index;

	void Init(size_t index, GraphNodeData* glb) {
		this->index = index;
		auto ite = glb->glbData->runtimeType.Find(str);
		if (ite) {
			glb->vec.emplace_back(&ite.Value());
		} else {
			glb->vec.emplace_back(
				&glb->glbData->runtimeType.Emplace(
											  str)
					 .Value());
		}
	}
	decltype(auto) Get(GraphNodeData* glb) const {
		if constexpr (std::is_pointer_v<T>) {
			return reinterpret_cast<T>(glb->vec[index]->ptr);
		} else {
			T& pp = *reinterpret_cast<T*>(glb->vec[index]->ptr);
			return pp;
		}
	}

public:
	ResourceName(vengine::string_view str) : str(str) {}
	ResourceName(char const* str) : str(str) {}
};

template<>
class TupleCall<> {
public:
	template<typename Ret, typename Func, typename... FrontArgs>
	static void Compile(vengine::string_view retValueName, Func&& func, GraphNodeData* glb, size_t index) {
		glb->ProcessRetValue<Ret>(retValueName);
		glb->executeFunc = [=, f = std::forward<Func>(func)]() {
			if constexpr (std::is_same_v<Ret, void>) {
				f();
			} else {
				if constexpr (std::is_pointer_v<Ret>) {
					glb->retValue->ptr = reinterpret_cast<void*>(std::move(f()));
				} else {
					glb->retValue->ptr = new (vengine_malloc(sizeof(Ret))) Ret(std::move(f()));
				}
			}
		};
	}
};

template<typename T>
class TupleCall<T> {
public:
	template<typename Ret, typename Func, typename... FrontArgs>
	static void Compile(vengine::string_view retValueName, Func&& func, GraphNodeData* glb, size_t index, ResourceName<FrontArgs>&&... frontArgs, ResourceName<T>&& t) {
		glb->ProcessRetValue<Ret>(retValueName);
		t.Init(index, glb);
		glb->executeFunc = [=, f = std::forward<Func>(func)]() {
			if constexpr (std::is_same_v<Ret, void>) {
				f(frontArgs.Get(glb)..., t.Get(glb));
			} else {
				if constexpr (std::is_pointer_v<Ret>) {
					glb->retValue->ptr = reinterpret_cast<void*>(std::move(
						f(frontArgs.Get(glb)..., t.Get(glb))));
				} else {
					glb->retValue->ptr = new (vengine_malloc(sizeof(Ret))) Ret(std::move(
						f(frontArgs.Get(glb)..., t.Get(glb))));
				}
			}
		};
	}
};

template<typename T, typename... Types>
class TupleCall<T, Types...> {
public:
	template<typename Ret, typename Func, typename... FrontArgs>
	static void Compile(vengine::string_view retValueName, Func&& func, GraphNodeData* glb, size_t index, ResourceName<FrontArgs>&&... frontArgs, ResourceName<T>&& t, ResourceName<Types>&&... args) {
		t.Init(index, glb);
		TupleCall<Types...>::template Compile<Ret, Func, FrontArgs..., T>(
			retValueName,
			std::forward<Func>(func),
			glb, index + 1,
			std::forward<ResourceName<FrontArgs>>(frontArgs)...,
			std::forward<ResourceName<T>>(t),
			std::forward<ResourceName<Types>>(args)...);
	}
};

template<typename T>
class TupleCallFilter;
template<typename T, typename... Types>
class TupleCallFilter<T(Types...)> {
public:
	using Type = TupleCall<Types...>;
	using RetType = T;
};

template<typename Func, typename... ArgNames>
GraphNodeData const* CompileJob(vengine::string_view retValue, Func&& func, GraphGlobalData* glb, ArgNames&&... names) {
	using Filter = TupleCallFilter<FuncType<Func>>;
	using TupleCallType = typename Filter::Type;
	auto node = glb->GetNode();
	TupleCallType::template Compile<typename Filter::RetType, Func>(
		retValue, std::forward<Func>(func),
		node, 0, names...);
	return node;
}