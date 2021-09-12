#include <Common/DynamicLink.h>
namespace vstd {
struct LinkTarget {
	Runnable<void(), VEngine_AllocType::Default> funcPtr;
	Type funcType;
	LinkTarget(
		Type funcType,
		Runnable<void(), VEngine_AllocType::Default>&& funcPtr) : funcType(funcType), funcPtr(std::move(funcPtr)) {}
	void const* Ptr() const {
		return reinterpret_cast<void const*>(&funcPtr);
	}
};
struct LinkMap {
	using HashMapType = HashMap<string_view, LinkTarget, hash<string_view>, std::equal_to<string_view>, VEngine_AllocType::Default>;
	StackObject<HashMapType> map;
	luisa::spin_mutex mtx;
	LinkMap() {
		{
			std::lock_guard lck(mtx);
			map.New(256);
		}
	}
	HashMapType* operator->() { return map; }
	~LinkMap() {
		map.Delete();
	}
};
static LinkMap& GetLinkerHashMap() {
	static LinkMap hashMap;
	return hashMap;
}
void AddFunc(
	string_view const& name,
	Type funcType,
	Runnable<void(), VEngine_AllocType::Default>&& funcPtr) {
	auto&& map = GetLinkerHashMap();
	bool isNew;
	{
		std::lock_guard lck(map.mtx);
		map->TryEmplace(
			isNew,
			name,
			funcType,
			std::move(funcPtr));
	}
	if (!isNew) {
		vstl_log(
			{"Functor Name Conflict: ",
			 name});
		VSTL_ABORT();
	}
}

void const* GetFuncPair(
	Type checkType,
	string_view const& name) {
	auto&& map = GetLinkerHashMap();
	std::lock_guard lck(map.mtx);
	auto ite = map->Find(name);
	if (ite) {
		auto&& v = ite.Value();
		//Not Same Type!
#ifdef VSTL_DEBUG
		if (strcmp(v.funcType.GetType().name(), checkType.GetType().name()) != 0) {
			vstl_log(
				{"Try to access function: ",
				 name,
				 " with wrong type!\n",
				 "input type: ",
				 checkType.GetType().name(),
				 "\ntarget type: ",
				 v.funcType.GetType().name()});
			VSTL_ABORT();
			return nullptr;
		}
#endif
		return v.Ptr();
	}
	return 0;
}

}// namespace vstd