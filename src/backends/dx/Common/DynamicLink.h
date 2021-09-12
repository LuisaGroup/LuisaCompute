#pragma once
#include <Common/Common.h>
#include <util/Runnable.h>
#ifdef VSTL_DEBUG
#include <string.h>
#endif
namespace vstd {
LUISA_DLL void AddFunc(
	string_view const& name,
	Type funcType,
	Runnable<void(), VEngine_AllocType::Default>&& funcPtr);
LUISA_DLL void const* GetFuncPair(
	Type checkType,
	string_view const& name);
template<typename T>
void LoadFunction_T(
	string_view const& name,
	Runnable<functor_t<T>, VEngine_AllocType::Default>&& func) {
	AddFunc(name, typeid(functor_t<T>), reinterpret_cast<Runnable<void(), VEngine_AllocType::Default>&&>(func));
}
template<typename T>
struct FunctionLoader {
	FunctionLoader(
		string_view const& name,
		Runnable<functor_t<T>, VEngine_AllocType::Default> func) {
		LoadFunction_T<T>(name, std::move(func));
	}
};
template<typename T>
Runnable<functor_t<T>, VEngine_AllocType::Default> const& TryGetFunction(
	string_view const& name) {
	auto pair = GetFuncPair(typeid(functor_t<T>), name);
#ifdef VSTL_DEBUG
	if (pair == 0) {
		vstl_log(
			{"Try Get Function "_sv,
			 name,
			 " Failed!\n"});
		VSTL_ABORT();
	}
#endif
	return *reinterpret_cast<Runnable<functor_t<T>, VEngine_AllocType::Default> const*>(pair);
}
}// namespace vstd
#define VENGINE_LINK_FUNC(FUNC) static vstd::FunctionLoader<decltype(FUNC)> VE_##FUNC(#FUNC##_sv, FUNC)
#define VENGINE_LINK_CLASS(FUNC, ...) static vstd::FunctionLoader<decltype(FUNC)> VE_##FUNC(#FUNC##_sv, FUNC(__VA_ARGS__))
