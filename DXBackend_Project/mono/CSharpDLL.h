#pragma once
#include <Common/Common.h>
#include <Common/string_view.h>
#include <mono/jit/jit.h>
#include <mono/metadata/assembly.h>
#include <mono/metadata/class.h>
#include <mono/metadata/debug-helpers.h>
#include <mono/metadata/mono-config.h>
using UIntPtr = void*;
using IntPtr = void*;
namespace CSharpDLL {
struct GlobalData;
struct Utility;
struct Domain {
	friend class Utility;
	friend struct GlobalData;

private:
	MonoDomain* domain;
	MonoImage* image;

public:
	MonoDomain* GetDomain() const {
		return domain;
	};
	MonoImage* GetImage() const {
		return image;
	};
	~Domain();

	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	Domain(const char* name);
	KILL_COPY_CONSTRUCT(Domain)
};
struct Class;
struct Method {
private:
	MonoMethod* method;
	vengine::string methodName;
	void CallFunction(void** ptr) const;

public:
	vengine::string const& GetName() const { return methodName; }
	MonoMethod* GetMethod() const { return method; }

	Method(
		Class const& cls,
		vengine::string_view const& methodName);
	template<typename T>
	void CallFunction(T& arg) const {
		using Type = std::remove_cvref_t<T>;
		Type& pureValue = (Type&)(arg);
		void* ptr = &pureValue;
		CallFunction(&ptr);
	}
	void CallFunction() const {
		int32_t a = 0;
		void* ptr = &a;
		CallFunction(&ptr);
	}
	~Method();
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
struct Class {
	friend class Utility;
	friend struct GlobalData;

private:
	vengine::string clsName;
	MonoClass* mono_class;
	HashMap<vengine::string, Method> methods;
	void AddInternalMethod(vengine::string_view const& name, void const* funcPtr);

public:
	vengine::string const& GetName() const { return clsName; }
	MonoClass* GetClass() const { return mono_class; }
	Method* GetMethod(vengine::string const& name);
	void AddExternalMethod(Method const& methd);
	void AddExternalMethod(vengine::string_view const& methd);
	template<typename T>
	void AddInternalMethod(vengine::string_view const& name, funcPtr_t<T> funcPtr) {
		//TODO: May need type check
		AddInternalMethod(name, reinterpret_cast<void const*>(funcPtr));
	}
	~Class();
	Class();
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
private:
	void Create(Domain const& domain, vengine::string_view const& clsName);
	KILL_COPY_CONSTRUCT(Class)
};
struct Utility {
	static void Initialize(const char* name);
	static Class* GetClass(vengine::string const& clsName);
	static void Dispose();

	Utility() = delete;
	KILL_COPY_CONSTRUCT(Utility)
};
}// namespace CSharpDLL