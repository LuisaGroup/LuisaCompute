#include "CSharpDLL.h"
#include <mono/metadata/threads.h>
namespace CSharpDLL
{
	Domain::Domain(const char* name)
	{
		domain = mono_jit_init(name);
		MonoAssembly* assembly = mono_domain_assembly_open(domain, (vengine::string(name) + ".dll").c_str());
		image = mono_assembly_get_image(assembly);
	}
	Domain::~Domain()
	{
		mono_jit_cleanup(domain);
	}
	Method* Class::GetMethod(vengine::string const& name)
	{
		auto ite = methods.Find(name);
		if (!ite) return nullptr;
		return &ite.Value();
	}
	Class::Class() :
		methods(32)
	{
	}
	void Class::Create(Domain const& domain, vengine::string_view const& clsName)
	{
		this->clsName.clear();
		this->clsName.push_back_all(clsName.begin(), clsName.size());
		mono_class = mono_class_from_name(domain.GetImage(), "", clsName.c_str());
	}
	void Class::AddExternalMethod(Method const& methd)
	{
		methods.Insert(methd.GetName(), methd);
	}
	void Class::AddExternalMethod(vengine::string_view const& methd)
	{
		methods.Insert(vengine::string(methd.begin(), methd.end()), Method(*this, methd));
	}
	void Class::AddInternalMethod(vengine::string_view const& name, void const* funcPtr)
	{
		vengine::string funcName = clsName;
		funcName += "::";
		funcName.push_back_all(name.begin(), name.size());
		mono_add_internal_call(funcName.c_str(), funcPtr);
	}
	Class::~Class()
	{
	}
	Method::Method(
		Class const& cls,
		vengine::string_view const& methodName)
	{
		size_t sz = cls.GetName().size() + methodName.size();
		sz += 8;//():
		vengine::string str;
		str.reserve(sz);
		str.push_back_all(cls.GetName().c_str(), cls.GetName().size());
		str.push_back(':');
		str.push_back_all(methodName.c_str(), methodName.size());
		str += "(void*)";
		MonoMethodDesc* method_desc = mono_method_desc_new(str.c_str(), false);
		method = mono_method_desc_search_in_class(method_desc, cls.GetClass());
		mono_method_desc_free(method_desc);
	}
	Method::~Method() {}
	void Method::CallFunction(void** ptr) const
	{
		mono_thread_attach(mono_get_root_domain());
		mono_runtime_invoke(method, nullptr, ptr, nullptr);
	}
	struct GlobalData
	{
		Domain d;
		HashMap<vengine::string, Class> clsDict;
		GlobalData(const char* c) : d(c), clsDict(32)
		{
		}
	};
	StackObject<GlobalData, true> glb;
	void Utility::Initialize(const char* name)
	{
		glb.New(name);
	}
	Class* Utility::GetClass(vengine::string const& clsName)
	{
		auto ite = glb->clsDict.Find(clsName);
		if (!ite)
		{
			ite = glb->clsDict.Insert(clsName);
			ite.Value().Create(glb->d, clsName);
			return &ite.Value();
		}
		return &ite.Value();
	}
	void Utility::Dispose()
	{
		glb.Delete();
	}
}
