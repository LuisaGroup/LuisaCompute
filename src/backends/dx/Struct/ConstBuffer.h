#pragma once
#include <util/MetaLib.h>
#include <util/vector.h>
class VENGINE_DLL_COMMON CBVarBase {
protected:
	size_t offset;
	size_t size;
	funcPtr_t<void(void*, CBVarBase const*)> copyFunc;
	CBVarBase(
		size_t sz,
		funcPtr_t<void(void*, CBVarBase const*)> copyFunc,
		bool isSub);

public:
	size_t GetOffset() const {
		return offset;
	}
	size_t GetSize() const {
		return size;
	}
	funcPtr_t<void(void*, CBVarBase const*)> GetCopyFunc() const {
		return copyFunc;
	}
};
template<typename T>
class CBVar;
class VENGINE_DLL_COMMON ConstBuffer {
private:
	static ConstBuffer*& GetBaseBufferRef();
	static void SetSubMode(bool b);
	size_t size = 0;
	size_t byteAlign = 0;
	vstd::vector<size_t> vars;
	friend class CBVarBase;

public:
	template<typename Func>
	void GetVars(Func const& func) {
		for (auto& i : vars) {
			CBVarBase* ptr =
				reinterpret_cast<CBVarBase*>(
					reinterpret_cast<size_t>(this) + i);
			func(ptr);
		}
	}
	ConstBuffer();
	size_t GetSize() const;
	void CopyDataTo(void* dest) const;
	virtual ~ConstBuffer();
};

template<typename T>
class CBVar final: public CBVarBase {

public:
	StackObject<T> data;
	static constexpr bool IsSubBuffer = std::is_base_of_v<ConstBuffer, T>;
	CBVar() : CBVarBase(
		sizeof(T),
		[](void* dest, CBVarBase const* source) -> void {
			CBVar const* ptr = static_cast<CBVar const*>(source);
			if constexpr (IsSubBuffer) {
				auto cb = static_cast<ConstBuffer const*>(ptr->data.operator T const *());
				cb->CopyDataTo(dest);
			} else {
				*reinterpret_cast<T*>(dest) = *ptr->data;
			}
		},
		IsSubBuffer) {
		static_assert(IsSubBuffer || alignof(T) == 4, "ConstantBuffer Must be 4 byte aligned!");
		data.New();
	}

	~CBVar() {
		data.Delete();
	}

	void operator=(T const& t) {
		*data = t;
	}
	operator T&() {
		return *data;
	}
	T* operator->() {
		return data;
	}
	T const* operator->() const {
		return data;
	}
	operator T const &() const {
		return *data;
	}
};
#include <Common/DXMath/DXMath.h>
using CBFloat = CBVar<float>;
using CBFloat2 = CBVar<float2>;
using CBFloat3 = CBVar<float3>;
using CBFloat4 = CBVar<float4>;

using CBFloat4x4 = CBVar<float4x4>;
using CBFloat3x4 = CBVar<float3x4>;
using CBFloat4x3 = CBVar<float4x3>;
using CBFloat3x3 = CBVar<float3x3>;

using CBUInt = CBVar<uint>;
using CBUInt2 = CBVar<uint2>;
using CBUInt3 = CBVar<uint3>;
using CBUInt4 = CBVar<uint4>;

using CBInt = CBVar<int32_t>;
using CBInt2 = CBVar<int2>;
using CBInt3 = CBVar<int3>;
using CBInt4 = CBVar<int4>;
