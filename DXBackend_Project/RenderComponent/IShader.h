#pragma once
#include "../Common/GFXUtil.h"
#include "../Common/VObject.h"
#include "../CJsonObject/SerializedObject.h"
#include "../Struct/ShaderVariableType.h"
class DescriptorHeap;
class StructuredBuffer;
class ComputeShaderCompiler;
class ComputeShaderReader;
class ShaderCompiler;
class ThreadCommand;
class IShader : public VObject {
protected:
	virtual bool SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const = 0;

public:
	~IShader() {}
	virtual SerializedObject const* GetJsonObject() const = 0;
	virtual size_t VariableLength() const = 0;
	virtual int32_t GetPropertyRootSigPos(uint id) const = 0;
	virtual void BindShader(ThreadCommand* commandList) const = 0;
	virtual void BindShader(ThreadCommand* commandList, const DescriptorHeap* heap) const = 0;
	virtual bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const = 0;
	virtual vengine::string const& GetName() const = 0;
	template<typename T>
	bool SetResource(ThreadCommand* commandList, uint id, T* targetObj, uint64 indexOffset) const {
		return SetRes(commandList, id, targetObj, indexOffset, typeid(PureType_t<T>));
	}
};