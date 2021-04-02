#pragma once
#include "IShader.h"
#include "../Struct/ShaderVariableType.h"
class JobBucket;
class DescriptorHeap;
class CommandSignature;
struct ShaderPass {
	vengine::string name;
	Microsoft::WRL::ComPtr<ID3DBlob> vsShader = nullptr;
	Microsoft::WRL::ComPtr<ID3DBlob> psShader = nullptr;
	Microsoft::WRL::ComPtr<ID3DBlob> hsShader = nullptr;
	Microsoft::WRL::ComPtr<ID3DBlob> dsShader = nullptr;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};

struct ShaderInputPass {
	vengine::string filePath;
	vengine::string vertex;
	vengine::string fragment;
	D3D12_RASTERIZER_DESC rasterizeState;
	D3D12_DEPTH_STENCIL_DESC depthStencilState;
	D3D12_BLEND_DESC blendState;
};

class ShaderLoader;
class VENGINE_DLL_RENDERER Shader final : public IShader {
	friend class ShaderLoader;
	friend class CommandSignature;

private:
	vengine::vector<ShaderPass> allPasses;
	Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature;
	HashMap<uint, uint> mVariablesDict;
	HashMap<vengine::string, uint> passName;
	vengine::vector<ShaderVariable> mVariablesVector;
	vengine::string name;
	StackObject<SerializedObject, true> serObj;
	Shader() {}
	~Shader();
	Shader(vengine::string const& name, GFXDevice* device, const vengine::string& csoFilePath);
	//bool TrySetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, const std::type_info& tyid) const;
	bool SetRes(ThreadCommand* commandList, uint id, const VObject* targetObj, uint64 indexOffset, ResourceType tyid) const override;
	bool SetResWithoutCheck(ThreadCommand* commandList, HashMap<uint, uint>::Iterator& ite, const VObject* targetObj, uint64 indexOffset, ResourceType tyid) const;

public:
	SerializedObject const* GetJsonObject() const override {
		return serObj.Initialized() ? serObj : nullptr;
	}
	vengine::string const& GetName() const override {
		return name;
	}
	uint GetPassIndex(const vengine::string& name) const;
	void GetPassPSODesc(uint pass, D3D12_GRAPHICS_PIPELINE_STATE_DESC* targetPSO) const;
	void BindShader(ThreadCommand* commandList, const DescriptorHeap* descHeap) const override;
	void BindShader(ThreadCommand* commandList) const override;
	int32_t GetPropertyRootSigPos(uint id) const override;
	bool SetBufferByAddress(ThreadCommand* commandList, uint id, GpuAddress address) const override;
	size_t VariableLength() const override { return mVariablesVector.size(); }
	template<typename Func>
	void IterateVariables(Func&& f) {
		for (int32_t i = 0; i < mVariablesVector.size(); ++i) {
			f(mVariablesVector[i]);
		}
	}

	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(Shader)
};
