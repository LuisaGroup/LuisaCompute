#pragma once
#include "../../Common/GFXUtil.h"
class Shader;
struct MultiDrawCommand
{
	D3D12_GPU_VIRTUAL_ADDRESS objectCBufferAddress; // Object Constant Buffer Address
	D3D12_VERTEX_BUFFER_VIEW vertexBuffer;			// Vertex Buffer Address
	D3D12_INDEX_BUFFER_VIEW indexBuffer;			//Index Buffer Address
	D3D12_DRAW_INDEXED_ARGUMENTS drawArgs;			//Draw Arguments
};

struct InstanceIndirectCommand
{
	D3D12_DRAW_INDEXED_ARGUMENTS drawArgs;			//Draw Arguments
};

class CommandSignature
{
public:
	enum class SignatureType : uint
	{
		MultiDrawIndirect = 0,
		DrawInstanceIndirect = 1,
		DispatchComputeIndirect = 2
	};
private:
	Microsoft::WRL::ComPtr<ID3D12CommandSignature> mCommandSignature;
	SignatureType sigType;
public:
	CommandSignature(GFXDevice* device, SignatureType sigType, Shader const* drawShader = nullptr);
	ID3D12CommandSignature* GetSignature() const { return mCommandSignature.Get(); }

};