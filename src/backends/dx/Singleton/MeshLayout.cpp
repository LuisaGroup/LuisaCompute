#include <Singleton/MeshLayout.h>
MeshLayout* MeshLayout::current = nullptr;
ArrayList<D3D12_INPUT_ELEMENT_DESC>* MeshLayout::GetMeshLayoutValue(uint index) {
	return current->layoutValues[index];
}
void MeshLayout::Initialize() {
	if (!current)
		current = new MeshLayout;
}
void MeshLayout::Dispose() {
	if (current) {
		delete current;
		current = nullptr;
	}
}
MeshLayout::~MeshLayout() {
	for (auto& i : layoutValues) {
		delete i;
	}
}
void MeshLayout::GenerateDesc(
	ArrayList<D3D12_INPUT_ELEMENT_DESC>& target,
	MeshLayoutKey const& layoutKey) {
	target.reserve(10);
	vstd::vector<uint> byteOffsets;
	byteOffsets.reserve(10);
	auto InputStringByte = [&](char const* name, GFXFormat format, uint byteSize, uint semanticIndex, int32 slot) -> void {
		if (slot >= 0) {
			for (uint i = byteOffsets.size(); i <= slot; ++i) {
				byteOffsets.emplace_back(0);
			}
			uint& curOffset = byteOffsets[slot];
			target.push_back(
				{name,
				 semanticIndex,
				 (DXGI_FORMAT)format,
				 (uint)slot,
				 curOffset,
				 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
				 0});
			curOffset += byteSize;
		} else {
			target.push_back(
				{name,
				 semanticIndex,
				 (DXGI_FORMAT)format,
				 0,
				 0,
				 D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
				 0});
		}
	};
	InputStringByte("POSITION", GFXFormat_R32G32B32_Float, 12, 0, layoutKey.position);
	InputStringByte("NORMAL", GFXFormat_R32G32B32_Float, 12, 0, layoutKey.normal);
	InputStringByte("TANGENT", GFXFormat_R32G32B32A32_Float, 16, 0, layoutKey.tangent);
	InputStringByte("COLOR", GFXFormat_R32G32B32A32_Float, 16, 0, layoutKey.color);
	InputStringByte("TEXCOORD", GFXFormat_R32G32_Float, 8, 0, layoutKey.uv0);
	InputStringByte("TEXCOORD", GFXFormat_R32G32_Float, 8, 1, layoutKey.uv1);
	InputStringByte("TEXCOORD", GFXFormat_R32G32_Float, 8, 2, layoutKey.uv2);
	InputStringByte("TEXCOORD", GFXFormat_R32G32_Float, 8, 3, layoutKey.uv3);
	InputStringByte("BONEINDEX", GFXFormat_R32G32B32A32_SInt, 16, 0, layoutKey.boneIndex);
	InputStringByte("BONEWEIGHT", GFXFormat_R32G32B32A32_Float, 16, 0, layoutKey.boneWeight);
	/*
	target.push_back(
		{"POSITION", 0, (DXGI_FORMAT)GFXFormat_R32G32B32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	if (normal) {
		target.push_back(
			{"NORMAL", 0, (DXGI_FORMAT)GFXFormat_R32G32B32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 12;
	} else {
		target.push_back(
			{"NORMAL", 0, (DXGI_FORMAT)GFXFormat_R32G32B32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (tangent) {
		target.push_back(
			{"TANGENT", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 16;
	} else {
		target.push_back(
			{"TANGENT", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (color) {
		target.push_back(
			{"COLOR", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 16;
	} else {
		target.push_back(
			{"COLOR", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (uv0) {
		target.push_back(
			{"TEXCOORD", 0, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 8;
	} else {
		target.push_back(
			{"TEXCOORD", 0, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (uv2) {
		target.push_back(
			{"TEXCOORD", 1, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 8;
	} else {
		target.push_back(
			{"TEXCOORD", 1, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (uv3) {
		target.push_back(
			{"TEXCOORD", 2, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 8;
	} else {
		target.push_back(
			{"TEXCOORD", 2, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (uv4) {
		target.push_back(
			{"TEXCOORD", 3, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 8;
	} else {
		target.push_back(
			{"TEXCOORD", 3, (DXGI_FORMAT)GFXFormat_R32G32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}
	if (bone) {
		target.push_back(
			{"BONEINDEX", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_SInt, 0, offset, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		target.push_back(
			{"BONEWEIGHT", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, offset + 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		offset += 32;
	} else {
		target.push_back(
			{"BONEINDEX", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_SInt, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
		target.push_back(
			{"BONEWEIGHT", 0, (DXGI_FORMAT)GFXFormat_R32G32B32A32_Float, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0});
	}*/
}
#include <mutex>
namespace MeshLayoutGlobal {
std::mutex mtx;
}
uint MeshLayout::GetMeshLayoutIndex(
	MeshLayoutKey const& layoutKey) {
	using namespace MeshLayoutGlobal;
	std::lock_guard lck(mtx);
	auto ite = current->layoutDict.Find(layoutKey);
	if (!ite) {
		ArrayList<D3D12_INPUT_ELEMENT_DESC>* desc = new ArrayList<D3D12_INPUT_ELEMENT_DESC>();
		GenerateDesc(*desc, layoutKey);
		current->layoutDict.Insert(layoutKey, current->layoutValues.size());
		uint value = (uint)current->layoutValues.size();
		current->layoutValues.push_back(desc);
		return value;
	}
	return ite.Value();
}
