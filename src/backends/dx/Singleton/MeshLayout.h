#pragma once
#include <Common/GFXUtil.h>
#include <Common/HashMap.h>
#include <Common/vector.h>
#include <Common/DLL.h>
//Mesh layout generate key
//input buffer slot, -1 for nothing
struct MeshLayoutKey {
	int32 position;
	int32 normal;
	int32 tangent;
	int32 color;
	int32 uv0;
	int32 uv1;
	int32 uv2;
	int32 uv3;
	int32 boneIndex;
	int32 boneWeight;
	bool operator==(MeshLayoutKey const& other) const{
		int32 const* ptr = (int32 const*)this;
		int32 const* otherPtr = (int32 const*)&other;
		static constexpr size_t SELF_SIZE = sizeof(MeshLayoutKey) / sizeof(int32);
		for (size_t i = 0; i < SELF_SIZE; ++i) {
			if (ptr[i] != otherPtr[i]) return false;
		}
		return true;
	}
};

class MeshLayout
{
private:
	static MeshLayout* current;
	HashMap<MeshLayoutKey, uint> layoutDict;
	vengine::vector<ArrayList<D3D12_INPUT_ELEMENT_DESC>*> layoutValues;
	static void GenerateDesc(
		ArrayList<D3D12_INPUT_ELEMENT_DESC>& target,
		MeshLayoutKey const& layoutKey
	);
	~MeshLayout();
public:
	static void Initialize();
	static void Dispose();
	static ArrayList<D3D12_INPUT_ELEMENT_DESC>* GetMeshLayoutValue(uint index);
	static uint GetMeshLayoutIndex(
		MeshLayoutKey const& layoutKey
	);
};