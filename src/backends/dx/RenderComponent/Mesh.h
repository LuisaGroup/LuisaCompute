#pragma once
#include <RenderComponent/IMesh.h>
class IBufferAllocator;
class DescriptorHeap;
namespace MeshGlobal {
class MeshLoadCommand;
}
struct MeshData {
	GFXFormat indexFormat = GFXFormat_R16_UInt;
	ArrayList<char>* datas = nullptr;
	int64_t vertexCount = -1;
	int64_t indexCount = -1;
	int64_t vertexOffset = -1;
	int64_t normalOffset = -1;
	int64_t tangentOffset = -1;
	int64_t uvOffset = -1;
	int64_t uv2Offset = -1;
	int64_t uv3Offset = -1;
	int64_t uv4Offset = -1;
	int64_t colorOffset = -1;
	int64_t boneIndexOffset = -1;
	int64_t boneWeightOffset = -1;
	int64_t bindPosesOffset = -1;
	int64_t bindPosesCount = -1;
	int64_t indexDataOffset = -1;
	ArrayList<SubMesh> subMeshes;
	float3 boundingCenter;
	float3 boundingExtent;
};
class VENGINE_DLL_RENDERER Mesh final : public GPUResourceBase, public IMesh {
	friend class MeshGlobal::MeshLoadCommand;
	// Data about the buffers

	uint VertexByteStride = 0;
	uint VertexBufferByteSize = 0;
	uint meshLayoutIndex;
	uint mVertexCount;
	GFXFormat indexFormat;
	uint indexCount;
	GPUResourceState initState = GPUResourceState_CopyDest;
	void* indexArrayPtr;
	GFXVertexBufferView vertexView;
	GFXIndexBufferView indexView;
	IBufferAllocator* allocator = nullptr;
	ArrayList<Math::Matrix4> bindPoses;
	ArrayList<SubMesh> subMeshes;
	ObjectPtr<bool> flagPtr;
	mutable spin_mutex srvDescLock;
	mutable uint vboSrvDescIndex = -1;
	mutable uint iboSrvDescIndex = -1;
	mutable uint vboUavDescIndex = -1;
	mutable uint iboUavDescIndex = -1;
	float3 boundingCenter;
	float3 boundingExtent;
	bool flag = false;
	void GenerateVBOSRVGlobalDescriptor(GFXDevice* device) const;
	void GenerateIBOSRVGlobalDescriptor(GFXDevice* device) const;
	void GenerateVBOUAVGlobalDescriptor(GFXDevice* device) const;
	void GenerateIBOUAVGlobalDescriptor(GFXDevice* device) const;
public:
	IOBJECTREFERENCE_OVERRIDE_FUNCTION
	uint GetVertexStride() const { return VertexByteStride; }
	uint GetVerticesByteSize() const {
		return VertexBufferByteSize;
	}
	GFXResourceState GetGFXResourceState(GPUResourceState gfxState) const override;
	bool IsLoaded() const { return flag; }
	static uint64_t GetMeshSize(
		uint vertexCount,
		bool positions,
		bool normals,
		bool tangents,
		bool colors,
		bool uv,
		bool uv1,
		bool uv2,
		bool uv3,
		bool boneIndex,
		bool boneWeight,
		uint indexCount,
		GFXFormat indexFormat);
	uint GetIndexCount() const override { return indexCount; }
	uint GetIndexFormat() const { return indexFormat; }
	uint GetLayoutIndex() const override { return meshLayoutIndex; }
	uint GetVertexCount() const override { return mVertexCount; }
	GFXVertexBufferView const* VertexBufferViews() const override { return &vertexView; }//static mesh support only one vbv
	uint VertexBufferViewCount() const override { return 1; }
	GFXIndexBufferView IndexBufferView() const override { return indexView; }
	uint GetSubMeshCount() const override;
	SubMesh const& GetSubMesh(uint i) const override;
	float3 GetBoundingCenter() const override;
	float3 GetBoundingExtent() const override;
	uint GetVBOSRVDescIndex(GFXDevice* device) const override;
	uint GetIBOSRVDescIndex(GFXDevice* device) const override;
	uint GetVBOUAVDescIndex(GFXDevice* device) const;
	uint GetIBOUAVDescIndex(GFXDevice* device) const;
	void VboBindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	void VboBindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	void IboBindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	void IboBindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	Mesh(
		uint vertexCount,
		float3* positions,
		float3* normals,
		float4* tangents,
		float4* colors,
		float2* uv,
		float2* uv1,
		float2* uv2,
		float2* uv3,
		int4* boneIndex,
		float4* boneWeight,
		GFXDevice* device,
		GFXFormat indexFormat,
		uint indexCount,
		void* indexArrayPtr,
		IBufferAllocator* allocator,
		SubMesh* subMeshes = nullptr,
		uint subMeshCount = 0,
		float3 const& boundingCenter = {0, 0, 0},
		float3 const& boundingExtent = {0, 0, 0});
	Mesh(
		uint vertexCount,
		float3* positions,
		float3* normals,
		float4* tangents,
		float4* colors,
		float2* uv,
		float2* uv1,
		float2* uv2,
		float2* uv3,
		int4* boneIndex,
		float4* boneWeight,
		GFXDevice* device,
		GFXFormat indexFormat,
		uint indexCount,
		void* indexArrayPtr,
		const Microsoft::WRL::ComPtr<GFXResource>& defaultBuffer = nullptr,
		uint64_t defaultOffset = 0,
		const Microsoft::WRL::ComPtr<GFXResource>& uploadBuffer = nullptr,
		uint64_t uploadOffset = 0,
		SubMesh* subMeshes = nullptr,
		uint subMeshCount = 0,
		float3 const& boundingCenter = {0, 0, 0},
		float3 const& boundingExtent = {0, 0, 0});

	static ObjectPtr<Mesh> LoadMeshFromFile(
		MeshData& meshData,
		const vengine::string& str,
		GFXDevice* device,
		bool normals,
		bool tangents,
		bool colors,
		bool uv,
		bool uv1,
		bool uv2,
		bool uv3,
		bool bone,
		IBufferAllocator* allocator = nullptr,
		ArrayList<char>* dataPtr = nullptr);

	static ObjectPtr<Mesh> LoadMeshFromFile(
		const vengine::string& str,
		GFXDevice* device,
		bool normals,
		bool tangents,
		bool colors,
		bool uv,
		bool uv1,
		bool uv2,
		bool uv3,
		bool bone,
		IBufferAllocator* allocator = nullptr,
		ArrayList<char>* dataPtr = nullptr);
	static bool LoadMeshToArray(
		const vengine::string& str,
		bool normals,
		bool tangents,
		bool colors,
		bool uv,
		bool uv1,
		bool uv2,
		bool uv3,
		bool bone,
		ArrayList<char>& dataPtr,
		MeshData& meshData);
	static void LoadMeshFromFiles(
		const vengine::vector<vengine::string>& str, GFXDevice* device,
		bool normals,
		bool tangents,
		bool colors,
		bool uv,
		bool uv1,
		bool uv2,
		bool uv3,
		bool bone,
		vengine::vector<ObjectPtr<Mesh>>& results,
		ArrayList<char>* dataPtr = nullptr);
	~Mesh();
	GPUResourceState GetInitState() const {
		return initState;
	}
	Math::Matrix4* BindPosesData() const noexcept {
		return bindPoses.data();
	}
	uint64_t BindPosesCount() const noexcept {
		return bindPoses.size();
	}
	KILL_COPY_CONSTRUCT(Mesh)
};