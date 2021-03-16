#pragma once
#include "../../Common/GFXUtil.h"
#include "../../Common/LockFreeArrayQueue.h"
#include "../../Struct/RenderPackage.h"
#include "../CBufferAllocator.h"
class RenderTexture;
class ITextureAllocator;
class IBufferAllocator;
class ComputeShader;
class UploadBuffer;
class VirtualVoxelMap {
public:
	enum class DirtyMaskType : uint8_t {
		ALL = 255,
		None = 0,
		Left = 1,
		Right = 2,
		Down = 4,
		Top = 8,
		Back = 16,
		Forward = 32
	};

	enum CustomDataType : uint8_t {
		None = 1,
		UInt = 2,
		UInt3 = 4
	};

private:
	static GFXFormat GetIndirectRTFormat(CustomDataType type) {
		switch (type) {
			case CustomDataType::None:
				return GFXFormat_R16_UInt;
			case CustomDataType::UInt:
				return GFXFormat_R16G16_UInt;
			case CustomDataType::UInt3:
				return GFXFormat_R16G16B16A16_UInt;
		}
	}
	struct Chunk {
		std::unique_ptr<RenderTexture> rt;
		DirtyMaskType dirtyMask;
		Chunk();
		Chunk(Chunk&& c);
		~Chunk();
	};
	enum class CommandType : uint8_t {
		GenerateEdge
	};
	struct Command {
		CommandType type;
		union {
			uint3 generateEdgeIndex;
		};
		Command() {}
		Command(uint3 generateEdgeIndex)
			: type(CommandType::GenerateEdge) {
			this->generateEdgeIndex = generateEdgeIndex;
		}
	};
	struct ComputeProps {
		ComputeShader const* shader;
		uint _SelfMap;
		uint _IndirectTex;
		uint _IndirectBuffer;
		uint _SRV_IndirectTex;
	};
	ComputeProps props;
	vengine::vector<Chunk> allChunks;
	SingleThreadArrayQueue<Command> executeCommand;
	std::unique_ptr<RenderTexture> indirectRT;
	std::unique_ptr<UploadBuffer> indirectUpload;
	uint3 indirectSize;
	uint3 voxelChunkResolution;
	CustomDataType customDataType;
	GFXFormat format;
	bool indirectIsDirty = true;
	bool generateEdge;
	uint3 RepeatIndex(int3 voxelIndex) const;
	uint GetIndex(int3 voxelIndex) const;
	uint GetIndex(uint3 voxelIndex) const;
	void MarkChunkAsDirty(int3 indirectIndex);

public:
	uint3 GetResolution() const { return voxelChunkResolution; }
	uint3 GetIndirectMapSize() const { return indirectSize; }
	VirtualVoxelMap(
		GFXDevice* device,
		ITextureAllocator* texAlloc,
		IBufferAllocator* buffAlloc,
		uint3 maxIndirectSize,
		uint3 voxelChunkResolution,
		GFXFormat format,
		bool processEdge,
		CustomDataType customDataType = CustomDataType::None);
	RenderTexture* GetRenderTextureChunk(int3 indirectIndex) const;
	RenderTexture* GetIndirectTex() const;
	bool IsChunkExists(int3 indirectIndex) const;
	void TryCreateChunk(GFXDevice* device, ITextureAllocator* allocator, int3 indirectIndex, uint3 customData);
	void TryDestroyChunk(int3 indirectIndex);
	void UpdateTextureChunk(int3 indirectIndex);
	void ExecuteCommands(RenderPackage const& package, Runnable<CBufferChunk(size_t)> const& getCBufferChunk);
	~VirtualVoxelMap();
};