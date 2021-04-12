#pragma once
#include <RenderComponent/Shader.h>
#include <Struct/RenderTarget.h>
#include <Common/HashMap.h>
#include <Common/vector.h>
#include <JobSystem/JobInclude.h>
#include <RenderComponent/IMesh.h>
#include <mutex>
#include <atomic>
class ThreadCommand;
class Graphics;
class IMesh;
class PSOContainer;
class PSODescriptor {
	friend class PSOContainer;
	friend class vengine::hash<PSODescriptor>;
	friend class vengine::hash<std::pair<uint, PSODescriptor>>;

private:
	uint64_t GenerateHash() const;

public:
	const Shader* shaderPtr;
	uint shaderPass;
	uint meshLayoutIndex;
	PrimitiveTopologyType topology;
	PSODescriptor(
		const Shader* shader,
		uint shaderPass,
		IMesh* mesh,
		PrimitiveTopologyType topology = PrimitiveTopologyType_Triangle)
		: shaderPtr(shader),
		  shaderPass(shaderPass),
		  meshLayoutIndex(mesh->GetLayoutIndex()),
		  topology(topology) {
	}
	PSODescriptor(
		const Shader* shader,
		uint shaderPass,
		uint meshLayout,
		PrimitiveTopologyType topology = PrimitiveTopologyType_Triangle)
		: shaderPtr(shader),
		  shaderPass(shaderPass),
		  meshLayoutIndex(meshLayout),
		  topology(topology) {
	}
	bool operator==(const PSODescriptor& other) const {
		return BinaryEqualTo_Size(
			&shaderPtr, &other.shaderPtr,
			sizeof(shaderPtr) + sizeof(shaderPass) + sizeof(meshLayoutIndex) + sizeof(topology));
	}
	PSODescriptor() : topology(PrimitiveTopologyType_Triangle) {
	}
};

namespace vengine {
template<>
struct hash<PSODescriptor> {
	uint64_t operator()(const PSODescriptor& key) const {
		return key.GenerateHash();
	}
};
template<>
struct hash<std::pair<uint, PSODescriptor>> {
	uint64_t operator()(const std::pair<uint, PSODescriptor>& key) const {
		return (Hash::_FNV_offset_basis ^ key.first ^ key.second.GenerateHash()) * Hash::_FNV_prime;
	}
};

}// namespace vengine
namespace PSOGlobal {
struct LoadCommand;
struct AsyncLoadTask;
struct PSOGlobalData;
struct PSOKey;
struct PSOKeyHash;
}// namespace PSOGlobal
class VENGINE_DLL_RENDERER PSOContainer {
	friend class Graphics;
	friend struct PSOGlobal::LoadCommand;
	friend struct PSOGlobal::AsyncLoadTask;
	friend struct PSOGlobal::PSOGlobalData;
	friend struct PSOGlobal::PSOKey;
	friend struct PSOGlobal::PSOKeyHash;
	friend class ThreadCommand;

public:
	struct PSORTSetting {
		GFXFormat depthFormat;
		uint rtCount;
		GFXFormat rtFormat[8];
		PSORTSetting() {}
		PSORTSetting(GFXFormat* colorFormats, uint colorFormatCount, GFXFormat depthFormat) noexcept;
		bool operator==(const PSORTSetting& a) const noexcept;
		bool operator!=(const PSORTSetting& a) const noexcept {
			return !operator==(a);
		}
	};
	enum class PSOLoadState : uint8_t {
		Deleted,
		Loading,
		Loaded
	};
	struct PSO {
		Microsoft::WRL::ComPtr<GFXPipelineState> obj;
		PSOContainer::PSORTSetting rtSettings;
		PSODescriptor desc;
		std::atomic_uint8_t loadState;
		PSO(
			Microsoft::WRL::ComPtr<GFXPipelineState> const& obj,
			PSOContainer::PSORTSetting const& rtSettings,
			PSODescriptor const& desc,
			PSOLoadState loadState) : obj(obj),
									  rtSettings(rtSettings),
									  desc(desc),
									  loadState((uint8_t)loadState) {
		}
		PSO() : loadState((uint8_t)PSOLoadState::Loading) {}
		PSO(PSO const& another) : obj(another.obj),
								  loadState(static_cast<uint8_t>(another.loadState)),
								  rtSettings(another.rtSettings),
								  desc(another.desc) {}
		void operator=(
			PSO const& another) {
			obj = another.obj;
			loadState = static_cast<uint8_t>(another.loadState);
			rtSettings = another.rtSettings;
			desc = another.desc;
		}
		void Delete() {
			obj = nullptr;
			loadState = (uint8_t)PSOLoadState::Deleted;
		}
	};
	struct hash_RTSetting {
		size_t operator()(const PSORTSetting& key) const {
			vengine::hash<GFXFormat> fmtHash;
			size_t h = fmtHash(key.depthFormat);
			for (uint i = 0; i < key.rtCount; ++i) {
				h ^= fmtHash(key.rtFormat[i]);
				h *= Hash::_FNV_prime;
			}
			return h;
		}
	};
	struct PSOKey {
		PSOContainer::PSORTSetting rtSettings;
		PSODescriptor desc;
		uint64_t hash;
		void GenerateHash(uint64 rtFormatHash) {
			static vengine::hash<PSODescriptor> descHash;
			auto bHash = (descHash(desc) << 8);
			hash = (rtFormatHash ^ static_cast<uint64_t>(bHash)) * Hash::_FNV_prime;
		}
	};

	struct PSOKeyHash {
		size_t operator()(PSOKey const& key) const {
			return key.hash;
		}
	};
	struct PSOKeyEqual {
		bool operator()(PSOKey const& a, PSOKey const& b) const {
			return a.rtSettings == b.rtSettings && a.desc == b.desc;
		}
	};

	static void Initialize();
	static void Dispose();
	static void BlockLoading();
	static void ReleasePSO(
		Shader const* shader);
	static void Reload(JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& handles, GFXDevice* device);

private:
	void UpdateHash();
	PSOContainer();
	~PSOContainer();
	void SetRenderTarget(
		ThreadCommand* commandList,
		RenderTexture const* const* renderTargets,
		uint rtCount,
		RenderTexture const* depthTex = nullptr);
	void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTexture const*>& renderTargets,
		RenderTexture const* depthTex = nullptr);
	void SetRenderTarget(
		ThreadCommand* commandList,
		const RenderTarget* renderTargets,
		uint rtCount,
		const RenderTarget& depth);
	void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTarget>& init,
		const RenderTarget& depth);
	void SetRenderTarget(
		ThreadCommand* commandList,
		const RenderTarget* renderTargets,
		uint rtCount);
	void SetRenderTarget(
		ThreadCommand* commandList,
		const std::initializer_list<RenderTarget>& init);

	GFXPipelineState* GetPSOState(PSODescriptor const& desc, GFXDevice* device);
	GFXPipelineState* TryGetPSOStateAsync(
		PSODescriptor const& desc, GFXDevice* device);

	HashMap<PSOContainer::PSOKey, PSOContainer::PSO*, PSOContainer::PSOKeyHash, PSOContainer::PSOKeyEqual> localMap;
	PSORTSetting rtFormats;
	uint64 rtFormatHash = 0;
};