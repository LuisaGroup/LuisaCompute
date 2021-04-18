//#endif
#include <RenderComponent/PSOContainer.h>
#include <Singleton/MeshLayout.h>
#include <Singleton/Graphics.h>
#include <RenderComponent/RenderTexture.h>
#include <Common/Hash.h>
#include <Common/LockFreeArrayQueue.h>
#include <Utility/TaskThread.h>
#include <PipelineComponent/ThreadCommand.h>
namespace PSOGlobal {
struct LoadCommand {
	GFXDevice* device;
	PSOContainer::PSO* dest;
};

struct PSOGlobalData {
	using PSOHashMap = HashMap<PSOContainer::PSOKey, PSOContainer::PSO, PSOContainer::PSOKeyHash, PSOContainer::PSOKeyEqual>;
	PSOHashMap allPSOState;
	HashMap<Shader const*, vengine::vector<PSOHashMap::Iterator>> shaderToPSOState;
	std::mutex loadMtx;
	spin_mutex searchMtx;
	uint rtCount = 0;
	PSOGlobalData()
		: allPSOState(16),
		  shaderToPSOState(16) {
	}
};
struct UnloadCommand {
	PSOGlobalData::PSOHashMap::Iterator ite;
};
void GeneratePSO(
	GFXDevice* device,
	PSODescriptor const& desc,
	PSOContainer::PSORTSetting const& set,
	PSOContainer::PSO* pso) {
	D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;
	ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
	vengine::vector<D3D12_INPUT_ELEMENT_DESC>* inputElement = MeshLayout::GetMeshLayoutValue(desc.meshLayoutIndex);
	opaquePsoDesc.InputLayout = {inputElement->data(), (uint)inputElement->size()};
	desc.shaderPtr->GetPassPSODesc(desc.shaderPass, &opaquePsoDesc);
	opaquePsoDesc.SampleMask = UINT_MAX;
	opaquePsoDesc.PrimitiveTopologyType = (D3D12_PRIMITIVE_TOPOLOGY_TYPE)desc.topology;
	opaquePsoDesc.NumRenderTargets = set.rtCount;
	memcpy(&opaquePsoDesc.RTVFormats, set.rtFormat, set.rtCount * sizeof(GFXFormat));
	opaquePsoDesc.SampleDesc.Count = 1;
	opaquePsoDesc.SampleDesc.Quality = 0;
	opaquePsoDesc.DSVFormat = (DXGI_FORMAT)set.depthFormat;
	//compile PSO, very costly function
	HRESULT testResult = device->device()->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&pso->obj));
	pso->loadState = (uint8_t)PSOContainer::PSOLoadState::Loaded;
	ThrowIfFailed(testResult);
}
StackObject<TaskThread> taskThread;
StackObject<PSOGlobalData> globalData;
struct AsyncLoadTask {
	struct UnionCommand {
		bool isLoad;
		union {
			LoadCommand loadCmd;
			UnloadCommand unloadCmd;
		};
		UnionCommand() {}
		UnionCommand(LoadCommand const& ld) : loadCmd(ld) {
			isLoad = true;
		}
		UnionCommand(UnloadCommand const& unload) : unloadCmd(unload) {
			isLoad = false;
		}
		void operator=(LoadCommand const& ld) {
			loadCmd = ld;
			isLoad = true;
		}
		void operator=(UnloadCommand const& ld) {
			unloadCmd = ld;
			isLoad = false;
		}
	};
	LockFreeArrayQueue<UnionCommand> commands;
	template<typename T>
	void AddTask(T const& cmd) {
		commands.Push(cmd);
	}
	void operator()() {
		while (true) {
			UnionCommand cmd;
			if (!commands.Pop(&cmd))
				return;
			if (cmd.isLoad) {
				auto&& desc = cmd.loadCmd.dest->desc;
				auto&& set = cmd.loadCmd.dest->rtSettings;
				GeneratePSO(
					cmd.loadCmd.device,
					desc,
					set,
					cmd.loadCmd.dest);
			} else {
				std::lock_guard lck(globalData->searchMtx);
				cmd.unloadCmd.ite.Value().Delete();
			}
		}
	};
};
StackObject<AsyncLoadTask> loadTask;
static PSOContainer::hash_RTSetting rtHash;
}// namespace PSOGlobal
void PSOContainer::Reload(JobBucket* bucket, HashMap<Shader const*, vengine::vector<JobHandle>>& handles, GFXDevice* device) {
	using namespace PSOGlobal;
	globalData->allPSOState.IterateAll(
		[&handles, device, bucket](PSOKey const& key, PSOContainer::PSO& value) -> void {
			auto&& ite = handles.Find(key.desc.shaderPtr);
			JobHandle* ptr = nullptr;
			uint handleCount = 0;
			if (ite) {
				ptr = ite.Value().data();
				handleCount = ite.Value().size();
			}
			bucket->GetTask(ptr, handleCount, [&key, &value, device]() -> void {
				GeneratePSO(
					device,
					key.desc,
					key.rtSettings,
					&value);
			});
		});
}
void PSOContainer::Initialize() {
	using namespace PSOGlobal;
	loadTask.New();
	taskThread.New();
	taskThread->SetFunctor(*loadTask);
	globalData.New();
}
void PSOContainer::Dispose() {
	using namespace PSOGlobal;
	taskThread.Delete();
	loadTask.Delete();
	globalData.Delete();
}
void PSOContainer::BlockLoading() {
	using namespace PSOGlobal;
	taskThread->Complete();
}
void PSOContainer::ReleasePSO(Shader const* shader) {
	using namespace PSOGlobal;
	std::lock_guard lck(globalData->loadMtx);
	auto shaderIte = globalData->shaderToPSOState.Find(shader);
	if (!shaderIte) return;
	for (auto& ite : shaderIte.Value()) {
		std::lock_guard lck(globalData->searchMtx);
		auto&& v = ite.Value();
		loadTask->AddTask(
			UnloadCommand{ite});
	}
	taskThread->ExecuteNext();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	RenderTexture const* const* renderTargets,
	uint rtCount,
	RenderTexture const* depthTex) {

	Graphics::SetRenderTarget(commandList, renderTargets, rtCount, depthTex);
	rtFormats.rtCount = rtCount;
	for (uint i = 0; i < rtCount; ++i) {
		rtFormats.rtFormat[i] = renderTargets[i]->GetFormat();
	}
	if (depthTex)
		rtFormats.depthFormat = depthTex->GetFormat();
	else
		rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTexture const*>& renderTargets,
	RenderTexture const* depthTex) {
	Graphics::SetRenderTarget(commandList, renderTargets, depthTex);
	rtFormats.rtCount = renderTargets.size();
	for (uint i = 0; i < rtFormats.rtCount; ++i) {
		rtFormats.rtFormat[i] = renderTargets.begin()[i]->GetFormat();
	}
	if (depthTex)
		rtFormats.depthFormat = depthTex->GetFormat();
	else
		rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	const RenderTarget* renderTargets,
	uint rtCount,
	const RenderTarget& depth) {
	Graphics::SetRenderTarget(commandList, renderTargets, rtCount, depth);
	rtFormats.rtCount = rtCount;
	for (uint i = 0; i < rtCount; ++i) {
		rtFormats.rtFormat[i] = renderTargets[i].rt->GetFormat();
	}
	if (depth.rt)
		rtFormats.depthFormat = depth.rt->GetFormat();
	else
		rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTarget>& init,
	const RenderTarget& depth) {
	Graphics::SetRenderTarget(commandList, init, depth);
	rtFormats.rtCount = init.size();
	for (uint i = 0; i < rtFormats.rtCount; ++i) {
		rtFormats.rtFormat[i] = init.begin()[i].rt->GetFormat();
	}
	if (depth.rt)
		rtFormats.depthFormat = depth.rt->GetFormat();
	else
		rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	const RenderTarget* renderTargets,
	uint rtCount) {
	Graphics::SetRenderTarget(commandList, renderTargets, rtCount);
	rtFormats.rtCount = rtCount;
	for (uint i = 0; i < rtCount; ++i) {
		rtFormats.rtFormat[i] = renderTargets[i].rt->GetFormat();
	}
	rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::SetRenderTarget(
	ThreadCommand* commandList,
	const std::initializer_list<RenderTarget>& init) {
	Graphics::SetRenderTarget(commandList, init);
	rtFormats.rtCount = init.size();
	for (uint i = 0; i < rtFormats.rtCount; ++i) {
		rtFormats.rtFormat[i] = init.begin()[i].rt->GetFormat();
	}
	rtFormats.depthFormat = GFXFormat_Unknown;
	UpdateHash();
}
void PSOContainer::UpdateHash() {
	rtFormatHash = PSOGlobal::rtHash(rtFormats);
}
PSOContainer::PSOContainer() {
	rtFormats.depthFormat = GFXFormat_Unknown;
	rtFormats.rtCount = 0;
}
PSOContainer::~PSOContainer() {
}
GFXPipelineState* PSOContainer::TryGetPSOStateAsync(
	PSODescriptor const& desc, GFXDevice* device) {
	using namespace PSOGlobal;
	PSOKey key =
		{
			rtFormats,
			desc};
	key.GenerateHash(rtFormatHash);
	auto localIte = localMap.Find(key);
	if (localIte) {
		auto&& v = localIte.Value();
		uint8_t state = v->loadState;
		switch ((PSOLoadState)state) {
			case PSOLoadState::Loaded:
				return v->obj.Get();
			case PSOLoadState::Loading:
				return nullptr;
		}
	} else {
		localIte = localMap.Insert(key);
	}
	PSORTSetting& rtSetting = key.rtSettings;
	PSOGlobalData::PSOHashMap::Iterator ite;
	{
		std::lock_guard lck(globalData->searchMtx);
		ite = globalData->allPSOState.Find(key);
		if (ite) {
			auto&& v = ite.Value();
			uint8_t state = v.loadState;
			localIte.Value() = &v;
			switch ((PSOLoadState)state) {
				case PSOLoadState::Loaded:
					return v.obj.Get();
				case PSOLoadState::Loading:
					return nullptr;
			}
		}
		ite = globalData->allPSOState.Insert(key);
	}
	PSO& pso = ite.Value();
	localIte.Value() = &pso;
	pso.loadState = (uint8_t)PSOLoadState::Loading;
	{
		std::lock_guard lck(globalData->loadMtx);
		auto shaderMapIte = globalData->shaderToPSOState.Insert(
			desc.shaderPtr);
		shaderMapIte.Value().push_back(ite);
		pso.rtSettings = key.rtSettings;
		pso.desc = desc;
		loadTask->AddTask(
			LoadCommand{device, &pso});
		taskThread->ExecuteNext();
	}
	return nullptr;
}
GFXPipelineState* PSOContainer::GetPSOState(PSODescriptor const& desc, GFXDevice* device) {
	using namespace PSOGlobal;
	PSOKey key =
		{
			rtFormats,
			desc};
	key.GenerateHash(rtFormatHash);
	PSORTSetting& rtSetting = key.rtSettings;
	auto WaitTask = [&](auto&& v) -> decltype(auto) {
		if (v.loadState != (uint8_t)PSOLoadState::Deleted) {
			while (v.loadState != (uint8_t)PSOLoadState::Loaded) {
				std::this_thread::yield();
			}
			return v.obj.Get();
		}
	};
	auto localIte = localMap.Find(key);
	if (localIte) {
		auto&& v = *localIte.Value();
		return WaitTask(v);
	} else {
		localIte = localMap.Insert(key);
	}
	PSOGlobalData::PSOHashMap::Iterator ite;
	{
		std::unique_lock lck(globalData->searchMtx);
		ite = globalData->allPSOState.Find(key);
		if (ite) {
			auto&& v = ite.Value();
			lck.unlock();
			localIte.Value() = &v;
			return WaitTask(v);
		}
		ite = globalData->allPSOState.Insert(key);
	}
	{
		std::lock_guard lck(globalData->loadMtx);
		auto shaderMapIte = globalData->shaderToPSOState.Insert(
			desc.shaderPtr);
		shaderMapIte.Value().push_back(ite);
		PSO& pso = ite.Value();
		pso.rtSettings = key.rtSettings;
		pso.desc = desc;
		GeneratePSO(
			device,
			desc,
			key.rtSettings,
			&pso);
		localIte.Value() = &pso;
		return pso.obj.Get();
	}
}
PSOContainer::PSORTSetting::PSORTSetting(GFXFormat* colorFormats, uint colorFormatCount, GFXFormat depthFormat) noexcept {
	rtCount = Min<size_t>((size_t)8, colorFormatCount);
	memcpy(rtFormat, colorFormats, rtCount * sizeof(GFXFormat));
	this->depthFormat = depthFormat;
}
bool PSOContainer::PSORTSetting::operator==(const PSORTSetting& a) const noexcept {
	if (depthFormat == a.depthFormat && rtCount == a.rtCount) {
		for (uint i = 0; i < rtCount; ++i) {
			if (rtFormat[i] != a.rtFormat[i]) return false;
		}
		return true;
	}
	return false;
}
uint64_t PSODescriptor::GenerateHash() const {
	return Hash::CharArrayHash(
		(const char* const)shaderPtr,
		sizeof(shaderPtr) + sizeof(shaderPass) + sizeof(meshLayoutIndex) + sizeof(topology));
}
