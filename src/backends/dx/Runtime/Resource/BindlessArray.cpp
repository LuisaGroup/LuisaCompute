#pragma vengine_package vengine_directx
#include <Resource/BindlessArray.h>
#include <Resource/TextureBase.h>
#include <Resource/Buffer.h>
#include <Resource/DescriptorHeap.h>
#include <Runtime/CommandBuffer.h>
#include <Runtime/CommandAllocator.h>
namespace toolhub::directx {
BindlessArray::BindlessArray(
	Device* device, uint arraySize)
	: Resource(device),
	  buffer(device, 3 * arraySize * sizeof(uint), device->defaultAllocator) {
	TupleType tpl =
		{
			std::pair<BufferView, uint>{BufferView(nullptr), std::numeric_limits<uint>::max()},
			std::pair<Tex2D, uint>{Tex2D(nullptr), std::numeric_limits<uint>::max()},
			std::pair<Tex3D, uint>{Tex3D(nullptr), std::numeric_limits<uint>::max()}};
	binded.push_back_func(
		[&]() { return tpl; },
		arraySize);
}
template<>
void BindlessArray::TryReturnBind<BufferView>(BufferView& view) {
	if (!view.buffer) return;
	bindedResource.Remove(view);
	view.buffer = nullptr;
}
template<typename T>
void BindlessArray::TryReturnBind(T& view) {
	if (!view.tex) return;
	bindedResource.Remove(view);
	view.tex = nullptr;
}

template<typename T>
void BindlessArray::RemoveLast(T& pair) {
	TryReturnBind(pair.first);
	if (pair.second == std::numeric_limits<uint>::max()) return;
	disposeQueue.Push(pair.second);
	pair.second = std::numeric_limits<uint>::max();
}
template<typename T>
void BindlessArray::AddNew(std::pair<T, uint>& pair, T const& newValue, uint index) {
	pair.first = newValue;
	pair.second = GetNewIndex();
	bindedResource.ForceEmplace(newValue, pair.second);
	updateMap.ForceEmplace(Property::IndexOf<T> * binded.size() + index, pair.second);
}
uint BindlessArray::GetNewIndex() {
	return device->globalHeap->AllocateIndex();
}

BindlessArray::~BindlessArray() {
}
BufferView BindlessArray::GetBufferArray() const {
	return {
		&buffer,
		uint64(0),
		uint64(binded.size() * sizeof(uint))};
}
BufferView BindlessArray::GetTex2DArray() const {
	return {
		&buffer,
		uint64(binded.size() * sizeof(uint)),
		uint64(binded.size() * sizeof(uint))};
}
BufferView BindlessArray::GetTex3DArray() const {
	return {
		&buffer,
		uint64(binded.size() * 2 * sizeof(uint)),
		uint64(binded.size() * sizeof(uint))};
}

void BindlessArray::BindBuffer(BufferView prop, uint index) {
	auto&& pa = std::get<0>(binded[index]);
	RemoveLast(pa);
	AddNew(pa, prop, index);
	auto desc = prop.buffer->GetColorSrvDesc(prop.offset, prop.byteSize);
	if (!desc) {
		VEngine_Log("illegal buffer binding!\n");
		VENGINE_EXIT;
	}
	device->globalHeap->CreateSRV(prop.buffer->GetResource(), *desc, pa.second);
}
void BindlessArray::BindTex2D(Tex2D prop, uint index) {
	auto&& pa = std::get<1>(binded[index]);
	RemoveLast(pa);
	AddNew(pa, prop, index);
	device->globalHeap->CreateSRV(prop.tex->GetResource(), prop.tex->GetColorSrvDesc(), index);
}
void BindlessArray::BindTex3D(Tex3D prop, uint index) {
	auto&& pa = std::get<2>(binded[index]);
	RemoveLast(pa);
	AddNew(pa, prop, index);
	device->globalHeap->CreateSRV(prop.tex->GetResource(), prop.tex->GetColorSrvDesc(), index);
}
void BindlessArray::UnBind(Property prop) {
	prop.visit(
		[&](auto& v) {
			TryReturnBind(v);
		});
}
vstd::optional<uint> BindlessArray::PropertyIdx(Property prop) const {
	auto ite = bindedResource.Find(prop);
	if (ite) return ite.Value();
	return {};
}
void BindlessArray::Update(
	CommandBufferBuilder& builder) {
	auto alloc = builder.GetCB()->GetAlloc();
	auto cmd = builder.CmdList();
	if (updateMap.size() > 0) {
		D3D12_RESOURCE_BARRIER transBarrier;
		transBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		transBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		transBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		transBarrier.Transition.pResource = buffer.GetResource();
		transBarrier.Transition.StateBefore = buffer.GetInitState();
		transBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		cmd->ResourceBarrier(1, &transBarrier);
		auto d = vstd::create_disposer([&] {
			std::swap(transBarrier.Transition.StateBefore, transBarrier.Transition.StateAfter);
			cmd->ResourceBarrier(1, &transBarrier);
		});
		for (auto&& kv : updateMap) {
			builder.Upload(
				BufferView(
					&buffer,
					kv.first * sizeof(uint),
					sizeof(uint)),
				&kv.second);
		}
		updateMap.Clear();
	}

	while (auto i = disposeQueue.Pop()) {
		device->globalHeap->ReturnIndex(*i);
	}
}

}// namespace toolhub::directx