#pragma once
class IBufferAllocator;
class ITextureAllocator;
namespace luisa::compute {
class DXAllocator {
public:
	//TODO
	static IBufferAllocator* GetBufferAllocator() {
		return nullptr;
	}
	static ITextureAllocator* GetTextureAllocator() {
		return nullptr;
	}
};
}// namespace luisa::compute