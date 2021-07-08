#pragma once
#include <Common/GFXUtil.h>
#include <Common/VObject.h>
#include <Common/MetaLib.h>
#include <Common/vector.h>
#include <RenderComponent/UploadBuffer.h>
struct ConstBufferElement
{
	UploadBuffer const* const buffer;
	uint const element;
	ConstBufferElement() : buffer(nullptr),
		element(0)
	{	}
	void operator=(const ConstBufferElement& ele)
	{
		new (this)ConstBufferElement(ele);
	}
	ConstBufferElement(const ConstBufferElement& ele) :
		buffer(ele.buffer),
		element(ele.element) {}
	ConstBufferElement(UploadBuffer const* const buffer, uint const element) :
		buffer(buffer), element(element) {}
};
class VENGINE_DLL_RENDERER CBufferPool
{
private:
	vstd::vector<std::unique_ptr<UploadBuffer>> arr;
	ArrayList<ConstBufferElement> poolValue;
	uint capacity;
	uint stride;
	bool isConstantBuffer;
	void Add(GFXDevice* device);
public:
	CBufferPool(uint stride, uint initCapacity, bool isConstantBuffer = true);
	~CBufferPool();
	ConstBufferElement Get(GFXDevice* device);
	void Return(const ConstBufferElement& target);
	KILL_COPY_CONSTRUCT(CBufferPool)
};