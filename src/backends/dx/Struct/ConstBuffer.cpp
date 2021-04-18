#include <Struct/ConstBuffer.h>
namespace ConstBufferGlobal {
static thread_local ConstBuffer* CURRENT_CBUFFER = nullptr;
static thread_local bool CURRENT_CBUFFER_ENABLE = false;
}// namespace ConstBufferGlobal
ConstBuffer*& ConstBuffer::GetBaseBufferRef() {
	return ConstBufferGlobal::CURRENT_CBUFFER;
}
void ConstBuffer::SetSubMode(bool b) {
	ConstBufferGlobal::CURRENT_CBUFFER_ENABLE = b;
}

ConstBuffer::ConstBuffer() {
	if (!ConstBufferGlobal::CURRENT_CBUFFER_ENABLE)
		GetBaseBufferRef() = this;
}
ConstBuffer::~ConstBuffer() {}
size_t ConstBuffer::GetSize() const {
	return size;
}
void ConstBuffer::CopyDataTo(void* dest) const {
	uint8_t* ptr = reinterpret_cast<uint8_t*>(dest);
	for (auto& i : vars) {
		CBVarBase* var =
			reinterpret_cast<CBVarBase*>(
				reinterpret_cast<size_t>(this) + i);
		var->GetCopyFunc()(
			ptr + var->GetOffset(),
			var);
	}
}
CBVarBase::CBVarBase(
	size_t sz,
	funcPtr_t<void(void*, CBVarBase const*)> copyFunc,
	bool isSub) : copyFunc(copyFunc) {
	ConstBuffer::SetSubMode(isSub);
	if (!isSub) {
		auto cb = ConstBuffer::GetBaseBufferRef();
		if (cb->byteAlign + sz <= 16) {
			cb->byteAlign += sz;
		} else {
			if (cb->byteAlign > 0)
				cb->size += 16 - cb->byteAlign;
			cb->byteAlign = sz & 15;
		}
		offset = cb->size;
		cb->size += sz;
		size = sz;
		cb->vars.push_back(
			reinterpret_cast<size_t>(this) - reinterpret_cast<size_t>(cb));
	}
}