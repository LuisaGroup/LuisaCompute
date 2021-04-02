#include "BackBuffer.h"
BackBuffer ::~BackBuffer() {}

GFXFormat BackBuffer::GetBackBufferFormat() const {
	return (GFXFormat)Resource->GetDesc().Format;
}
