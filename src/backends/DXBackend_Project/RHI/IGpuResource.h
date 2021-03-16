#pragma once
#include "../Common/Common.h"

namespace lc_rhi {
class IGpuResource {
public:
	virtual ~IGpuResource() {}
	virtual uint GetSrvIndex() const = 0;
	virtual uint GetUavIndex() const = 0;
};
}// namespace lc_rhi