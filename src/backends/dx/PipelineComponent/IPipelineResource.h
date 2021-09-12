#pragma once
#include <util/Memory.h>
#include <mutex>
class IPipelineResource
{
public:
	VSTL_OVERRIDE_OPERATOR_NEW
	virtual ~IPipelineResource() {}
};
