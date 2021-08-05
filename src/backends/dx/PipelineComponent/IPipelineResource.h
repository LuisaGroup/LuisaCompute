#pragma once
#include <util/Memory.h>
#include <mutex>
class IPipelineResource
{
public:
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	virtual ~IPipelineResource() {}
};
