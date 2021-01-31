#pragma once
#include "../Common/Memory.h"
#include <mutex>
class IPipelineResource
{
public:
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	virtual ~IPipelineResource() {}
};