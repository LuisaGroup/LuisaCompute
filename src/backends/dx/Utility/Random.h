#pragma once
#include <random>
#include <Common/DLL.h>
typedef uint32_t uint;
class  Random
{
private:
	std::mt19937 eng;
	std::uniform_int_distribution<uint> dist;// (eng.min(), eng.max());
public:
	Random();
	double GetNormFloat();
	uint GetInt();
	double GetRangedFloat(double min, double max);
};