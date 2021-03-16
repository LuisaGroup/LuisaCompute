#include "Random.h"
#include <time.h>
#undef max
#undef min
Random::Random() : eng(time(nullptr)), dist(eng.min(), eng.max()) {}
double Random::GetNormFloat() {
	return dist(eng) / (double)(eng.max());
}
uint Random::GetInt() {
	return dist(eng);
}
double Random::GetRangedFloat(double min, double max) {
	double range = max - min;
	return dist(eng) / (((double)eng.max()) / range) + min;
}
