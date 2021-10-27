#include "lib.h"
export void irun(
uniform uint x,
uniform uint y,
uniform uint z,
uniform uint64 arg) {
uint64* intPtr = (uint64*)arg;
uint64* vecPtr = (uint64*)intPtr[0];
foreach(i = 0 ... x, j = 0 ... y, k = 0 ... z){
vecPtr[i] += intPtr[1];
}
}