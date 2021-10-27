static const std::string_view headerName = R"(#include "lib.h"
export void irun(
    uniform uint x_c,
    uniform uint y_c,
    uniform uint z_z,
    uniform uint64 arg) {

  uint64 a0 = *((uint64*)(arg + 0ull));
  float2 a1 = *((float2*)(arg + 8ull));
  float4 a2 = *((float4*)(arg + 16ull));
)"sv;
static const std::string_view foreachName = "foreach(x = 0 ... x_c, y = 0 ... y_c, z = 0 ... z_c)"sv;
