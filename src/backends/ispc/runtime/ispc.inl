static const std::string_view headerName = R"(#include "lib.h"
export void irun(
    uniform uint xc,
    uniform uint yc,
    uniform uint zc,
    uniform uint64 arg) {
)"sv;
static const std::string_view foreachName = "foreach(x = 0 ... xc, y = 0 ... yc, z = 0 ... zc){"sv;
