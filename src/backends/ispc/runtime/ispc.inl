static const std::string_view headerName="#include \"lib.h\"\nexport void irun(\nuniform uint x_c,\nuniform uint y_c,\nuniform uint z_z,\nuniform uint64 arg) {\nuint64 a0 = *((uint64*)(arg + 0ull));\nfloat2 a1 = *((float2*)(arg + 8ull));\nfloat4 a2 = *((float4*)(arg + 16ull));\n"sv;
static const std::string_view foreachName = "foreach(x = 0 ... x_c, y = 0 ... y_c, z = 0 ... z_c)"sv;
