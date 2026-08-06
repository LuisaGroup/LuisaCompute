#include <luisa/luisa-compute.h>

int main() {
    static_assert(LUISA_COMPUTE_VERSION > 0);
    luisa::log_level_info();
    luisa::log_flush();
    return 0;
}
