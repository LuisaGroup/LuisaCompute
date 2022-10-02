#include <api/runtime.h>

int main(int argc, char** argv) {
    LCContext ctx = luisa_compute_context_create(argv[0]);
    LCDevice device = luisa_compute_device_create(ctx, "cuda", "{}");
    luisa_compute_device_release(device);
    luisa_compute_context_destroy(ctx);
    return 0;
}