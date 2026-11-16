#include "metal_static_backend.h"

// Referencing the real registration function pulls the backend and its runtime
// dependencies out of their static archives. This is a cross-link test, not a
// claim that a GPU kernel has run on an iPhone.
int main() {
    LUISA_IOS_REGISTER_BACKEND();
    return 0;
}
