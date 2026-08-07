#if defined(LUISA_COMPUTE_SYSTEM_DEPENDENCY_MISSING)
#error "The system GLFW package is unavailable"
#else

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

int main() {
    return glfwGetPlatform() == GLFW_PLATFORM_WAYLAND ? 0 : 1;
}

#endif
