#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

int main() {
    return glfwGetPlatform() == GLFW_PLATFORM_WAYLAND ? 0 : 1;
}
