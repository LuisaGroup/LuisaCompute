#ifdef __aarch64__
#define STBI_NEON
#define STBIR_NEON
#endif

#if _WIN32 || _WIN64
#define STBIDEF  _declspec(dllexport)
#define STBIWDEF  _declspec(dllexport)
#define STBIRDEF  _declspec(dllexport)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// FMA disabled for compatibility
// #define STBIR_USE_FMA
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>
