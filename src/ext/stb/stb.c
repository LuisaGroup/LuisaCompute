#ifdef __aarch64__
#define STBI_NEON
#endif
#if _WIN32 || _WIN64
#define STBIWDEF  _declspec(dllexport)
#define STBIDEF  _declspec(dllexport)
#define STBIRDEF  _declspec(dllexport)
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
