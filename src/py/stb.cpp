#include <core/stl/memory.h>

#define STBIR_MALLOC(size,c) ((void)(c), luisa::detail::allocator_allocate(size,0))
#define STBIR_FREE(ptr,c)    ((void)(c), luisa::detail::allocator_deallocate(ptr,0))

#define STBI_MALLOC(size) luisa::detail::allocator_allocate(size,0)
#define STBI_FREE(ptr) luisa::detail::allocator_deallocate(ptr,0)
#define STBI_REALLOC(p,newsz) luisa::detail::allocator_reallocate(p,newsz,0)

#define STBIW_MALLOC(size) luisa::detail::allocator_allocate(size,0)
#define STBIW_FREE(ptr) luisa::detail::allocator_deallocate(ptr,0)
#define STBIW_REALLOC(p,newsz) luisa::detail::allocator_reallocate(p,newsz,0)

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
