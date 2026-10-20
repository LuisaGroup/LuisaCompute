// Compatibility shim for stb_image_resize2.h.
//
// Upstream nothings/stb moved stb_image_resize2.h out of the repository
// (it now lives in nothings/stb_image_resize) and only keeps a deprecated
// copy under stb/deprecated/. Recent submodule updates therefore no longer
// provide <stb/stb_image_resize2.h> from the submodule itself.
//
// This shim is only reached when the submodule does not ship the real header
// (the submodule include directory is searched first), so builds keep working
// with both the old and the new upstream layout.
#ifndef LUISA_STB_IMAGE_RESIZE2_COMPAT_H
#define LUISA_STB_IMAGE_RESIZE2_COMPAT_H
#include <stb/deprecated/stb_image_resize2.h>
#endif
