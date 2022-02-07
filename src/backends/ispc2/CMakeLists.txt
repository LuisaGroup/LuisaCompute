message(STATUS "Build with ISPC")

set(ISPC_BACKEND_SOURCES
        ispc_device.cpp ispc_device.h
        ispc_stream.cpp ispc_stream.h
        ispc_event.cpp ispc_event.h
        ispc_codegen.cpp ispc_codegen.h)

luisa_compute_add_backend(ispc SOURCES ${ISPC_BACKEND_SOURCES})
