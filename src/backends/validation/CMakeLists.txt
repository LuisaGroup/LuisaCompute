set(LUISA_COMPUTE_VALIDATION_SOURCES
        accel.cpp accel.h
        bindless_array.h
        buffer.h
        depth_buffer.h
        device.cpp device.h
        event.h
        mesh.h
        procedural_primitives.h
        resource.cpp resource.h
        rw_resource.cpp rw_resource.h
        shader.cpp shader.h
        stream.cpp stream.h
        swap_chain.h
        texture.h)

luisa_compute_add_backend(validation
        SOURCES ${LUISA_COMPUTE_VALIDATION_SOURCES})
