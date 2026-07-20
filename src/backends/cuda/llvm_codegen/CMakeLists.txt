luisa_compute_link_llvm_into_backend(cuda
        COMPONENTS core nvptxcodegen support irreader passes analysis
        DEFINITIONS LUISA_COMPUTE_ENABLE_LLVM=1)
if (LUISA_COMPUTE_CUDA_LINKED_LLVM)
    message(STATUS "Linking LLVM into the CUDA backend")
    # find CUDA libdevice
    set(CUDA_LIBDEVICE_SEARCH_PATHS
            ${CUDAToolkit_LIBRARY_DIR}/../nvvm/libdevice
            ${CUDAToolkit_BIN_DIR}/../nvvm/libdevice
    )
    find_file(CUDA_LIBDEVICE_FILE NAMES libdevice.10.bc HINTS ${CUDA_LIBDEVICE_SEARCH_PATHS} NO_DEFAULT_PATH)
    if (CUDA_LIBDEVICE_FILE)
        message(STATUS "Found CUDA libdevice: ${CUDA_LIBDEVICE_FILE}")
    else ()
        message(FATAL_ERROR "Could not find CUDA libdevice in ${CUDA_LIBDEVICE_SEARCH_PATHS}")
    endif ()
    # use luisa-embed-device-lib to embed libdevice
    set(LUISA_COMPUTE_CUDA_DEVICE_LIB_SOURCES ${CUDA_LIBDEVICE_FILE})
    add_custom_target(luisa-compute-backend-cuda-embed-libdevice
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${LUISA_COMPUTE_CUDA_DEVICE_LIB_SOURCES} --unsigned --prefix luisa_compute_cuda_ -o cuda_codegen_llvm_device_bitcode.cpp -h cuda_codegen_llvm_device_bitcode.h
            BYPRODUCTS ${CMAKE_CURRENT_LIST_DIR}/cuda_codegen_llvm_device_bitcode.cpp ${CMAKE_CURRENT_LIST_DIR}/cuda_codegen_llvm_device_bitcode.h
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE}
            SOURCES ${CUDA_LIBDEVICE_FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding CUDA built-in sources..."
            VERBATIM)
    target_sources(luisa-compute-backend-cuda PRIVATE
            cuda_codegen_llvm_device_bitcode.cpp cuda_codegen_llvm_device_bitcode.h
            cuda_codegen_llvm_config.h
            cuda_codegen_llvm.cpp cuda_codegen_llvm.h
            cuda_codegen_llvm_impl.cpp cuda_codegen_llvm_impl.h
            cuda_codegen_llvm_impl_analysis.cpp
            cuda_codegen_llvm_impl_type.cpp
            cuda_codegen_llvm_impl_inst.cpp
            cuda_codegen_llvm_impl_sreg.cpp
            cuda_codegen_llvm_impl_func.cpp
            cuda_codegen_llvm_impl_cast.cpp
            cuda_codegen_llvm_impl_const.cpp
            cuda_codegen_llvm_impl_var.cpp
            cuda_codegen_llvm_impl_cflow.cpp
            cuda_codegen_llvm_impl_phi.cpp
            cuda_codegen_llvm_impl_atomic.cpp
            cuda_codegen_llvm_impl_autodiff.cpp
            cuda_codegen_llvm_impl_arith.cpp
            cuda_codegen_llvm_impl_cta.cpp
            cuda_codegen_llvm_impl_debug.cpp
            cuda_codegen_llvm_impl_resource.cpp
            cuda_codegen_llvm_impl_rq.cpp
    )
endif ()
