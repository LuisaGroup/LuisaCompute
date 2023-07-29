find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

file(GLOB_RECURSE DCUB_SOURCES CONFIGURE_DEPENDS "private/dcub/*.cu")
add_library(luisa-compute-cuda-ext-dcub SHARED ${DCUB_SOURCES})
target_compile_definitions(luisa-compute-cuda-ext-dcub PRIVATE DCUB_DLL_EXPORTS=1)
target_link_libraries(luisa-compute-cuda-ext-dcub PUBLIC CUDA::cudart)
target_include_directories(luisa-compute-cuda-ext-dcub PRIVATE "${CMAKE_SOURCE_DIR}/include") # -> LuisaCompute/include
set_target_properties(luisa-compute-cuda-ext-dcub PROPERTIES OUTPUT_NAME "lc-cuda-dcub")
install(TARGETS luisa-compute-cuda-ext-dcub EXPORT LuisaComputeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

file(GLOB LCUB_SOURCES CONFIGURE_DEPENDS "*.cpp")
add_library(luisa-compute-cuda-ext-lcub SHARED ${LCUB_SOURCES})
target_link_libraries(luisa-compute-cuda-ext-lcub PUBLIC luisa-compute-runtime luisa-compute-cuda-ext-dcub)
target_compile_definitions(luisa-compute-cuda-ext-lcub PRIVATE LC_BACKEND_EXPORT_DLL=1)
set_target_properties(luisa-compute-cuda-ext-lcub PROPERTIES OUTPUT_NAME "lc-cuda-lcub")
install(TARGETS luisa-compute-cuda-ext-lcub EXPORT LuisaComputeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# add_compile_options(
#     $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#     $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
#     $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
#     $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>
# )
