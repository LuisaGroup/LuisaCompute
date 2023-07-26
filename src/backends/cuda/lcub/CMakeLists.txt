find_package(CUDAToolkit REQUIRED)
enable_language(CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_library(cuda-dcub SHARED)
target_compile_definitions(cuda-dcub PRIVATE "-DDCUB_DLL_EXPORTS")
target_compile_features(cuda-dcub PRIVATE cxx_std_17)

file(GLOB_RECURSE DCUB_SOURCES "private/dcub/*.cu")
target_sources(cuda-dcub PRIVATE ${DCUB_SOURCES})

target_link_libraries(cuda-dcub PRIVATE CUDA::cudart)
target_include_directories(cuda-dcub PRIVATE "${PROJECT_SOURCE_DIR}/include") # -> LuisaCompute/include

