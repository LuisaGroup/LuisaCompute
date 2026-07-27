option(LUISA_COMPUTE_DOWNLOAD_LLVM "Download and patch LLVM if not found" OFF)

if (LUISA_COMPUTE_DOWNLOAD_LLVM)
    set(_luisa_download_llvm_had_msvc_runtime_library FALSE)
    if (DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
        set(_luisa_download_llvm_had_msvc_runtime_library TRUE)
        set(_luisa_download_llvm_project_msvc_runtime_library
                "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    endif ()
    set(LLVM_DOWNLOAD_VERSION "19.1.5")
    set(LLVM_WIN_BINARY_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_DOWNLOAD_VERSION}/clang+llvm-${LLVM_DOWNLOAD_VERSION}-x86_64-pc-windows-msvc.tar.xz")
    message(STATUS "LLVM not found. Downloading official prebuilt binaries ${LLVM_DOWNLOAD_VERSION} from ${LLVM_WIN_BINARY_URL}.")
    include(FetchContent)
    FetchContent_Declare(
            llvm
            URL ${LLVM_WIN_BINARY_URL}
    )
    FetchContent_MakeAvailable(llvm)

    set(LLVM_DIR ${llvm_SOURCE_DIR}/lib/cmake/llvm CACHE PATH "Path to LLVMConfig.cmake" FORCE)
    find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR})
    if (_luisa_download_llvm_had_msvc_runtime_library)
        set(CMAKE_MSVC_RUNTIME_LIBRARY
                "${_luisa_download_llvm_project_msvc_runtime_library}")
    else ()
        unset(CMAKE_MSVC_RUNTIME_LIBRARY)
    endif ()
else ()
    message(WARNING "LLVM not found. Please either set `LLVM_DIR` to the directory containing 'LLVMConfig.cmake' or set `LUISA_COMPUTE_DOWNLOAD_LLVM=ON` to let LuisaCompute download it for you.")
endif ()
