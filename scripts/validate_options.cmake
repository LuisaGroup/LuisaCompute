if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LuisaCompute only supports 64-bit platforms")
endif ()

if (LUISA_COMPUTE_BUILD_TESTS)
    if (NOT LUISA_COMPUTE_ENABLE_DSL)
        message(WARNING "DSL is required for tests. The DSL will be enabled.")
        set(LUISA_COMPUTE_ENABLE_DSL ON CACHE BOOL "Enable C++ DSL" FORCE)
    endif ()
endif ()

# check Rust support
if (NOT DEFINED CARGO_HOME)
    if ("$ENV{CARGO_HOME}" STREQUAL "")
        if (CMAKE_HOST_WIN32)
            set(CARGO_HOME "$ENV{USERPROFILE}/.cargo")
        else ()
            set(CARGO_HOME "$ENV{HOME}/.cargo")
        endif ()
    else ()
        set(CARGO_HOME "$ENV{CARGO_HOME}")
    endif ()
endif ()

find_program(CARGO_EXE cargo NO_CACHE HINTS "${CARGO_HOME}" PATH_SUFFIXES "bin")
if (CARGO_EXE)
    set(LUISA_COMPUTE_ENABLE_RUST ON)
else ()
    set(LUISA_COMPUTE_ENABLE_RUST OFF)
endif ()

if (NOT LUISA_COMPUTE_ENABLE_RUST)
    message(FATAL_ERROR "\nRust is required for future releases. \n\
    To install Rust, run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` on unix environment\n\
    or download and run the installer from https://static.rust-lang.org/rustup/dist/i686-pc-windows-gnu/rustup-init.exe on windows environment.\n\
    please set LUISA_COMPUTE_DISABLE_RUST_OVERRIDE to ON to acknowledge this")
endif ()


function(report_feature_not_available option_name feature_name)
    if (LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
        message(WARNING "The ${feature_name} is not available. The ${feature_name} will be disabled.")
        set(LUISA_COMPUTE_ENABLE_${option_name} OFF CACHE BOOL "Enable ${feature_name}" FORCE)
    else ()
        message(FATAL_ERROR "The ${feature_name} is not available. Please install the dependencies to enable the ${feature_name}.")
    endif ()
endfunction()

if (LUISA_COMPUTE_ENABLE_DX)
    if (NOT WIN32)
        report_feature_not_available(DX "DirectX backend")
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_METAL)
    if (NOT APPLE OR NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang" OR NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        report_feature_not_available(METAL "Metal backend")
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA)
    find_package(CUDAToolkit 11.7 QUIET)
    if (NOT CUDAToolkit_FOUND)
        report_feature_not_available(CUDA "CUDA backend")
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if (NOT Vulkan_FOUND)
        report_feature_not_available(VULKAN "Vulkan backend")
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE)
    if (NOT LUISA_COMPUTE_ENABLE_RUST)
        report_feature_not_available(CPU "CPU backend")
        report_feature_not_available(REMOTE "Remote backend")
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA)
    option(LUISA_COMPUTE_DOWNLOAD_NVCOMP "Download the nvCOMP library for CUDA GPU decompression" OFF)
endif ()

if (SKBUILD)
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET REQUIRED)
endif ()

if (LUISA_COMPUTE_ENABLE_GUI)
    # currently nothing to check
endif ()
