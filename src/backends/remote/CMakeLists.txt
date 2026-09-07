include(FetchContent)
find_package(Threads REQUIRED)

FetchContent_Declare(luisa_compute_asio
        URL https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-38-2.tar.gz
        URL_HASH SHA256=9f2648fa483e58a6bf848d970ee0ea650ca19ed7769dfa520ed4f7b8d27af1db
        DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(luisa_compute_asio)

add_library(luisa-compute-ext-asio INTERFACE)
target_include_directories(luisa-compute-ext-asio SYSTEM INTERFACE
        ${luisa_compute_asio_SOURCE_DIR}/include)
target_compile_definitions(luisa-compute-ext-asio INTERFACE
        ASIO_STANDALONE=1
        ASIO_NO_DEPRECATED=1)
target_link_libraries(luisa-compute-ext-asio INTERFACE Threads::Threads)

set(LUISA_COMPUTE_REMOTE_COMMON_SOURCES
        remote_blob_cache.cpp
        remote_blob_cache.h
        remote_protocol.cpp
        remote_protocol.h
        remote_transport.cpp
        remote_transport.h)

add_library(luisa-compute-remote-common STATIC
        ${LUISA_COMPUTE_REMOTE_COMMON_SOURCES})
target_link_libraries(luisa-compute-remote-common
        PUBLIC luisa-compute-core
        PRIVATE luisa-compute-ext-asio)
target_include_directories(luisa-compute-remote-common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(luisa-compute-remote-common PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        FOLDER backends/remote)

add_library(luisa-compute-remote-runtime STATIC
        remote_command_codec.cpp
        remote_command_codec.h
        remote_server.cpp
        remote_server.h)
target_link_libraries(luisa-compute-remote-runtime
        PUBLIC
        luisa-compute-remote-common
        luisa-compute-ast
        luisa-compute-runtime
        PRIVATE
        luisa-compute-ext-asio)
target_include_directories(luisa-compute-remote-runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(luisa-compute-remote-runtime PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        FOLDER backends/remote)

luisa_compute_add_backend(remote SOURCES
        remote_device.cpp
        remote_device.h)
target_link_libraries(luisa-compute-backend-remote PRIVATE
        luisa-compute-remote-runtime)

add_executable(luisa-remote-server
        remote_server_main.cpp)
target_link_libraries(luisa-remote-server PRIVATE
        luisa-compute-remote-runtime
        luisa-compute-runtime
        luisa-compute-ext-asio)
set_target_properties(luisa-remote-server PROPERTIES
        OUTPUT_NAME luisa-remote-server
        FOLDER backends/remote)
install(TARGETS luisa-remote-server
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
