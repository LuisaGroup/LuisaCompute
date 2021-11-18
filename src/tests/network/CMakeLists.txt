if (OpenCV_FOUND)
    set(LUISA_COMPUTE_NETWORK_SOURCES
            binary_buffer.h
            render_buffer.cpp render_buffer.h
            render_scheduler.cpp render_scheduler.h
            render_tile.h
            render_worker_session.cpp render_worker_session.h
            render_server.cpp render_server.h
            render_config.cpp render_config.h)
    add_library(luisa-compute-network SHARED ${LUISA_COMPUTE_NETWORK_SOURCES})
    target_link_libraries(luisa-compute-network PUBLIC luisa::compute ${OpenCV_LIBS})
    target_include_directories(luisa-compute-network PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_target_properties(luisa-compute-network PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            UNITY_BUILD ON)
    add_executable(test_network_master test_network_master.cpp)
    target_link_libraries(test_network_master PRIVATE luisa-compute-network)
    add_executable(test_network_worker test_network_worker.cpp)
    target_link_libraries(test_network_worker PRIVATE luisa-compute-network)
    add_executable(test_network_client test_network_client.cpp)
    target_link_libraries(test_network_client PRIVATE luisa-compute-network)
endif ()
