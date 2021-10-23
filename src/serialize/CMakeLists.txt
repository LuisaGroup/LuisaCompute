set(LUISA_COMPUTE_SERIALIZE_SOURCES
        IJsonObject.h
        IJsonDatabase.h
        INoSqlDatabase.h
        serialize.cpp)

add_library(luisa-compute-serialize SHARED ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize PUBLIC luisa-compute-vstl)
set_target_properties(luisa-compute-serialize PROPERTIES
        UNITY_BUILD ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
