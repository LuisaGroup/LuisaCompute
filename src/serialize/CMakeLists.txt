set(LUISA_COMPUTE_SERIALIZE_SOURCES
    Common.h
        DatabaseInclude.h
        IJsonDatabase.h
        IJsonObject.h
        serializer.h
        SimpleBinaryJson.cpp SimpleBinaryJson.h
        SimpleJsonLoader.cpp SimpleJsonLoader.h
        SimpleJsonObject.h
        SimpleJsonValue.cpp SimpleJsonValue.h
        SimpleParser.cpp SimpleParser.h)

add_library(luisa-compute-serialize SHARED ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize PUBLIC luisa-compute-vstl)
set_target_properties(luisa-compute-serialize PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
