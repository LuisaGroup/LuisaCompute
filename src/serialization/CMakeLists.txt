set(LUISA_COMPUTE_SERIALIZATION_SOURCES
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

add_library(luisa-compute-serialization SHARED ${LUISA_COMPUTE_SERIALIZATION_SOURCES})
target_link_libraries(luisa-compute-serialization PUBLIC luisa-compute-util)
set_target_properties(luisa-compute-serialization PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
