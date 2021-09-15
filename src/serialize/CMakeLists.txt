set(LUISA_COMPUTE_SERIALIZE_SOURCES
        Common.h
        Config.h
        IJsonDatabase.h
        IJsonObject.h
        interface.h
        PythonLib.cpp PythonLib.h
        SerdeExtern.cpp
        serializer.h
        SimpleBinaryJson.cpp SimpleBinaryJson.h
        SimpleJsonLoader.cpp SimpleJsonLoader.h
        SimpleJsonObject.h
        SimpleJsonValue.cpp SimpleJsonValue.h
        SimpleParser.cpp SimpleParser.h)

add_library(luisa-compute-serialize-json MODULE ${LUISA_COMPUTE_SERIALIZE_SOURCES})
target_link_libraries(luisa-compute-serialize-json PRIVATE luisa-compute-vstl)
set_target_properties(luisa-compute-serialize-json PROPERTIES UNITY_BUILD ON)

add_library(luisa-compute-serialize INTERFACE)
add_dependencies(luisa-compute-serialize luisa-compute-serialize-json)
