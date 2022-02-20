set(SOURCEFILE
NoSqlPython.cpp
PythonLib.cpp
SerdeExtern.cpp
SimpleBinaryJson.cpp
SimpleJsonLoader.cpp
SimpleJsonValue.cpp
SimpleParser.cpp
config.h
DatabaseInclude.h
IJsonDatabase.h
IJsonObject.h
JsonNetServerRPC.h
NoSqlExample.h
Parser/StateRecorder.h
ParserException.h
PythonLib.h
SimpleBinaryJson.h
SimpleJsonLoader.h
SimpleJsonObject.h
SimpleJsonValue.h)

add_library(luisa-compute-serde MODULE ${SOURCEFILE})
target_link_libraries(luisa-compute-serde PRIVATE luisa-compute-runtime luisa-compute-ast)
set_target_properties(luisa-compute-serde PROPERTIES UNITY_BUILD ON)
