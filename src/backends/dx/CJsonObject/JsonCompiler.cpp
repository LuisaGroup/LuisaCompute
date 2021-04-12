#include <CJsonObject/JsonCompiler.h>
void JsonCompiler::Serialize(neb::CJsonObject& jsonObj, vengine::vector<char>& data) {
	using namespace SerializeStruct;
	auto serializecJSon = [&data](cJSON* ptr) -> void {
		switch (ptr->type) {
			case cJSON_False:
				PushBackData(data, ObjectType::False);
				break;
			case cJSON_True:
				PushBackData(data, ObjectType::True);
				break;
			case cJSON_NULL:
				PushBackData(data, ObjectType::Null);
				break;
			case cJSON_Int:
				PushBackData(data, ObjectType::Int);
				PushBackData(data, ptr->valueint);
				break;
			case cJSON_Double:
				PushBackData(data, ObjectType::Float);
				PushBackData(data, ptr->valuedouble);
				break;
			case cJSON_String: {
				PushBackData(data, ObjectType::String);
				StringHeader header;
				header.stringLength = strlen(ptr->valuestring);
				PushBackData(data, header);
				PushBackAll<char>(data, ptr->valuestring, header.stringLength);
			} break;
			case cJSON_Array:
			case cJSON_Object: {
				char* pJsonString = cJSON_Print(ptr);
				vengine::string strJsonData = pJsonString;
				vengine_free(pJsonString);
				neb::CJsonObject obj;
				if (obj.Parse(strJsonData)) {
					Serialize(obj, data);
				}
			} break;
		}
	};
	if (jsonObj.IsArray()) {
		PushBackData(data, ObjectType::JsonArray);
		uint64 arraySize = jsonObj.GetArraySize();
		PushBackData(data, arraySize);
		for (size_t i = 0; i < arraySize; ++i) {
			cJSON* ptr = jsonObj.Get(i);
			serializecJSon(ptr);
		}
	} else {
		PushBackData(data, ObjectType::JsonObject);
		uint64 jsonObjSizeIndex = data.size();
		data.resize(data.size() + sizeof(uint64));
		jsonObj.ResetTraversing();
		vengine::string keyStr;
		uint64 count = 0;
		while (jsonObj.GetKey(keyStr)) {
			PushBackData(data, ObjectType::String);
			PushBackData(data, keyStr);
			cJSON* ptr = jsonObj.Get(keyStr);
			serializecJSon(ptr);
			count++;
		}
		*(uint64*)(data.data() + jsonObjSizeIndex) = count;
	}
}
