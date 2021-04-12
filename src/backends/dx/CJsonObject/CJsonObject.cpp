#include <CJsonObject/CJsonObject.hpp>
namespace neb {
CJsonObject::CJsonObject()
	: m_pJsonData(NULL), m_pExternJsonDataRef(NULL), m_pKeyTravers(NULL) {
	// m_pJsonData = cJSON_CreateObject();
}
CJsonObject::CJsonObject(const vengine::string& strJson)
	: m_pJsonData(NULL), m_pExternJsonDataRef(NULL), m_pKeyTravers(NULL) {
	Parse(strJson);
}
CJsonObject::CJsonObject(const CJsonObject* pJsonObject)
	: m_pJsonData(NULL), m_pExternJsonDataRef(NULL), m_pKeyTravers(NULL) {
	if (pJsonObject) {
		Parse(pJsonObject->ToString());
	}
}
CJsonObject::CJsonObject(const CJsonObject& oJsonObject)
	: m_pJsonData(NULL), m_pExternJsonDataRef(NULL), m_pKeyTravers(NULL) {
	Parse(oJsonObject.ToString());
}
CJsonObject::~CJsonObject() {
	Clear();
}
CJsonObject& CJsonObject::operator=(const CJsonObject& oJsonObject) {
	Parse(oJsonObject.ToString().c_str());
	return (*this);
}
bool CJsonObject::operator==(const CJsonObject& oJsonObject) const {
	return (this->ToString() == oJsonObject.ToString());
}
bool CJsonObject::AddEmptySubObject(const vengine::string& strKey) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateObject();
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("create sub empty object error!"_sv);
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::AddEmptySubArray(const vengine::string& strKey) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateArray();
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("create sub empty array error!"_sv);
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::GetKey(vengine::string& strKey) {
	if (IsArray()) {
		return (false);
	}
	if (m_pKeyTravers == NULL) {
		if (m_pJsonData != NULL) {
			m_pKeyTravers = m_pJsonData;
		} else if (m_pExternJsonDataRef != NULL) {
			m_pKeyTravers = m_pExternJsonDataRef;
		}
		return (false);
	} else if (m_pKeyTravers == m_pJsonData || m_pKeyTravers == m_pExternJsonDataRef) {
		cJSON* c = m_pKeyTravers->child;
		if (c) {
			strKey = c->string;
			m_pKeyTravers = c->next;
			return (true);
		} else {
			return (false);
		}
	} else {
		strKey = m_pKeyTravers->string;
		m_pKeyTravers = m_pKeyTravers->next;
		return (true);
	}
}
void CJsonObject::ResetTraversing() {
	if (m_pJsonData != NULL) {
		m_pKeyTravers = m_pJsonData;
	} else {
		m_pKeyTravers = m_pExternJsonDataRef;
	}
}
CJsonObject& CJsonObject::operator[](const vengine::string& strKey) {
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (!iter) {
		cJSON* pJsonStruct = NULL;
		if (m_pJsonData != NULL) {
			if (m_pJsonData->type == cJSON_Object) {
				pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
			}
		} else if (m_pExternJsonDataRef != NULL) {
			if (m_pExternJsonDataRef->type == cJSON_Object) {
				pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
			}
		}
		if (pJsonStruct == NULL) {
			CJsonObject* pJsonObject = new CJsonObject();
			m_mapJsonObjectRef.Insert(strKey, pJsonObject);
			return (*pJsonObject);
		} else {
			CJsonObject* pJsonObject = new CJsonObject(pJsonStruct);
			m_mapJsonObjectRef.Insert(strKey, pJsonObject);
			return (*pJsonObject);
		}
	} else {
		return (*(iter.Value()));
	}
}
CJsonObject& CJsonObject::operator[](uint32_t uiWhich) {
	auto iter = m_mapJsonArrayRef.Find(uiWhich);
	if (!iter) {
		cJSON* pJsonStruct = NULL;
		if (m_pJsonData != NULL) {
			if (m_pJsonData->type == cJSON_Array) {
				pJsonStruct = cJSON_GetArrayItem(m_pJsonData, uiWhich);
			}
		} else if (m_pExternJsonDataRef != NULL) {
			if (m_pExternJsonDataRef->type == cJSON_Array) {
				pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, uiWhich);
			}
		}
		if (pJsonStruct == NULL) {
			CJsonObject* pJsonObject = new CJsonObject();
			m_mapJsonArrayRef.Insert(uiWhich, pJsonObject);
			return (*pJsonObject);
		} else {
			CJsonObject* pJsonObject = new CJsonObject(pJsonStruct);
			m_mapJsonArrayRef.Insert(uiWhich, pJsonObject);
			return (*pJsonObject);
		}
	} else {
		return (*(iter.Value()));
	}
}
vengine::string CJsonObject::operator()(const vengine::string& strKey) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (vengine::string(""_sv));
	}
	if (pJsonStruct->type == cJSON_String) {
		return (pJsonStruct->valuestring);
	} else if (pJsonStruct->type == cJSON_Int) {
		char szNumber[128] = {0};
		if (pJsonStruct->sign == -1) {
			if (pJsonStruct->valueint <= (int64)INT_MAX && (int64)pJsonStruct->valueint >= (int64)INT_MIN) {
				snprintf(szNumber, sizeof(szNumber), "%d", (int32)pJsonStruct->valueint);
			} else {
				snprintf(szNumber, sizeof(szNumber), "%lld", (int64)pJsonStruct->valueint);
			}
		} else {
			if ((uint64)pJsonStruct->valueint <= (uint64)UINT_MAX) {
				snprintf(szNumber, sizeof(szNumber), "%u", (uint32_t)pJsonStruct->valueint);
			} else {
				snprintf(szNumber, sizeof(szNumber), "%llu", pJsonStruct->valueint);
			}
		}
		return (vengine::string(szNumber));
	} else if (pJsonStruct->type == cJSON_Double) {
		char szNumber[128] = {0};
		if (fabs(pJsonStruct->valuedouble) < 1.0e-6 || fabs(pJsonStruct->valuedouble) > 1.0e9) {
			snprintf(szNumber, sizeof(szNumber), "%e", pJsonStruct->valuedouble);
		} else {
			snprintf(szNumber, sizeof(szNumber), "%f", pJsonStruct->valuedouble);
		}
	} else if (pJsonStruct->type == cJSON_False) {
		return (vengine::string("false"_sv));
	} else if (pJsonStruct->type == cJSON_True) {
		return (vengine::string("true"_sv));
	}
	return (vengine::string(""_sv));
}
vengine::string CJsonObject::operator()(uint32_t uiWhich) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, uiWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, uiWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (vengine::string(""_sv));
	}
	if (pJsonStruct->type == cJSON_String) {
		return (pJsonStruct->valuestring);
	} else if (pJsonStruct->type == cJSON_Int) {
		char szNumber[128] = {0};
		if (pJsonStruct->sign == -1) {
			if (pJsonStruct->valueint <= (int64)INT_MAX && (int64)pJsonStruct->valueint >= (int64)INT_MIN) {
				snprintf(szNumber, sizeof(szNumber), "%d", (int32)pJsonStruct->valueint);
			} else {
				snprintf(szNumber, sizeof(szNumber), "%lld", (int64)pJsonStruct->valueint);
			}
		} else {
			if ((uint64)pJsonStruct->valueint <= (uint64)UINT_MAX) {
				snprintf(szNumber, sizeof(szNumber), "%u", (uint32_t)pJsonStruct->valueint);
			} else {
				snprintf(szNumber, sizeof(szNumber), "%llu", pJsonStruct->valueint);
			}
		}
		return (vengine::string(szNumber));
	} else if (pJsonStruct->type == cJSON_Double) {
		char szNumber[128] = {0};
		if (fabs(pJsonStruct->valuedouble) < 1.0e-6 || fabs(pJsonStruct->valuedouble) > 1.0e9) {
			snprintf(szNumber, sizeof(szNumber), "%e", pJsonStruct->valuedouble);
		} else {
			snprintf(szNumber, sizeof(szNumber), "%f", pJsonStruct->valuedouble);
		}
	} else if (pJsonStruct->type == cJSON_False) {
		return (vengine::string("false"_sv));
	} else if (pJsonStruct->type == cJSON_True) {
		return (vengine::string("true"_sv));
	}
	return (vengine::string(""_sv));
}
bool CJsonObject::Parse(const vengine::string& strJson) {
	Clear();
	m_pJsonData = cJSON_Parse(strJson.c_str());
	m_pKeyTravers = m_pJsonData;
	if (m_pJsonData == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	return (true);
}
void CJsonObject::Clear() {
	m_pExternJsonDataRef = NULL;
	m_pKeyTravers = NULL;
	if (m_pJsonData != NULL) {
		cJSON_Delete(m_pJsonData);
		m_pJsonData = NULL;
	}
	m_mapJsonArrayRef.IterateAll([&](uint32_t const& key, CJsonObject*& value) -> void {
		if (value != nullptr) {
			delete (value);
			value = nullptr;
		}
	});
	m_mapJsonArrayRef.Clear();
	m_mapJsonObjectRef.IterateAll([&](vengine::string const& key, CJsonObject*& value) -> void {
		if (value != nullptr) {
			delete value;
			value = NULL;
		}
	});
	m_mapJsonObjectRef.Clear();
}
bool CJsonObject::IsEmpty() const {
	if (m_pJsonData != NULL) {
		return (false);
	} else if (m_pExternJsonDataRef != NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::IsArray() const {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	}
	if (pFocusData == NULL) {
		return (false);
	}
	if (pFocusData->type == cJSON_Array) {
		return (true);
	} else {
		return (false);
	}
}
vengine::string CJsonObject::ToString() const {
	char* pJsonString = NULL;
	vengine::string strJsonData = ""_sv;
	if (m_pJsonData != NULL) {
		pJsonString = cJSON_PrintUnformatted(m_pJsonData);
	} else if (m_pExternJsonDataRef != NULL) {
		pJsonString = cJSON_PrintUnformatted(m_pExternJsonDataRef);
	}
	if (pJsonString != NULL) {
		strJsonData = pJsonString;
		vengine_free(pJsonString);
	}
	return (strJsonData);
}
vengine::string CJsonObject::ToFormattedString() const {
	char* pJsonString = NULL;
	vengine::string strJsonData = ""_sv;
	if (m_pJsonData != NULL) {
		pJsonString = cJSON_Print(m_pJsonData);
	} else if (m_pExternJsonDataRef != NULL) {
		pJsonString = cJSON_Print(m_pExternJsonDataRef);
	}
	if (pJsonString != NULL) {
		strJsonData = pJsonString;
		vengine_free(pJsonString);
	}
	return (strJsonData);
}
bool CJsonObject::Get(const vengine::string& strKey, CJsonObject& oJsonObject) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	char* pJsonString = cJSON_Print(pJsonStruct);
	vengine::string strJsonData = pJsonString;
	vengine_free(pJsonString);
	if (oJsonObject.Parse(strJsonData)) {
		return (true);
	} else {
		return (false);
	}
}
cJSON* CJsonObject::Get(const vengine::string& strKey) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	return pJsonStruct;
}
bool CJsonObject::Get(const vengine::string& strKey, vengine::string& strValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type != cJSON_String) {
		return (false);
	}
	strValue = pJsonStruct->valuestring;
	return (true);
}
bool CJsonObject::Get(const vengine::string& strKey, int32& iValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		iValue = (int32)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		iValue = (int32)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(const vengine::string& strKey, uint32_t& uiValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		uiValue = (uint32_t)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		uiValue = (uint32_t)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(const vengine::string& strKey, int64& llValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		llValue = (int64)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		llValue = (int64)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(const vengine::string& strKey, uint64& ullValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		ullValue = (uint64)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		ullValue = (uint64)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(const vengine::string& strKey, bool& bValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type > cJSON_True) {
		return (false);
	}
	bValue = pJsonStruct->type;
	return (true);
}
bool CJsonObject::Get(const vengine::string& strKey, float& fValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Double || pJsonStruct->type == cJSON_Int) {
		fValue = (float)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(const vengine::string& strKey, double& dValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Double || pJsonStruct->type == cJSON_Int) {
		dValue = pJsonStruct->valuedouble;
		return (true);
	}
	return (false);
}
bool CJsonObject::IsNull(const vengine::string& strKey) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pJsonData, strKey.c_str());
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Object) {
			pJsonStruct = cJSON_GetObjectItem(m_pExternJsonDataRef, strKey.c_str());
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type != cJSON_NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, const CJsonObject& oJsonObject) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_Parse(oJsonObject.ToString().c_str());
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, const vengine::string& strValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateString(strValue.c_str());
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, int32 iValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)iValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, uint32_t uiValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)uiValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, int64 llValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)llValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, uint64 ullValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt(ullValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, bool bValue, bool bValueAgain) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateBool(bValue);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, float fValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)fValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Add(const vengine::string& strKey, double dValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)dValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::AddNull(const vengine::string& strKey) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateObject();
		m_pKeyTravers = m_pJsonData;
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) != NULL) {
		m_strErrMsg = "key exists!"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateNull();
	if (pJsonStruct == NULL) {
		return (false);
	}
	cJSON_AddItemToObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Delete(const vengine::string& strKey) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON_DeleteItemFromObject(pFocusData, strKey.c_str());
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	m_pKeyTravers = pFocusData;
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, const CJsonObject& oJsonObject) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_Parse(oJsonObject.ToString().c_str());
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, const vengine::string& strValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateString(strValue.c_str());
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, int32 iValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)iValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, uint32_t uiValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)uiValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, int64 llValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)llValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, uint64 ullValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)ullValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, bool bValue, bool bValueAgain) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateBool(bValue);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, float fValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)fValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(const vengine::string& strKey, double dValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)dValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::ReplaceWithNull(const vengine::string& strKey) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Object) {
		m_strErrMsg = "not a json object! json array?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateNull();
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonObjectRef.Find(strKey);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonObjectRef.Remove(iter);
	}
	cJSON_ReplaceItemInObject(pFocusData, strKey.c_str(), pJsonStruct);
	if (cJSON_GetObjectItem(pFocusData, strKey.c_str()) == NULL) {
		return (false);
	}
	return (true);
}
int32_t CJsonObject::GetArraySize() {
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			return (cJSON_GetArraySize(m_pJsonData));
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			return (cJSON_GetArraySize(m_pExternJsonDataRef));
		}
	}
	return (0);
}
bool CJsonObject::Get(int32_t iWhich, CJsonObject& oJsonObject) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	char* pJsonString = cJSON_Print(pJsonStruct);
	vengine::string strJsonData = pJsonString;
	vengine_free(pJsonString);
	if (oJsonObject.Parse(strJsonData)) {
		return (true);
	} else {
		return (false);
	}
}

cJSON* CJsonObject::Get(int32_t iWhich) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	return pJsonStruct;
}
bool CJsonObject::Get(int32_t iWhich, vengine::string& strValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type != cJSON_String) {
		return (false);
	}
	strValue = pJsonStruct->valuestring;
	return (true);
}
bool CJsonObject::Get(int32_t iWhich, int32& iValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		iValue = (int32)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		iValue = (int32)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(int32_t iWhich, uint32_t& uiValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		uiValue = (uint32_t)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		uiValue = (uint32_t)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(int32_t iWhich, int64& llValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		llValue = (int64)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		llValue = (int64)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(int32_t iWhich, uint64& ullValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Int) {
		ullValue = (uint64)(pJsonStruct->valueint);
		return (true);
	} else if (pJsonStruct->type == cJSON_Double) {
		ullValue = (uint64)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(int32_t iWhich, bool& bValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type > cJSON_True) {
		return (false);
	}
	bValue = pJsonStruct->type;
	return (true);
}
bool CJsonObject::Get(int32_t iWhich, float& fValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Double || pJsonStruct->type == cJSON_Int) {
		fValue = (float)(pJsonStruct->valuedouble);
		return (true);
	}
	return (false);
}
bool CJsonObject::Get(int32_t iWhich, double& dValue) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type == cJSON_Double || pJsonStruct->type == cJSON_Int) {
		dValue = pJsonStruct->valuedouble;
		return (true);
	}
	return (false);
}
bool CJsonObject::IsNull(int32_t iWhich) const {
	cJSON* pJsonStruct = NULL;
	if (m_pJsonData != NULL) {
		if (m_pJsonData->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pJsonData, iWhich);
		}
	} else if (m_pExternJsonDataRef != NULL) {
		if (m_pExternJsonDataRef->type == cJSON_Array) {
			pJsonStruct = cJSON_GetArrayItem(m_pExternJsonDataRef, iWhich);
		}
	}
	if (pJsonStruct == NULL) {
		return (false);
	}
	if (pJsonStruct->type != cJSON_NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(const CJsonObject& oJsonObject) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_Parse(oJsonObject.ToString().c_str());
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	uint32_t uiLastIndex = (uint32_t)cJSON_GetArraySize(pFocusData) - 1;
	deleteKeys.reserve(10);
	m_mapJsonArrayRef.IterateAll([&](uint32_t key, CJsonObject*& value) -> void {
		if (key >= uiLastIndex) {
			if (value != NULL) {
				delete value;
				value = NULL;
			}
			deleteKeys.push_back(key);
		}
	});
	for (auto ite = deleteKeys.begin(); ite != deleteKeys.end(); ++ite) {
		m_mapJsonArrayRef.Remove(*ite);
	}
	deleteKeys.clear();
	return (true);
}
bool CJsonObject::Add(const vengine::string& strValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateString(strValue.c_str());
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(int32 iValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)iValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(uint32_t uiValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)uiValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(int64 llValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)llValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(uint64 ullValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)ullValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(int32_t iAnywhere, bool bValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateBool(bValue);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(float fValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)fValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Add(double dValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)dValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddNull() {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateNull();
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArray(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(const CJsonObject& oJsonObject) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_Parse(oJsonObject.ToString().c_str());
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	deleteKeys.reserve(10);
	m_mapJsonArrayRef.IterateAll([&](uint32_t key, CJsonObject*& value) -> void {
		if (value != NULL) {
			delete value;
			value = NULL;
		}
		deleteKeys.push_back(key);
	});
	for (auto ite = deleteKeys.begin(); ite != deleteKeys.end(); ++ite) {
		m_mapJsonArrayRef.Remove(*ite);
	}
	deleteKeys.clear();
	return (true);
}
bool CJsonObject::AddAsFirst(const vengine::string& strValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateString(strValue.c_str());
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(int32 iValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)iValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(uint32_t uiValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)uiValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(int64 llValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)llValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(uint64 ullValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)ullValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(int32_t iAnywhere, bool bValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateBool(bValue);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(float fValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)fValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddAsFirst(double dValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)dValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::AddNullAsFirst() {
	cJSON* pFocusData = NULL;
	if (m_pJsonData != NULL) {
		pFocusData = m_pJsonData;
	} else if (m_pExternJsonDataRef != NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		m_pJsonData = cJSON_CreateArray();
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateNull();
	if (pJsonStruct == NULL) {
		return (false);
	}
	int32_t iArraySizeBeforeAdd = cJSON_GetArraySize(pFocusData);
	cJSON_AddItemToArrayHead(pFocusData, pJsonStruct);
	int32_t iArraySizeAfterAdd = cJSON_GetArraySize(pFocusData);
	if (iArraySizeAfterAdd == iArraySizeBeforeAdd) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Delete(int32_t iWhich) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON_DeleteItemFromArray(pFocusData, iWhich);
	deleteKeys.reserve(10);
	m_mapJsonArrayRef.IterateAll([&](uint32_t key, CJsonObject*& value) -> void {
		if (key >= (uint32_t)iWhich) {
			if (value != NULL) {
				delete value;
				value = NULL;
			}
			deleteKeys.push_back(key);
		}
	});
	for (auto ite = deleteKeys.begin(); ite != deleteKeys.end(); ++ite) {
		m_mapJsonArrayRef.Remove(*ite);
	}
	deleteKeys.clear();
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, const CJsonObject& oJsonObject) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_Parse(oJsonObject.ToString().c_str());
	if (pJsonStruct == NULL) {
		m_strErrMsg = vengine::string("prase json string error at "_sv) + cJSON_GetErrorPtr();
		return (false);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, const vengine::string& strValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateString(strValue.c_str());
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, int32 iValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)iValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, uint32_t uiValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)uiValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, int64 llValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)((uint64)llValue), -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, uint64 ullValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateInt((uint64)ullValue, 1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, bool bValue, bool bValueAgain) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateBool(bValue);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, float fValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)fValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::Replace(int32_t iWhich, double dValue) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateDouble((double)dValue, -1);
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
bool CJsonObject::ReplaceWithNull(int32_t iWhich) {
	cJSON* pFocusData = NULL;
	if (m_pJsonData == NULL) {
		pFocusData = m_pExternJsonDataRef;
	} else {
		pFocusData = m_pJsonData;
	}
	if (pFocusData == NULL) {
		m_strErrMsg = "json data is null!"_sv;
		return (false);
	}
	if (pFocusData->type != cJSON_Array) {
		m_strErrMsg = "not a json array! json object?"_sv;
		return (false);
	}
	cJSON* pJsonStruct = cJSON_CreateNull();
	if (pJsonStruct == NULL) {
		return (false);
	}
	auto iter = m_mapJsonArrayRef.Find(iWhich);
	if (iter) {
		if (iter.Value() != NULL) {
			delete (iter.Value());
			iter.Value() = NULL;
		}
		m_mapJsonArrayRef.Remove(iter);
	}
	cJSON_ReplaceItemInArray(pFocusData, iWhich, pJsonStruct);
	if (cJSON_GetArrayItem(pFocusData, iWhich) == NULL) {
		return (false);
	}
	return (true);
}
CJsonObject::CJsonObject(cJSON* pJsonData)
	: m_pJsonData(NULL), m_pExternJsonDataRef(pJsonData), m_pKeyTravers(pJsonData) {
}
}// namespace neb
neb::CJsonObject* ReadJson(const vengine::string& filePath) {
	std::ifstream ifs(filePath.data());
	if (!ifs) return nullptr;
	ifs.seekg(0, std::ios::end);
	size_t len = ifs.tellg();
	ifs.seekg(0, 0);
	vengine::string s(len + 1, ' ');
	ifs.read((char*)s.data(), len);
	neb::CJsonObject* jsonObj = new neb::CJsonObject(s);
	if (jsonObj->IsEmpty()) {
		delete jsonObj;
		return nullptr;
	}
	return jsonObj;
}
float GetFloatFromChar(char* c, size_t t) {
	if (t == 0) return 0;
	float num = 0;
	float pointer = 0;
	float rate = 1;
	float type = 1;
	size_t i = 0;
	if (c[i] == '-') {
		type = -1;
		i++;
	}
	for (; i < t; ++i) {
		if (c[i] == '.') {
			++i;
			break;
		}
		char n = c[i] - 48;
		num *= 10;
		num += n;
	}
	for (; i < t; ++i) {
		rate *= 0.1f;
		pointer += (c[i] - 48) * rate;
	}
	return (pointer + num) * type;
}
int32_t GetIntFromChar(char* c, size_t t) {
	if (t == 0) return 0;
	int32_t num = 0;
	int32_t type = 1;
	size_t i = 0;
	if (c[i] == '-') {
		type = -1;
		i++;
	}
	for (; i < t; ++i) {
		if (c[i] == '.') {
			++i;
			break;
		}
		char n = c[i] - 48;
		num *= 10;
		num += n;
	}
	return num * type;
}
//#endif
double GetDoubleFromChar(char* c, size_t t) {
	if (t == 0) return 0;
	double num = 0;
	double pointer = 0;
	double rate = 1;
	double type = 1;
	size_t i = 0;
	if (c[i] == '-') {
		type = -1;
		i++;
	}
	for (; i < t; ++i) {
		if (c[i] == '.') {
			++i;
			break;
		}
		char n = c[i] - 48;
		num *= 10;
		num += n;
	}
	for (; i < t; ++i) {
		rate *= 0.1;
		pointer += (c[i] - 48) * rate;
	}
	return (pointer + num) * type;
}
/*
void ReadJson(const vengine::string& filePath, StackObject<neb::CJsonObject, true>& placementPtr)
{
	std::ifstream ifs(filePath.data());
	if (!ifs) return;
	ifs.seekg(0, std::ios::end);
	size_t len = ifs.tellg();
	ifs.seekg(0, 0);
	vengine::string s(len + 1, ' ');
	ifs.read((char*)s.data(), len);
	placementPtr.New();
	if (!placementPtr->Parse(s))
	{
		placementPtr.Delete();
	}
}*/
