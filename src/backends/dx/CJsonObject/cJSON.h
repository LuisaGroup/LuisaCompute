/*
 Copyright (c) 2009 Dave Gamble

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef cJSON__h
#define cJSON__h

#include <stdint.h>

typedef int32_t int32;
#ifndef _WIN32
#if __WORDSIZE == 64
typedef long int64;
typedef unsigned long uint64;
#endif
#else
typedef long long int64;
typedef unsigned long long uint64;
#endif


/* cJSON Types: */
#define cJSON_False 0
#define cJSON_True 1
#define cJSON_NULL 2
#define cJSON_Int 3
#define cJSON_Double 4
#define cJSON_String 5
#define cJSON_Array 6
#define cJSON_Object 7

#define cJSON_IsReference 256

/* The cJSON structure: */
typedef struct cJSON
{
    struct cJSON* next, * prev; /* next/prev allow you to walk array/object chains. Alternatively, use GetArraySize/GetArrayItem/GetObjectItem */
    struct cJSON* child; /* An array or object item will have a child pointer pointing to a chain of the items in the array/object. */

    int32_t type; /* The type of the item, as above. */

    char* valuestring; /* The item's string, if type==cJSON_String */
    int64 valueint; /* The item's number, if type==cJSON_Number */
    double valuedouble; /* The item's number, if type==cJSON_Number */
    int32_t sign;   /* sign of valueint, 1(unsigned), -1(signed) */

    char* string; /* The item's name string, if this item is the child of, or is in the list of subitems of an object. */
} cJSON;

typedef struct cJSON_Hooks
{
    void* (*malloc_fn)(size_t sz);
    void(*free_fn)(void* ptr);
} cJSON_Hooks;

/* Supply malloc, realloc and free functions to cJSON */
void cJSON_InitHooks(cJSON_Hooks* hooks);

/* Supply a block of JSON, and this returns a cJSON object you can interrogate. Call cJSON_Delete when finished. */
cJSON* cJSON_Parse(const char* value);
/* Render a cJSON entity to text for transfer/storage. Free the char* when finished. */
char* cJSON_Print(cJSON* item);
/* Render a cJSON entity to text for transfer/storage without any formatting. Free the char* when finished. */
char* cJSON_PrintUnformatted(cJSON* item);
/* Delete a cJSON entity and all subentities. */
void cJSON_Delete(cJSON* c);

/* Returns the number of items in an array (or object). */
int32_t cJSON_GetArraySize(cJSON* array);
/* Retrieve item number "item" from array "array". Returns NULL if unsuccessful. */
cJSON* cJSON_GetArrayItem(cJSON* array, int32_t item);
/* Get item "string" from object. Case insensitive. */
cJSON* cJSON_GetObjectItem(cJSON* object, const char* string);

/* For analysing failed parses. This returns a pointer to the parse error. You'll probably need to look a few chars back to make sense of it. Defined when cJSON_Parse() returns 0. 0 when cJSON_Parse() succeeds. */
const char* cJSON_GetErrorPtr();

/* These calls create a cJSON item of the appropriate type. */
cJSON* cJSON_CreateNull();
cJSON* cJSON_CreateTrue();
cJSON* cJSON_CreateFalse();
cJSON* cJSON_CreateBool(int32_t b);
cJSON* cJSON_CreateDouble(double num, int32_t sign);
cJSON* cJSON_CreateInt(uint64 num, int32_t sign);
cJSON* cJSON_CreateString(const char* string);
cJSON* cJSON_CreateArray();
cJSON* cJSON_CreateObject();

/* These utilities create an Array of count items. */
cJSON* cJSON_CreateIntArray(int32_t* numbers, int32_t sign, int32_t count);
cJSON* cJSON_CreateFloatArray(float* numbers, int32_t count);
cJSON* cJSON_CreateDoubleArray(double* numbers, int32_t count);
cJSON* cJSON_CreateStringArray(const char** strings, int32_t count);

/* Append item to the specified array/object. */
void cJSON_AddItemToArray(cJSON* array, cJSON* item);
void cJSON_AddItemToArrayHead(cJSON* array, cJSON* item);    /* add by Bwar on 2015-01-28 */
void cJSON_AddItemToObject(cJSON* object, const char* string,
    cJSON* item);
/* Append reference to item to the specified array/object. Use this when you want to add an existing cJSON to a new cJSON, but don't want to corrupt your existing cJSON. */
void cJSON_AddItemReferenceToArray(cJSON* array, cJSON* item);
void cJSON_AddItemReferenceToObject(cJSON* object, const char* string,
    cJSON* item);

/* Remove/Detatch items from Arrays/Objects. */
cJSON* cJSON_DetachItemFromArray(cJSON* array, int32_t which);
void cJSON_DeleteItemFromArray(cJSON* array, int32_t which);
cJSON* cJSON_DetachItemFromObject(cJSON* object, const char* string);
void cJSON_DeleteItemFromObject(cJSON* object, const char* string);

/* Update array items. */
void cJSON_ReplaceItemInArray(cJSON* array, int32_t which, cJSON* newitem);
void cJSON_ReplaceItemInObject(cJSON* object, const char* string,
    cJSON* newitem);

#define cJSON_AddNullToObject(object,name)	cJSON_AddItemToObject(object, name, cJSON_CreateNull())
#define cJSON_AddTrueToObject(object,name)	cJSON_AddItemToObject(object, name, cJSON_CreateTrue())
#define cJSON_AddFalseToObject(object,name)		cJSON_AddItemToObject(object, name, cJSON_CreateFalse())
#define cJSON_AddNumberToObject(object,name,n)	cJSON_AddItemToObject(object, name, cJSON_CreateNumber(n))
#define cJSON_AddStringToObject(object,name,s)	cJSON_AddItemToObject(object, name, cJSON_CreateString(s))


#endif
