---
name: yyjson
description: yyjson C library for high-performance JSON parsing, creation, and manipulation. Use when reading, writing, querying, or modifying JSON in C/C++ code via yyjson.
---

# yyjson Usage Guide

High performance JSON library written in ANSI C (single `yyjson.h` + `yyjson.c`).

## Data Model

| Immutable | Mutable |
|-----------|---------|
| `yyjson_doc` | `yyjson_mut_doc` |
| `yyjson_val` | `yyjson_mut_val` |

- Parsed docs are immutable. To modify, copy to mutable with `yyjson_doc_mut_copy()`.
- All values/strings in a doc are held by the doc; free the doc to free everything.
- Arrays/objects are linked lists; index/key lookup is linear time. Prefer iterators.

## Read JSON

```c
// From string
yyjson_doc *doc = yyjson_read(json, len, 0);

// From file
yyjson_read_err err;
yyjson_doc *doc = yyjson_read_file("path", 0, NULL, &err);

// From string with options
yyjson_doc *doc = yyjson_read_opts((char *)dat, len, flg, alc, &err);

// From FILE*
yyjson_doc *doc = yyjson_read_fp(fp, flg, alc, &err);

// Incremental (for large docs)
yyjson_incr_state *state = yyjson_incr_new(buf, len, flg, alc);
yyjson_doc *doc = yyjson_incr_read(state, read_len, &err);
yyjson_incr_free(state);
```

**Common read flags** (combine with `|`):
- `YYJSON_READ_NOFLAG` (0) — RFC 8259 strict
- `YYJSON_READ_ALLOW_COMMENTS`
- `YYJSON_READ_ALLOW_TRAILING_COMMAS`
- `YYJSON_READ_ALLOW_INF_AND_NAN`
- `YYJSON_READ_ALLOW_INVALID_UNICODE`
- `YYJSON_READ_NUMBER_AS_RAW`
- `YYJSON_READ_BIGNUM_AS_RAW`
- `YYJSON_READ_INSITU` — modify input buffer in place; pad with `YYJSON_PADDING_SIZE` zeros
- `YYJSON_READ_STOP_WHEN_DONE` — stop at first doc (useful for NDJSON)
- `YYJSON_READ_JSON5` — enable JSON5 + extensions

**Error struct:**
```c
yyjson_read_err err;
// err.code, err.msg, err.pos
// Use yyjson_locate_pos(dat, len, err.pos, &line, &col, &chr) for line/col.
```

## Access Values

```c
yyjson_val *root = yyjson_doc_get_root(doc);

// Type checks
yyjson_is_null(val), yyjson_is_bool(val), yyjson_is_uint(val), yyjson_is_sint(val)
yyjson_is_int(val), yyjson_is_real(val), yyjson_is_num(val), yyjson_is_str(val)
yyjson_is_arr(val), yyjson_is_obj(val), yyjson_is_raw(val)

// Getters
bool        yyjson_get_bool(val);
uint64_t    yyjson_get_uint(val);
int64_t     yyjson_get_sint(val);
int         yyjson_get_int(val);
double      yyjson_get_real(val);
double      yyjson_get_num(val);
const char *yyjson_get_str(val);
size_t      yyjson_get_len(val);   // string len, array size, object size
const char *yyjson_get_raw(val);

// Array access (linear by index)
size_t      yyjson_arr_size(arr);
yyjson_val *yyjson_arr_get(arr, idx);
yyjson_val *yyjson_arr_get_first(arr);
yyjson_val *yyjson_arr_get_last(arr);

// Object access (linear by key)
size_t      yyjson_obj_size(obj);
yyjson_val *yyjson_obj_get(obj, "key");
yyjson_val *yyjson_obj_getn(obj, "key", key_len);

// String comparison
bool yyjson_equals_str(val, "abc");
bool yyjson_equals_strn(val, "abc", 3);
```

## Iterate Arrays/Objects

```c
// Immutable array
yyjson_val *v;
yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
while ((v = yyjson_arr_iter_next(&iter))) { ... }

// Immutable array foreach macro
size_t idx, max;
yyjson_val *v;
yyjson_arr_foreach(arr, idx, max, v) { ... }

// Immutable object
yyjson_val *k, *v;
yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
while ((k = yyjson_obj_iter_next(&iter))) {
    v = yyjson_obj_iter_get_val(k);
    ...
}

// Immutable object foreach macro
size_t idx, max;
yyjson_val *k, *v;
yyjson_obj_foreach(obj, idx, max, k, v) { ... }
```

Mutable variants use `yyjson_mut_` prefix: `yyjson_mut_arr_iter_with`, `yyjson_mut_arr_foreach`, `yyjson_mut_obj_iter_with`, `yyjson_mut_obj_foreach`.

Inside mutable iteration you can remove:
```c
yyjson_mut_arr_iter_remove(&iter);
yyjson_mut_obj_iter_remove(&iter);
```

## Create / Modify JSON

```c
// Create mutable doc
yyjson_mut_doc *doc = yyjson_mut_doc_new(alc);
yyjson_mut_val *root = yyjson_mut_obj(doc);
yyjson_mut_doc_set_root(doc, root);

// Create scalar values
yyjson_mut_val *v;
v = yyjson_mut_null(doc);
v = yyjson_mut_bool(doc, true);
v = yyjson_mut_uint(doc, 123);
v = yyjson_mut_sint(doc, -123);
v = yyjson_mut_int(doc, 123);
v = yyjson_mut_real(doc, 1.5);
v = yyjson_mut_str(doc, "hello");     // references caller's string
v = yyjson_mut_strn(doc, "hello", 5); // references caller's string
v = yyjson_mut_strcpy(doc, "hello");  // copies string into doc
v = yyjson_mut_strncpy(doc, "hello", 5);

// Array creation and modification
yyjson_mut_val *arr = yyjson_mut_arr(doc);
yyjson_mut_arr_append(arr, val);
yyjson_mut_arr_prepend(arr, val);
yyjson_mut_arr_insert(arr, val, idx);    // linear
yyjson_mut_arr_replace(arr, idx, val);   // linear, returns old
yyjson_mut_arr_remove(arr, idx);         // linear
yyjson_mut_arr_remove_first(arr);
yyjson_mut_arr_remove_last(arr);
yyjson_mut_arr_clear(arr);

// Array convenience adds
yyjson_mut_arr_add_int(doc, arr, 42);
yyjson_mut_arr_add_str(doc, arr, "x");
yyjson_mut_arr_add_arr(doc, arr);   // returns new sub-array
yyjson_mut_arr_add_obj(doc, arr);   // returns new sub-object

// Create array from C array
int nums[] = {1, 2, 3};
yyjson_mut_val *arr = yyjson_mut_arr_with_sint32(doc, nums, 3);

// Object creation and modification
yyjson_mut_val *obj = yyjson_mut_obj(doc);
yyjson_mut_obj_add(obj, key_val, val);    // allows duplicates
yyjson_mut_obj_put(obj, key_val, val);    // removes existing keys first (linear)
yyjson_mut_obj_remove(obj, key_val);
yyjson_mut_obj_clear(obj);

// Object convenience adds
yyjson_mut_obj_add_str(doc, obj, "name", "Mash");
yyjson_mut_obj_add_int(doc, obj, "star", 4);
yyjson_mut_obj_add_arr(doc, obj, "hits");   // returns new sub-array
yyjson_mut_obj_add_obj(doc, obj, "meta");   // returns new sub-object
yyjson_mut_obj_remove_str(obj, "name");
yyjson_mut_obj_rename_key(doc, obj, "old", "new");
```

**Important:** A `yyjson_mut_val` can only be added to one container. Adding the same value to multiple containers is wrong.

## Immutable -> Mutable -> Immutable

```c
// immutable to mutable
yyjson_mut_doc *mdoc = yyjson_doc_mut_copy(idoc, alc);
yyjson_mut_val *mval = yyjson_val_mut_copy(mdoc, ival);

// mutable to immutable
yyjson_doc *idoc = yyjson_mut_doc_imut_copy(mdoc, alc);
```

## Write JSON

```c
// To string
char *json = yyjson_write(doc, flg, &len);
char *json = yyjson_mut_write(mdoc, flg, &len);
char *json = yyjson_val_write(val, flg, &len);
char *json = yyjson_mut_val_write(mval, flg, &len);
free(json);

// To string with allocator/options
char *json = yyjson_write_opts(doc, flg, alc, &len, &err);

// To file
yyjson_write_file("out.json", doc, flg, alc, &err);
yyjson_mut_write_file("out.json", mdoc, flg, alc, &err);

// To FILE*
yyjson_write_fp(fp, doc, flg, alc, &err);

// To preallocated buffer
size_t n = yyjson_write_buf(buf, buf_len, doc, flg, &err);
```

**Common write flags** (combine with `|`):
- `YYJSON_WRITE_NOFLAG` (0) — minified
- `YYJSON_WRITE_PRETTY` — 4-space indent
- `YYJSON_WRITE_PRETTY_TWO_SPACES`
- `YYJSON_WRITE_ESCAPE_UNICODE`
- `YYJSON_WRITE_ESCAPE_SLASHES`
- `YYJSON_WRITE_ALLOW_INF_AND_NAN`
- `YYJSON_WRITE_INF_AND_NAN_AS_NULL`
- `YYJSON_WRITE_ALLOW_INVALID_UNICODE`
- `YYJSON_WRITE_NEWLINE_AT_END`
- `YYJSON_WRITE_FP_TO_FLOAT`
- `YYJSON_WRITE_FP_TO_FIXED(prec)` — prec 1..15

## JSON Pointer (RFC 6901)

Query:
```c
yyjson_val *v = yyjson_doc_ptr_get(doc, "/users/0/name");
yyjson_val *v = yyjson_ptr_get(root, "/a/b");
```

Modify (mutable only):
```c
yyjson_mut_doc_ptr_set(doc, "/a", yyjson_mut_int(doc, 9));
yyjson_mut_doc_ptr_add(doc, "/b/-", yyjson_mut_int(doc, 4));  // "-" appends to array
yyjson_mut_doc_ptr_remove(doc, "/b");
```

Error/context variants end with `x`:
```c
yyjson_ptr_err err;
yyjson_ptr_ctx ctx;
yyjson_mut_doc_ptr_getx(doc, ptr, len, &ctx, &err);
```

## JSON Patch / Merge Patch

```c
// RFC 6902 Patch
yyjson_mut_val *out = yyjson_patch(doc, orig, patch, &err);

// RFC 7386 Merge Patch
yyjson_mut_val *out = yyjson_merge_patch(doc, orig, patch);
```

## Memory Allocator

```c
// Default allocator: pass NULL for alc

// Stack allocator for small JSON
char buf[128 * 1024];
yyjson_alc alc;
yyjson_alc_pool_init(&alc, buf, sizeof(buf));
yyjson_doc *doc = yyjson_read_opts(dat, len, 0, &alc, NULL);

// Dynamic allocator
yyjson_alc *alc = yyjson_alc_dyn_new();
...
yyjson_alc_dyn_free(alc);

// Custom allocator
static const yyjson_alc MY_ALC = { my_malloc, my_realloc, my_free, ctx };
```

## Number Control

Per-value output format (mutable):
```c
yyjson_mut_set_fp_to_float(val, true);
yyjson_mut_set_fp_to_fixed(val, 6);
```

Standalone conversion:
```c
const char *end = yyjson_read_number(dat, val, flg, alc, &err);
char *end = yyjson_write_number(val, buf);
```

## Compile-time Options

Define before including `yyjson.h` or at build time:
- `YYJSON_DISABLE_READER`
- `YYJSON_DISABLE_WRITER`
- `YYJSON_DISABLE_INCR_READER`
- `YYJSON_DISABLE_UTILS` (Pointer/Patch/Merge Patch)
- `YYJSON_DISABLE_FAST_FP_CONV`
- `YYJSON_DISABLE_NON_STANDARD`
- `YYJSON_DISABLE_UTF8_VALIDATION`
- `YYJSON_READER_DEPTH_LIMIT`

## Null Safety

All public APIs null-check inputs. You can safely chain without manual null checks:
```c
yyjson_val *v = yyjson_obj_get(root, "key");
const char *s = yyjson_get_str(v); // NULL if v is NULL
```

For hot loops where non-null is guaranteed, use `unsafe_` prefix APIs to skip checks:
```c
unsafe_yyjson_is_str(key)
unsafe_yyjson_get_uint(val)
```

## Common Pattern: Read -> Modify -> Write

```c
yyjson_doc *idoc = yyjson_read_file("in.json", 0, NULL, NULL);
yyjson_mut_doc *doc = yyjson_doc_mut_copy(idoc, NULL);
yyjson_mut_val *root = yyjson_mut_doc_get_root(doc);

// iterate and remove nulls
yyjson_mut_val *k, *v;
yyjson_mut_obj_iter iter = yyjson_mut_obj_iter_with(root);
while ((k = yyjson_mut_obj_iter_next(&iter))) {
    v = yyjson_mut_obj_iter_get_val(k);
    if (yyjson_mut_is_null(v)) yyjson_mut_obj_iter_remove(&iter);
}

yyjson_mut_write_file("out.json", doc, YYJSON_WRITE_PRETTY, NULL, NULL);
yyjson_doc_free(idoc);
yyjson_mut_doc_free(doc);
```
