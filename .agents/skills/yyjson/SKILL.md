---
name: yyjson
description: High-performance C JSON parsing, creation, and modification with yyjson.
---

# yyjson Usage Guide

Single-header ANSI C (`yyjson.h` + `yyjson.c`). Two data models: immutable (`yyjson_doc`/`yyjson_val`) and mutable (`yyjson_mut_doc`/`yyjson_mut_val`). Free the doc to free all values. Arrays/objects are linked lists; index/key lookup is O(n) — prefer iterators.

## Read JSON

```c
yyjson_doc *doc = yyjson_read(json, len, 0);
yyjson_doc *doc = yyjson_read_file("path", 0, NULL, &err);
yyjson_doc *doc = yyjson_read_opts((char*)dat, len, flg, alc, &err);
yyjson_doc *doc = yyjson_read_fp(fp, flg, alc, &err);

// Incremental
yyjson_incr_state *s = yyjson_incr_new(buf, len, flg, alc);
yyjson_doc *doc = yyjson_incr_read(s, read_len, &err);
yyjson_incr_free(s);
```

**Read flags**: `YYJSON_READ_NOFLAG` (RFC 8259), `ALLOW_COMMENTS`, `ALLOW_TRAILING_COMMAS`, `ALLOW_INF_AND_NAN`, `ALLOW_INVALID_UNICODE`, `NUMBER_AS_RAW`, `BIGNUM_AS_RAW`, `INSITU` (modify input buffer; pad with `YYJSON_PADDING_SIZE` zeros), `STOP_WHEN_DONE` (NDJSON), `JSON5`.

**Error**: `yyjson_read_err err;` → `err.code`, `err.msg`, `err.pos`. `yyjson_locate_pos(dat, len, err.pos, &line, &col, &chr)` for line/col.

## Access Values

```c
yyjson_val *root = yyjson_doc_get_root(doc);

// Type checks
yyjson_is_null(v)  yyjson_is_bool(v)  yyjson_is_uint(v)  yyjson_is_sint(v)
yyjson_is_int(v)   yyjson_is_real(v)  yyjson_is_num(v)   yyjson_is_str(v)
yyjson_is_arr(v)   yyjson_is_obj(v)   yyjson_is_raw(v)

// Getters
bool b = yyjson_get_bool(v);          uint64_t u = yyjson_get_uint(v);
int64_t s = yyjson_get_sint(v);       int i = yyjson_get_int(v);
double d = yyjson_get_real(v);        const char *str = yyjson_get_str(v);
size_t len = yyjson_get_len(v);       const char *raw = yyjson_get_raw(v);

// Array (linear by index)
size_t sz = yyjson_arr_size(arr);
yyjson_val *v = yyjson_arr_get(arr, idx);
yyjson_val *v = yyjson_arr_get_first(arr) / yyjson_arr_get_last(arr);

// Object (linear by key)
size_t sz = yyjson_obj_size(obj);
yyjson_val *v = yyjson_obj_get(obj, "key");
yyjson_val *v = yyjson_obj_getn(obj, "key", key_len);

// String compare
yyjson_equals_str(v, "abc");  yyjson_equals_strn(v, "abc", 3);
```

## Iterate

```c
// Array
yyjson_val *v;
yyjson_arr_iter iter = yyjson_arr_iter_with(arr);
while ((v = yyjson_arr_iter_next(&iter))) { }
// Macro:
yyjson_arr_foreach(arr, idx, max, v) { }

// Object
yyjson_val *k, *v;
yyjson_obj_iter iter = yyjson_obj_iter_with(obj);
while ((k = yyjson_obj_iter_next(&iter))) { v = yyjson_obj_iter_get_val(k); }
// Macro:
yyjson_obj_foreach(obj, idx, max, k, v) { }
```

Mutable: use `yyjson_mut_` prefix. Inside mutable iter: `yyjson_mut_arr_iter_remove(&iter)` / `yyjson_mut_obj_iter_remove(&iter)`.

## Create / Modify JSON

```c
yyjson_mut_doc *doc = yyjson_mut_doc_new(alc);
yyjson_mut_val *root = yyjson_mut_obj(doc);  // or yyjson_mut_arr(doc)
yyjson_mut_doc_set_root(doc, root);

// Scalars
yyjson_mut_null(doc)  yyjson_mut_bool(doc, true)
yyjson_mut_uint(doc, 123)  yyjson_mut_sint(doc, -123)  yyjson_mut_int(doc, 123)
yyjson_mut_real(doc, 1.5)  yyjson_mut_str(doc, "hello")       // refs caller's string
yyjson_mut_strcpy(doc, "hello")                                // copies into doc
yyjson_mut_strn(doc, "hello", 5)  yyjson_mut_strncpy(doc, "hello", 5)  // n-variants

// Array
yyjson_mut_arr_append(arr, val)  yyjson_mut_arr_prepend(arr, val)
yyjson_mut_arr_insert(arr, val, idx)  yyjson_mut_arr_replace(arr, idx, val)
yyjson_mut_arr_remove(arr, idx)  yyjson_mut_arr_remove_first/last(arr)  yyjson_mut_arr_clear(arr)
// Convenience: yyjson_mut_arr_add_int(doc, arr, 42) / _add_str / _add_arr / _add_obj
// From C array: yyjson_mut_arr_with_sint32(doc, nums, 3)

// Object
yyjson_mut_obj_add(obj, key, val)      // allows duplicates
yyjson_mut_obj_put(obj, key, val)      // removes existing first (O(n))
yyjson_mut_obj_remove(obj, key)  yyjson_mut_obj_clear(obj)
// Convenience: yyjson_mut_obj_add_str/int/arr/obj(doc, obj, "key", ...)
yyjson_mut_obj_remove_str(obj, "key")  yyjson_mut_obj_rename_key(doc, obj, "old", "new")
```

**Important**: A `yyjson_mut_val` can only be added to one container.

## Immutable ↔ Mutable

```c
yyjson_mut_doc *mdoc = yyjson_doc_mut_copy(idoc, alc);
yyjson_mut_val *mval = yyjson_val_mut_copy(mdoc, ival);
yyjson_doc *idoc = yyjson_mut_doc_imut_copy(mdoc, alc);
```

## Write JSON

```c
char *json = yyjson_write(doc, flg, &len);      free(json);
char *json = yyjson_mut_write(mdoc, flg, &len); free(json);
char *json = yyjson_val_write(val, flg, &len);   free(json);    // per-value
yyjson_write_file("out.json", doc, flg, alc, &err);
yyjson_mut_write_file("out.json", mdoc, flg, alc, &err);
yyjson_write_fp(fp, doc, flg, alc, &err);
size_t n = yyjson_write_buf(buf, buf_len, doc, flg, &err);
```

**Write flags**: `YYJSON_WRITE_NOFLAG` (minified), `PRETTY` (4-space), `PRETTY_TWO_SPACES`, `ESCAPE_UNICODE`, `ESCAPE_SLASHES`, `ALLOW_INF_AND_NAN`, `INF_AND_NAN_AS_NULL`, `ALLOW_INVALID_UNICODE`, `NEWLINE_AT_END`, `FP_TO_FLOAT`, `FP_TO_FIXED(prec)` (1..15).

## JSON Pointer / Patch (RFC 6901/6902/7386)

```c
// Query
yyjson_val *v = yyjson_doc_ptr_get(doc, "/users/0/name");

// Modify (mutable)
yyjson_mut_doc_ptr_set(doc, "/a", yyjson_mut_int(doc, 9));
yyjson_mut_doc_ptr_add(doc, "/b/-", val);    // "-" appends to array
yyjson_mut_doc_ptr_remove(doc, "/b");

// Error variants: _getx/_setx with yyjson_ptr_ctx + yyjson_ptr_err

// Patch
yyjson_mut_val *out = yyjson_patch(doc, orig, patch, &err);     // RFC 6902
yyjson_mut_val *out = yyjson_merge_patch(doc, orig, patch);     // RFC 7386
```

## Memory Allocator

```c
// Stack allocator
char buf[128*1024];
yyjson_alc alc; yyjson_alc_pool_init(&alc, buf, sizeof(buf));
yyjson_doc *doc = yyjson_read_opts(dat, len, 0, &alc, NULL);

// Dynamic
yyjson_alc *alc = yyjson_alc_dyn_new();
// ...
yyjson_alc_dyn_free(alc);

// Custom
static const yyjson_alc MY_ALC = { my_malloc, my_realloc, my_free, ctx };
```

## Number & Compile-time

```c
// Per-value output format (mutable)
yyjson_mut_set_fp_to_float(val, true);
yyjson_mut_set_fp_to_fixed(val, 6);
```

Compile-time defines: `YYJSON_DISABLE_READER`, `YYJSON_DISABLE_WRITER`, `YYJSON_DISABLE_INCR_READER`, `YYJSON_DISABLE_UTILS` (Pointer/Patch), `YYJSON_DISABLE_FAST_FP_CONV`, `YYJSON_DISABLE_NON_STANDARD`, `YYJSON_DISABLE_UTF8_VALIDATION`. `YYJSON_READER_DEPTH_LIMIT`.

## Null Safety

All APIs null-check inputs — safe to chain: `yyjson_get_str(yyjson_obj_get(root, "key"))` returns NULL if missing. For hot loops, `unsafe_` prefix skips checks: `unsafe_yyjson_is_str(k)`, `unsafe_yyjson_get_uint(v)`.

## Common Pattern: Read → Modify → Write

```c
yyjson_doc *idoc = yyjson_read_file("in.json", 0, NULL, NULL);
yyjson_mut_doc *doc = yyjson_doc_mut_copy(idoc, NULL);
yyjson_mut_val *root = yyjson_mut_doc_get_root(doc);

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
