---
name: yyjson
description: High-performance C JSON parsing, creation, and modification with yyjson. Use when working with yyjson documents, string copy-vs-borrow ownership, dangling c_str() bugs, yyjson_alc allocators, or writer output freeing.
---

# yyjson Usage Guide

Single-header ANSI C (`yyjson.h` + `yyjson.c`). Two data models: immutable (`yyjson_doc`/`yyjson_val`) and mutable (`yyjson_mut_doc`/`yyjson_mut_val`). Free the doc to free all values. Arrays/objects are linked lists; index/key lookup is O(n) — prefer iterators.

**Ownership rule of thumb**: the document owns everything it allocated, and *borrows* every string you hand it unless you use a `cpy` API. See [Memory Management](#memory-management).

## Project Notes

- The bundled copy is in `src/ext/yyjson` (currently **0.12.0**). The build option `lc_yyjson_use_xrepo` switches to the xmake-repo package.
- Luisa code commonly supplies a custom `yyjson_alc` that forwards to `luisa::detail::allocator_allocate/reallocate/deallocate` with 16-byte alignment.
- `yyjson_write()` / `yyjson_mut_write()` / `yyjson_val_write()` allocate their output string with the default allocator. Use the `_opts` variants and pass an allocator if the output must use the same allocator as the document.
- The bundled target compiles `yyjson.c` with `/utf-8` on MSVC.
- Where it is used today (good patterns to copy): `src/ast/json2ast.cpp` (reader + budget-limited
  custom `yyjson_alc`, frees the doc on every exit), `src/xir/translators/xir2json.cpp` (builder +
  `_opts` writer sharing one 16-byte-aligned `yyjson_alc`), `dgm_ao_render/baker/ao_bake_meta.cpp`
  (builder with the default allocator, literal keys, `yyjson_mut_strcpy` values).
- Read [Memory Management](#memory-management) before adding or freeing anything: most yyjson bugs
  in this repo are borrowed-string lifetime bugs, not syntax bugs.

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

**Read flags**: `YYJSON_READ_NOFLAG` (RFC 8259), `ALLOW_COMMENTS`, `ALLOW_TRAILING_COMMAS`, `ALLOW_INF_AND_NAN`, `ALLOW_INVALID_UNICODE`, `NUMBER_AS_RAW`, `BIGNUM_AS_RAW` (overridden by `NUMBER_AS_RAW`), `ALLOW_BOM`, `ALLOW_EXT_NUMBER`, `ALLOW_EXT_ESCAPE`, `ALLOW_EXT_WHITESPACE`, `ALLOW_SINGLE_QUOTED_STR`, `ALLOW_UNQUOTED_KEY`, `INSITU` (modifies and *aliases* the input buffer — it must be padded with `YYJSON_PADDING_SIZE` zero bytes and outlive the doc; see [Reader memory](#reader-memory)), `STOP_WHEN_DONE` (NDJSON), `JSON5`.

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

**Important**: `yyjson_mut_str` / `_strn` / `_raw` / `_rawn` and **every object key** are *not
copied* — only the `cpy` variants (`yyjson_mut_strcpy`, `_strncpy`, `_rawcpy`, `_rawncpy`,
`obj_add_strcpy`/`obj_add_strncpy` for the **value** only) allocate into the document. See
[Memory Management](#memory-management) before adding any dynamic string.

## Immutable ↔ Mutable

```c
yyjson_mut_doc *mdoc = yyjson_doc_mut_copy(idoc, alc);        // imut -> mut document
yyjson_mut_val *mval = yyjson_val_mut_copy(mdoc, ival);       // imut -> mut value (into mdoc)
yyjson_mut_doc *mdoc2 = yyjson_mut_doc_mut_copy(mdoc, alc);   // mut -> mut (deep)
yyjson_mut_val *mval2 = yyjson_mut_val_mut_copy(mdoc2, mval); // mut -> mut value (into mdoc2)
yyjson_doc *idoc2 = yyjson_mut_doc_imut_copy(mdoc, alc);      // mut -> imut document
yyjson_doc *idoc3 = yyjson_mut_val_imut_copy(mval, alc);      // mut value -> *new document*
```

All of these **deep-copy, string bytes included** (`STR`/`RAW` values are re-allocated into the
target's `str_pool`), so the source document may be freed as soon as the copy returns. They are
also recursive — deep trees can overflow the stack. Note the `*_imut_copy` results are independent
**documents** (own pools, free with `yyjson_doc_free`), not values that belong to another doc —
the `*_mut_copy` results are values owned by the `doc` passed as the first argument.

## Write JSON

```c
char *json = yyjson_write(doc, flg, &len);      free(json);            // default allocator only
char *json = yyjson_mut_write(mdoc, flg, &len); free(json);            // default allocator only
char *json = yyjson_val_write(val, flg, &len);   free(json);           // default allocator only
// _opts variants allocate with the allocator you pass — release with the same one:
char *json = yyjson_mut_write_opts(mdoc, flg, &alc, &len, &err);       // alc.free(alc.ctx, json)
yyjson_write_file("out.json", doc, flg, alc, &err);        // allocates + frees internally
yyjson_mut_write_file("out.json", mdoc, flg, alc, &err);   // allocates + frees internally
yyjson_write_fp(fp, doc, flg, alc, &err);
size_t n = yyjson_write_buf(buf, buf_len, doc, flg, &err); // no allocation at all
```

See [Writer memory](#writer-memory) — release the string with the **same** allocator it was
allocated with: `free()` only matches a `NULL` / libc-backed `yyjson_alc`, and a custom `alc` must
be released via `alc.free(alc.ctx, json)`.

**Write flags**: `YYJSON_WRITE_NOFLAG` (minified), `PRETTY` (4-space), `PRETTY_TWO_SPACES`, `ESCAPE_UNICODE`, `ESCAPE_SLASHES`, `ALLOW_INF_AND_NAN`, `INF_AND_NAN_AS_NULL`, `ALLOW_INVALID_UNICODE`, `LOWERCASE_HEX`, `NEWLINE_AT_END`, `FP_TO_FLOAT`, `FP_TO_FIXED(prec)` (1..15).

## JSON Pointer / Patch (RFC 6901/6902/7386)

```c
// Query
yyjson_val *v = yyjson_doc_ptr_get(doc, "/users/0/name");

// Modify (mutable)
yyjson_mut_doc_ptr_set(doc, "/a", yyjson_mut_int(doc, 9));
yyjson_mut_doc_ptr_add(doc, "/b/-", val);    // "-" appends to array
yyjson_mut_doc_ptr_remove(doc, "/b");

// Error variants: _getx takes yyjson_ptr_err; _setx/_addx/_removex take yyjson_ptr_ctx + yyjson_ptr_err

// Patch
yyjson_mut_val *out = yyjson_patch(doc, orig, patch, &err);     // RFC 6902
yyjson_mut_val *out = yyjson_merge_patch(doc, orig, patch);     // RFC 7386
```
## Memory Management

Everything below was checked against the bundled sources (`src/ext/yyjson/src/yyjson.h`,
`yyjson.c`, v0.12.0).

### Who owns what

- **The document owns everything.** `yyjson_read*()` gives back a `yyjson_doc` holding all values
  and strings; `yyjson_mut_doc_new()` gives back a `yyjson_mut_doc` holding every value created
  through it. Values have exactly the lifetime of their document: no per-value free, no
  refcounting. Release with `yyjson_doc_free()` / `yyjson_mut_doc_free()` (both no-ops on `NULL`)
  and treat every `yyjson_val *`, `yyjson_mut_val *`, and every `const char *` returned by
  `yyjson_get_str()` / `yyjson_get_raw()` as dangling afterwards.
- **Immutable doc layout:** one block containing the `yyjson_doc` header followed by all
  `yyjson_val` slots (grown with `alc.realloc` *during* the read, so don't cache `yyjson_val *`
  across a read call; stable once it returns), plus one block holding the input-text copy
  (`doc->str_pool`). `yyjson_doc_free()` = at most two `alc.free` calls.
- **Mutable doc layout:** two bump arenas — `val_pool` (16-byte `yyjson_mut_val` slots) and
  `str_pool` (bytes for *copied* strings and raw values). Chunks double in size (start
  `0x100` B / 16 values, capped at `0x10000000` B / `0x1000000` values) and nothing is returned
  until `yyjson_mut_doc_free()`. `*_remove`, `*_iter_remove`, `*_clear` unlink values but reclaim
  **no** memory.
- Containers (array/object) are intrusive circular lists, so a `yyjson_mut_val` can live in **one**
  container only; putting it in two corrupts both. To repeat content, create a fresh value or
  `yyjson_mut_val_mut_copy(doc, val)` (recursive — deep trees can blow the stack).

### Borrow vs. copy (the #1 yyjson footgun)

Non-`cpy` creation APIs store the caller's pointer **as is**: the buffer must stay alive *and*
**unmodified** for the whole lifetime of the document.

| Copies into the doc | Borrows (no copy) |
|---|---|
| `yyjson_mut_strcpy(doc, s)` | `yyjson_mut_str(doc, s)` (needs NUL: calls `strlen`) |
| `yyjson_mut_strncpy(doc, s, n)` | `yyjson_mut_strn(doc, s, n)` |
| `yyjson_mut_rawcpy` / `yyjson_mut_rawncpy` | `yyjson_mut_raw` / `yyjson_mut_rawn` |
| `yyjson_mut_obj_add_strcpy(doc, o, k, v)` — **value only** | `yyjson_mut_obj_add_str` (key **and** value) |
| `yyjson_mut_obj_add_strncpy(doc, o, k, v, n)` — **value only** | `yyjson_mut_obj_add_strn`, plus the key `k` |
| `yyjson_mut_arr_add_strcpy` / `_add_strncpy` | `yyjson_mut_arr_add_str` / `_add_strn` |
| `yyjson_mut_arr_with_strcpy` / `_with_strncpy` | `yyjson_mut_arr_with_str` / `_with_strn` |
| — | **all object keys, always**: `yyjson_mut_obj_add_func` does `key->uni.str = _key` |

The one API that *does* copy a key is `yyjson_mut_obj_rename_key` / `_rename_keyn` ("the `new_key`
is copied and held by doc").

Consequences:

- There is **no** copy-the-key convenience. For a dynamic key build the pair yourself:
  `yyjson_mut_obj_add(doc, yyjson_mut_strcpy(doc, key.c_str()), value)`.
- `yyjson_mut_obj_add_strncpy(doc, o, "k", sv.data(), sv.size())` is the way to add a
  `luisa::string_view` / `std::string` that may embed no NUL terminator: length-delimited **and**
  copied.
- Borrowed content is aliased, not snapshotted — mutating the source buffer afterwards rewrites
  the JSON. Worse, "does this string need escaping?" is decided and cached in the value's tag at
  creation (`YYJSON_SUBTYPE_NOESC`, computed by `unsafe_yyjson_set_str`): `yyjson_mut_str`,
  `yyjson_mut_strcpy`, the **key** of every `yyjson_mut_obj_add_*`, and the value of
  `yyjson_mut_obj_add_str` — so mutating a borrowed string to later contain `"`, `\` or a control
  character emits **invalid** JSON instead of escaping it. The `n`-family (`yyjson_mut_strn`,
  `yyjson_mut_strncpy`, `obj_add_strn`, `obj_add_strncpy`, `obj_add_strcpy`) store
  `SUBTYPE_NONE` and are re-checked at write time. Either way, don't mutate what you added.
- `yyjson_mut_str` also *requires* a valid NUL terminator; `str`-family on an unterminated buffer
  is out-of-bounds reading, not just a wrong length.
- Copied strings live in the document's own `str_pool`, `len + 1` bytes (`cpy` helpers write the
  terminator, so `yyjson_get_str()` works even on `strncpy`-created values). To fill pool memory
  in place and skip the double copy, `unsafe_yyjson_mut_str_alc(doc, len)` returns raw space — but
  it does **not** null-terminate for you.

### Never hand a borrow API a temporary

No diagnostic, no crash at the call site — the doc keeps a pointer that dies at the end of the
full expression, and you see garbage (or a crash at write/`doc_free` time, when the allocator
reuses the block):

```cpp
// BAD: .string() is a temporary; the pointer is dead on the next line
yyjson_mut_obj_add_str(doc, root, "lightmap", cfg.output_path.filename().string().c_str());
// BAD, same class: luisa::format(...).c_str(), a local char buf[], a loop-scoped std::string
// added to an array that outlives the iteration, a std::string member written later

// GOOD: copy — the document owns the bytes
yyjson_mut_obj_add_val(doc, root, "lightmap",
                       yyjson_mut_strcpy(doc, cfg.output_path.filename().string().c_str()));
// GOOD: extend the source's lifetime past the document
auto name = cfg.output_path.filename().string();   // alive until end of scope
yyjson_mut_obj_add_str(doc, root, "lightmap", name.c_str());
// GOOD: length-delimited copy, no NUL requirement
yyjson_mut_obj_add_strncpy(doc, root, "lightmap", sv.data(), sv.size());
```

Safe to borrow: string literals and other static storage (keys like `"schema"`, `"version"`), and
buffers of objects *known* to outlive the document. Default to copying for anything computed — one
bump-arena allocation per string is cheaper than the bug. Real examples:
`src/xir/translators/xir2json.cpp` (literal keys + `add_strncpy` for dynamic values) and
`dgm_ao_render/baker/ao_bake_meta.cpp` (literal keys + `yyjson_mut_strcpy` for filenames and
names).

### Reader memory

- **Default (no `YYJSON_READ_INSITU`)**: the whole input is copied into a doc-owned buffer
  (`len + YYJSON_PADDING_SIZE`, padded with 4 zero bytes) and escapes are unescaped in place
  there. So `yyjson_get_str()` stays valid until `yyjson_doc_free()` even if the caller's buffer
  is gone — this is why `const_cast<char *>(sv.data())` in `yyjson_read_opts` is legal.
- **`YYJSON_READ_INSITU`**: no copy; values point **into your buffer**, the reader rewrites it, and
  the buffer must be padded with at least `YYJSON_PADDING_SIZE` (4) zero bytes and must outlive the
  doc. Never on `const` data, string literals, or shared/pooled memory.
- Don't return `string_view`s into a doc you then free. `src/ast/json2ast.cpp` is the pattern to
  follow: it decodes into the AST (copying what must survive) and only then calls
  `yyjson_doc_free(document)` on every exit path.
- Pre-sizing: `yyjson_read_max_memory_usage(len, flg)` = `len * (12 + !INSITU) + 256`; combine it
  with `yyjson_alc_pool_init` to read with zero malloc calls.

### Writer memory

- `yyjson_write_opts(doc, flg, alc, &len, &err)` / `yyjson_mut_write_opts` /
  `yyjson_val_write_opts` / `yyjson_mut_val_write_opts` return a NUL-terminated UTF-8 string
  allocated with the **`alc` you passed** (`len` excludes the terminator). Free it the matching
  way: `free()` for a `NULL` alc, otherwise `alc.free(alc.ctx, json)` — see
  `write_document()` in `src/xir/translators/xir2json.cpp`.
- The non-`_opts` helpers (`yyjson_write`, `yyjson_mut_write`, `yyjson_val_write`) always use the
  **libc default** allocator, so `free()` is the only correct release; with a custom `yyjson_alc`
  for the document, output and document then live in two different heaps (that is also where
  mismatched free bugs come from). Use the `_opts` variant to keep one allocator.
- `yyjson_write_file` / `yyjson_mut_write_file` / `yyjson_write_fp` allocate their temp buffer with
  the passed `alc`, free it before returning, and report failure via `bool` + `yyjson_write_err`
  (`YYJSON_WRITE_ERROR_MEMORY_ALLOCATION` vs. I/O codes). Nothing to clean up at the call site.
- `yyjson_write_buf(buf, cap, ...)` and `yyjson_write_number(val, buf)` allocate **nothing**:
  `write_buf` needs slack *beyond* the final JSON size (temporary space) and returns 0 if the
  buffer is too small; `write_number` needs a caller buffer of ≥ 21 bytes for integers and ≥ 40
  bytes for reals, and returns `NULL` (never overflows) on bad input.
- The writer only *reads* the document, so borrowed strings must still be alive at write time —
  writing after the source `std::string`s died is exactly the dangling case above.

### `yyjson_alc` and the repo's allocators

```c
// Fixed pool on the stack/scratch buffer (>= 8 words). NOT thread-safe, and designed for
// one document at a time: reuse across docs with partial frees fragments it.
char buf[128 * 1024];
yyjson_alc alc; yyjson_alc_pool_init(&alc, buf, sizeof(buf));
yyjson_doc *doc = yyjson_read_opts(dat, len, 0, &alc, NULL);  // NULL + MEMORY_ALLOCATION if full

// Growable pool: chunks from malloc, all released at once. NOT thread-safe.
yyjson_alc *alc = yyjson_alc_dyn_new();   // free with yyjson_alc_dyn_free() AFTER doc_free()

// Fully custom: signatures are (ctx,size), (ctx,ptr,old_size,size), (ctx,ptr)
yyjson_alc MY_ALC = { my_malloc, my_realloc, my_free, my_ctx };
```

- Both documents and readers **copy** the `yyjson_alc` struct (`doc->alc = *alc`), so the struct
  itself may be a stack temporary, but everything it references — a pool's buffer, the `ctx`
  object — must outlive the document (and, for writers, the output string).
- `NULL` alc ⇒ `YYJSON_DEFAULT_ALC` (libc `malloc`/`realloc`/`free`).
- In this repo: `xir2json.cpp` forwards to `luisa::detail::allocator_allocate/reallocate/deallocate`
  with 16-byte alignment; `json2ast.cpp` wraps libc with a `YYJsonBudget` `ctx` that returns `NULL`
  past `max_parse_memory_bytes`, turning a hostile/large document into a clean
  `YYJSON_READ_ERROR_MEMORY_ALLOCATION` instead of unbounded memory growth — pair it with
  `max_document_bytes` / `max_string_bytes` checks on the input.
- Allocation failures are recoverable, not fatal: every `yyjson_mut_*` creator returns `NULL` and
  every `yyjson_mut_obj_add_*` / `arr_add_*` returns `false`. If unchecked, you silently lose a
  field rather than crash — check them when the JSON is load-bearing.

### Thread safety

- Building a `yyjson_mut_doc` is single-threaded: arena cursors, no locks. One document per thread.
- `yyjson_read_opts` / `yyjson_write_opts` are thread-safe when `alc` is thread-safe (or `NULL`)
  and the input data isn't modified concurrently (guaranteed unless `INSITU`).
- A finished `yyjson_doc` can be read concurrently (`yyjson_get_*`, iterators are stack-local
  state), but never mutate values from two threads (`yyjson_val_set_*` / `yyjson_set_str_noesc`
  change tags in place) and never read while another thread frees it.
- `yyjson_alc_pool_init` and `yyjson_alc_dyn_new` allocators are explicitly documented as
  **not thread-safe**.

### C++ lifetime hygiene (Luisa side)

- Wrap the document: `std::unique_ptr<yyjson_mut_doc, void (*)(yyjson_mut_doc*)>` (or a scope
  guard) instead of hand-writing `yyjson_mut_doc_free()` before each `return`. Manual frees are
  easy to miss on early returns — `ao_bake_meta.cpp` frees on both paths, which is the fragile
  version.
- `LUISA_ERROR*` / `luisa::log_error` is `[[noreturn]] noexcept` and calls `std::abort()`, so it
  does **not** unwind; what does unwind is ordinary C++ work inside the build loop (`std::string`,
  `luisa::format`, `vector` growth ⇒ `std::bad_alloc`). Keep the loop body between `doc_new` and
  `doc_free` allocation-light, or hold the doc in a guard.
- Nothing in the doc needs `free()`-ing individually: not values, not copied strings, not keys.
  Mixing `free()` / `luisa::detail::allocator_deallocate` with yyjson-owned memory is a bug; only
  the writer's returned string (and `yyjson_alc_dyn_new`'s allocator) is ever freed by hand.
- `yyjson_mut_doc_set_str_pool_size(doc, total_string_bytes)` and
  `yyjson_mut_doc_set_val_pool_size(doc, value_count)` set the *next* chunk size (no immediate
  allocation) — worth it for large, predictable dumps (many `strcpy`ed strings or many values) to
  avoid the doubling-realloc churn. Note the string pool is only used by copied strings/raw values.

## Number & Compile-time

```c
// Per-value output format (mutable)
yyjson_mut_set_fp_to_float(val, true);
yyjson_mut_set_fp_to_fixed(val, 6);
```

Compile-time defines: `YYJSON_DISABLE_READER`, `YYJSON_DISABLE_WRITER`, `YYJSON_DISABLE_INCR_READER`, `YYJSON_DISABLE_UTILS` (Pointer/Patch), `YYJSON_DISABLE_FAST_FP_CONV`, `YYJSON_DISABLE_NON_STANDARD`, `YYJSON_DISABLE_UTF8_VALIDATION`, `YYJSON_DISABLE_UNALIGNED_MEMORY_ACCESS`. `YYJSON_READER_DEPTH_LIMIT`.

## Null Safety

All APIs null-check inputs — safe to chain: `yyjson_get_str(yyjson_obj_get(root, "key"))` returns NULL if missing. For hot loops, `unsafe_` prefix skips checks: `unsafe_yyjson_is_str(k)`, `unsafe_yyjson_get_uint(v)`.

## Common Pattern: Read → Modify → Write

```c
yyjson_doc *idoc = yyjson_read_file("in.json", 0, NULL, NULL);
yyjson_mut_doc *doc = yyjson_doc_mut_copy(idoc, NULL);   // deep copy: strings are copied here
yyjson_mut_val *root = yyjson_mut_doc_get_root(doc);

yyjson_mut_val *k, *v;
yyjson_mut_obj_iter iter = yyjson_mut_obj_iter_with(root);
while ((k = yyjson_mut_obj_iter_next(&iter))) {
    v = yyjson_mut_obj_iter_get_val(k);
    if (yyjson_mut_is_null(v)) yyjson_mut_obj_iter_remove(&iter);  // unlinks, frees nothing
}

yyjson_mut_write_file("out.json", doc, YYJSON_WRITE_PRETTY, NULL, NULL);
yyjson_doc_free(idoc);
yyjson_mut_doc_free(doc);
```

`yyjson_doc_mut_copy` deep-copies, so the mutable doc is independent of `idoc`'s memory (you may
free `idoc` immediately). The plain `*_write` calls above use the libc allocator → `free()`; if
the mutable doc was built with a custom `yyjson_alc`, use `yyjson_mut_write_opts(doc, flg, &alc, …)`
and `alc.free(alc.ctx, json)` instead.
