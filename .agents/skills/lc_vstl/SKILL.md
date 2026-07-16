---
name: lc_vstl
description: vstd containers: HashMap, queues, pools, variant, smart pointers, and utilities.
---

# LuisaCompute VSTL Container Guide

`vstd` = custom containers/utilities under `include/luisa/vstl/`. Many are aliases to `luisa::` STL replacements.

## Aliases

| vstd type | Actual |
|---|---|
| `vstd::vector<T>` | `luisa::vector<T>` |
| `vstd::fixed_vector<T,N>` | `luisa::fixed_vector<T,N>` |
| `vstd::span<T>` | `luisa::span<T>` |
| `vstd::string` | `std::basic_string<char,...,luisa::allocator<char>>` |
| `vstd::wstring` | `std::basic_string<wchar_t,...,luisa::allocator<wchar_t>>` |
| `vstd::string_view` | `luisa::string_view` |
| `vstd::function<T>` | `luisa::move_only_function<T>` |
| `vstd::shared_ptr<T>` | `luisa::shared_ptr<T>` |
| `vstd::spin_mutex` | `luisa::spin_mutex` |
| `vstd::unordered_map/set` | `luisa::unordered_map/set` |

## Vector Helpers

`#include <luisa/vstl/vector.h>`

```cpp
vstd::push_back_func(vec, 10, [&](size_t i) { return i * 2; });
vstd::push_back_func(vec, 5, [] { return Foo{}; });
vstd::push_back_all(vec, ptr, n);
vstd::push_back_all(vec, {1, 2, 3});
vstd::push_back_all(vec, some_span);
```

## HashMap

`#include <luisa/vstl/hash_map.h>`. Power-of-2 capacity, open addressing, per-bucket red-black trees.

```cpp
vstd::HashMap<Key, Value> map;           // default
vstd::HashMap<Key, Value> map(capacity); // pre-sized
vstd::HashMap<Key> set;                  // HashSet when V=void
```

Template: `HashMap<K, V=void, Hash=HashValue, Compare=compare<K>, allocType=VEngine>`

### API
```cpp
auto [idx, ok] = map.try_emplace(key, args...);  // insert if absent
auto idx = map.force_emplace(key, args...);       // insert or overwrite
auto idx = map.emplace(key, args...);             // = try_emplace().first
auto idx = map.find(key);
if (idx) { auto& k = idx.key(); auto& v = idx.value(); }
map.remove(key);   // or map.remove(idx), map.remove(it)
map.clear(); map.reserve(n);
map.size(); map.empty(); map.capacity();
```

For `V=void` (sets): `idx.Get()`, `idx->`, `idx*`.

### Iteration
```cpp
for (auto& kv : map) { }             // lvalue: Iterator → NodePair&
for (auto&& kv : std::move(map)) { } // move: MoveIterator → MoveNodePair&&
```

## ArenaHashMap

`#include <luisa/vstl/arena_hash_map.h>`. Arena-backed, **trivially destructible** K/V only.

```cpp
vstd::ArenaHashMap<ArenaType, Key, Value> map(capacity, std::move(arena));
// API: try_emplace, force_emplace, emplace, find, remove, clear, reserve
// No custom Index/remove(Index); key-based removal only.
```

## Object Pool

`#include <luisa/vstl/pool.h>`. Free-list pool using `vengine_malloc`.

```cpp
vstd::Pool<MyType> pool(initial_capacity, initialize=true);
// Pool<T, true>  — trivially destructible, lightweight
// Pool<T, false> — tracks live objects, supports iteration

T* obj = pool.create(args...);
T* obj = pool.create_lock(mtx, args...);  // thread-safe
pool.destroy(obj);
pool.destroy_lock(mtx, obj);
pool.destroy_all();

// Non-trivial only:
for (T* obj : pool.iterator()) { }
```

## Queues

`#include <luisa/vstl/lockfree_array_queue.h>`

### LockFreeArrayQueue
Mostly lock-free circular queue, spin-mutex on resize.

```cpp
vstd::LockFreeArrayQueue<T> q(capacity);
q.enqueue(args...);              // blocking
q.try_push(args...);             // try-lock
auto opt = q.dequeue();          // optional<T>
bool ok = q.pop(&dst);           // pop into pre-constructed T*
auto opt = q.try_pop();          // non-blocking
q.reserve(newCapa); size_t len = q.length();
```

### SingleThreadArrayQueue
SPSC, no locking.

```cpp
vstd::SingleThreadArrayQueue<T> q(capacity);
T* ptr = q.enqueue(args...);     // returns ptr to enqueued item
T* front = q.front();            // peek
auto opt = q.dequeue();
bool ok = q.pop(&dst);
q.pop_discard();
q.reserve(newCapa);
```

## StackAllocator

`#include <luisa/vstl/stack_allocator.h>`

```cpp
vstd::DefaultMallocVisitor visitor;
vstd::StackAllocator alloc(initCapacity, &visitor, expandRate=1.5);
auto chunk = alloc.allocate(size);        // {handle, offset}
auto chunk = alloc.allocate(size, align); // aligned
T* ptr = alloc.allocate_memory<T>();      // typed + zeroed
alloc.clear(); alloc.dispose();
```

## Smart Pointers

`#include <luisa/vstl/unique_ptr.h>`

```cpp
auto p = vstd::make_unique<T>(args...);
auto p = vstd::create_unique(raw_ptr);     // adopts raw, uses vengine_free
auto sp = vstd::make_shared<T>(args...);
auto sp = vstd::create_shared(raw_ptr);
// unique_ptr: if T derives from IDisposable, calls Dispose() on destroy
```

## Variant

`#include <luisa/vstl/meta_lib.h>`. Custom `vstd::variant<...>`.

```cpp
vstd::variant<int, float, std::string> v = 42;
size_t idx = v.index(); bool b = v.is_type_of<int>();
int& i = v.get<0>(); int* p = v.try_get<int>(); int& j = v.force_get<int>();

v.visit([&](auto& x) {});
v.multi_visit([&](int&){}, [&](float&){}, [&](std::string&){});
auto r = v.visit_or(fallback, [](auto& x) { return process(x); });

v.reset_as<int>(123); v.reset_as<2>(args...);  // reset by index
```

## Optional & StackObject

`#include <luisa/vstl/meta_lib.h>`

```cpp
vstd::StackObject<T, false> obj;  // manual lifetime
obj.create(args...); obj.destroy();
T& val = *obj; T* ptr = obj.ptr();

vstd::optional<T> opt(args...);   // = StackObject<T, true>, auto-destroy
if (opt.has_value()) { }
T val = opt.value_or(default_val);
```

## String Utilities

`#include <luisa/vstl/vstring.h>`, `<luisa/vstl/string_builder.h>`

```cpp
// vstd::string
vstd::string s = vstd::to_string(42);
s << value;  // operator<<

// StringBuilder (fixed_vector<char,32>)
vstd::StringBuilder sb;
sb.append("hello"); sb.append(view); sb.append('!'); sb << 42; sb += "suffix";
vstd::string_view v = sb.view();

// StringUtil
for (auto part : vstd::StringUtil::split(str, ',')) { }
vstd::StringUtil::to_lower(s); vstd::StringUtil::to_upper(s);
vstd::StringUtil::to_base64(binary_span, result);
vstd::StringUtil::from_base64(base64_str, byte_vec);
vstd::StringUtil::to_hex_string(binary_span, result, upper=true);
```

## Hash & Compare

`#include <luisa/vstl/hash.h>`, `<luisa/vstl/compare.h>`

```cpp
size_t h = vstd::hash<MyType>{}(value);
int32_t c = vstd::compare<MyType>{}(a, b);  // -1,0,1
// vstd::HashValue delegates to hash<T>; vstd::Hash::binary_hash uses xxHash64
// compare uses memcmp for non-arithmetic non-enum types
```

## Function Reference

`#include <luisa/vstl/functional.h>`

```cpp
vstd::FuncRef<void(int)> cb = vstd::make_func_ref(lambda);
cb(42);
vstd::FuncRef<int(double)> cb2 = &my_c_function;
```

## Ranges

`#include <luisa/vstl/ranges.h>`. One-shot ranges (debug: begin once).

```cpp
for (int64_t i : vstd::range(begin, end, step)) { }
for (int64_t i : vstd::range(end)) { }            // 0..end-1
for (T& x : vstd::ptr_range(ptr, count)) { }
for (auto& x : vstd::ite_range(container)) { }

// Chain
auto r = vstd::make_ite_range(container)
    | filter_range([](auto& x) { return x.active; })
    | transform_range([](auto& x) { return x.value; });

// Erased heap range: i_range()
```

## Others

```cpp
// Scope guard
auto guard = vstd::scope_exit([&] { cleanup(); });

// Macros
KILL_COPY_CONSTRUCT(ClassName)  KILL_MOVE_CONSTRUCT(ClassName)
VSTD_TRIVIAL_COMPARABLE(ClassName)  // memcmp-based == != > <

// Allocation (#include <luisa/vstl/memory.h>)
void* p = vengine_malloc(size); void* p = vengine_realloc(old, size); vengine_free(p);
T* obj = vengine_new<T>(args...); T* arr = vengine_new_array<T>(count, args...); vengine_delete(obj);

// Guid & MD5 (#include <luisa/vstl/v_guid.h>, <luisa/vstl/md5.h>)
vstd::Guid g(true);               // generate new
vstd::Guid g("12345678-...");
auto og = vstd::Guid::TryParseGuid(str);
vstd::MD5 md5(str);
vstd::string s = md5.to_string(upper=true);
```
