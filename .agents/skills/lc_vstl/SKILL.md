---
name: lc_vstl
---
# LuisaCompute VSTL Container Usage Guide

`vstd` is the V-Engine standard library namespace under `include/luisa/vstl/`. It provides custom containers, allocators, hash maps, queues, smart pointers, and utility types tailored for LuisaCompute.

---

## Namespace & Aliases

All types live in `namespace vstd`. Many are aliases to `luisa::` STL replacements:

| vstd type | Actual type |
|-----------|-------------|
| `vstd::vector<T>` | `luisa::vector<T>` |
| `vstd::fixed_vector<T, N>` | `luisa::fixed_vector<T, N>` |
| `vstd::span<T>` | `luisa::span<T>` |
| `vstd::string` | `std::basic_string<char, ..., luisa::allocator<char>>` |
| `vstd::wstring` | `std::basic_string<wchar_t, ..., luisa::allocator<wchar_t>>` |
| `vstd::string_view` | `luisa::string_view` |
| `vstd::function<T>` | `luisa::move_only_function<T>` |
| `vstd::shared_ptr<T>` | `luisa::shared_ptr<T>` |
| `vstd::spin_mutex` | `luisa::spin_mutex` |
| `vstd::unordered_map` / `vstd::unordered_set` | `luisa::unordered_map` / `luisa::unordered_set` |

---

## Vector Helpers

`#include <luisa/vstl/vector.h>`

### `push_back_func`
Emplace `n` elements into a `vector` or `fixed_vector` using callable(s). The callable may take `size_t index` or no args.

```cpp
vstd::push_back_func(vec, 10, [&](size_t i){ return i * 2; });
vstd::push_back_func(vec, 5, []{ return Foo{}; });
```

### `push_back_all`
Bulk-insert elements from pointer+count, initializer_list, or span.

```cpp
vstd::push_back_all(vec, ptr, n);
vstd::push_back_all(vec, {1, 2, 3});
vstd::push_back_all(vec, some_span);
```

---

## HashMap

`#include <luisa/vstl/hash_map.h>`

Custom hash map using power-of-2 capacity, open addressing with per-bucket red-black trees (`SmallTreeMap`).

```cpp
vstd::HashMap<Key, Value> map;                 // default capacity
vstd::HashMap<Key, Value> map(capacity);       // pre-sized
vstd::HashMap<Key> set;                  // HashSet when V=void
```

Template parameters: `HashMap<K, V=void, Hash=HashValue, Compare=compare<K>, allocType=VEngine>`

### Core Operations

| Method | Returns | Behavior |
|--------|---------|----------|
| `try_emplace(key, args...)` | `std::pair<Index, bool>` | Insert only if key absent |
| `force_emplace(key, args...)` | `Index` | Insert or overwrite existing value |
| `emplace(key, args...)` | `Index` | Same as `try_emplace(...).first` |
| `find(key)` | `Index` | Lookup; empty if missing |
| `remove(key)` | `void` | Erase by key |
| `remove(Index)` | `void` | Erase by index |
| `remove(Iterator)` | `void` | Erase by iterator |
| `clear()` | `void` | Destroy all entries |
| `reserve(n)` | `void` | Grow capacity to next power of 2 >= n |
| `size()` / `empty()` / `capacity()` | — | Stats |

### Index API

```cpp
auto idx = map.find(key);
if (idx) {
    auto& k = idx.key();      // const Key&
    auto& v = idx.value();    // Value&
}
```

For `V=void` (sets), `Index` behaves like a pointer: `idx.Get()`, `idx->`, `idx*`.

### Iteration

```cpp
for (auto& kv : map) { /* kv is pair-like */ }           // lvalue iteration
for (auto&& kv : std::move(map)) { /* move iteration */ } // MoveIterator
```

`Iterator` returns `NodePair&`; `MoveIterator` returns `MoveNodePair&&`.

---

## ArenaHashMap

`#include <luisa/vstl/arena_hash_map.h>`

Arena-backed HashMap for **trivially destructible** keys/values. Uses a single arena allocation for nodes and tree arrays.

```cpp
vstd::ArenaHashMap<ArenaType, Key, Value> map(capacity, std::move(arena));
```

API mirrors `HashMap` (`try_emplace`, `force_emplace`, `emplace`, `find`, `remove`, `clear`, `reserve`). No custom `Index`/`remove(Index)` overloads; only key-based removal.

The `Arena` type must provide `allocate(size_t) -> void*`.

---

## Object Pool

`#include <luisa/vstl/pool.h>`

Fast free-list object pool using `vengine_malloc`.

```cpp
vstd::Pool<MyType> pool(initial_capacity, initialize=true);
```

Specializations:
- `Pool<T, true>` — trivially destructible; lightweight.
- `Pool<T, false>` — tracks live objects; supports iteration over allocated objects.

### Methods

| Method | Description |
|--------|-------------|
| `create(args...)` | Allocate + construct; returns `T*` |
| `create_lock(mtx, args...)` | Thread-safe create |
| `destroy(ptr)` | Destruct + return to pool |
| `destroy_lock(mtx, ptr)` | Thread-safe destroy |
| `destroy_all()` | Return all live objects to free list (non-trivial only destroys on full reset) |

Non-trivial pools also provide:

```cpp
for (T* obj : pool.iterator()) { /* iterate live objects */ }
```

---

## Queues

`#include <luisa/vstl/lockfree_array_queue.h>`

### LockFreeArrayQueue

Mostly lock-free circular queue with spin-mutex fallbacks for resize.

```cpp
vstd::LockFreeArrayQueue<T> q(capacity);
q.enqueue(args...);          // blocking enqueue
q.try_push(args...);         // try-lock enqueue
auto opt = q.dequeue();      // returns optional<T>
bool ok = q.pop(&dst);       // pop into pre-constructed T*
auto opt = q.try_pop();      // non-blocking pop
q.reserve(newCapa);
size_t len = q.length();
```

### SingleThreadArrayQueue

Single-producer single-consumer queue with no locking.

```cpp
vstd::SingleThreadArrayQueue<T> q(capacity);
q.enqueue(args...);          // returns T* to enqueued item
T* front = q.front();        // peek
auto opt = q.dequeue();
bool ok = q.pop(&dst);
q.pop_discard();
q.reserve(newCapa);
```

---

## StackAllocator

`#include <luisa/vstl/stack_allocator.h>`

Stack-like allocator with expandable buffers.

```cpp
vstd::DefaultMallocVisitor visitor;
vstd::StackAllocator alloc(initCapacity, &visitor, expandRate=1.5);

auto chunk = alloc.allocate(size);          // { handle, offset }
auto chunk = alloc.allocate(size, align);   // aligned
T* ptr = alloc.allocate_memory<T>();        // typed + zeroed
alloc.clear();   // reset top
alloc.dispose(); // release all buffers
```

Built-in visitors: `DefaultMallocVisitor`, `VEngineMallocVisitor`.

---

## Smart Pointers

`#include <luisa/vstl/unique_ptr.h>`

```cpp
vstd::unique_ptr<T> p = vstd::make_unique<T>(args...);
vstd::unique_ptr<T> p = vstd::create_unique(raw_ptr);
vstd::shared_ptr<T> sp = vstd::make_shared<T>(args...);
vstd::shared_ptr<T> sp = vstd::create_shared(raw_ptr);
```

`vstd::unique_ptr` uses `vengine_free` by default. If `T` derives from `vstd::IDisposable`, it calls `Dispose()` instead.

---

## Variant

`#include <luisa/vstl/meta_lib.h>`

Custom `vstd::variant<...>` (distinct from `std::variant`).

```cpp
vstd::variant<int, float, std::string> v = 42;
size_t idx = v.index();               // active type index
bool b = v.is_type_of<int>();         // check active type

int& i = v.get<0>();                  // get by index
int* p = v.try_get<int>();            // nullptr if wrong type
int& j = v.force_get<int>();          // unchecked get

v.visit([&](auto& x){ /* ... */ });   // visit active value
v.multi_visit(
    [&](int& x){},
    [&](float& x){},
    [&](std::string& x){}
);

// With fallback:
auto r = v.visit_or(fallback, [](auto& x){ return process(x); });

v.reset_as<int>(123);                 // destroy + reconstruct
v.reset_as<2>(args...);               // reset by index
```

---

## Optional & StackObject

`#include <luisa/vstl/meta_lib.h>`

```cpp
vstd::optional<T> opt;            // uninitialized stack storage
vstd::StackObject<T, false> obj;  // manual lifetime

obj.create(args...);              // construct in place
obj.destroy();                    // explicit destroy
T& val = *obj;
T* ptr = obj.ptr();

vstd::optional<T> opt2(args...);  // constructed on creation
if (opt2.has_value()) { /* ... */ }
T val = opt2.value_or(default_val);
```

`optional<T>` is `StackObject<T, true>` with automatic destruction on scope exit.

---

## String Utilities

`#include <luisa/vstl/vstring.h>` / `<luisa/vstl/string_builder.h>`

### string / wstring

`vstd::string` is a custom-allocated `std::string` alias. Provides:

```cpp
vstd::string s = vstd::to_string(42);
vstd::string s = vstd::to_string(3.14);   // hex float + "f" suffix
s << value;                                // operator<< overload
```

Specializations: `vstd::hash<string>`, `vstd::compare<string>`.

### StringBuilder

Fast append-only string using `fixed_vector<char, 32>`.

```cpp
vstd::StringBuilder sb;
sb.append("hello");
sb.append(view);
sb.append('!');
sb << 42;
sb += "suffix";
vstd::string_view v = sb.view();
```

### StringUtil

```cpp
for (auto part : vstd::StringUtil::split(str, ',')) { }
for (auto part : vstd::StringUtil::split(str, vstd::string_view{"::"})) { }

vstd::StringUtil::to_lower(s);
vstd::StringUtil::to_upper(s);
vstd::StringUtil::to_base64(binary_span, result_string);
vstd::StringUtil::from_base64(base64_str, byte_vector);
vstd::StringUtil::to_hex_string(binary_span, result_string, upper=true);
```

---

## Hash & Compare

`#include <luisa/vstl/hash.h>` / `<luisa/vstl/compare.h>`

```cpp
size_t h = vstd::hash<MyType>{}(value);
int32_t c = vstd::compare<MyType>{}(a, b);   // -1, 0, 1
```

`vstd::HashValue` is a generic functor delegating to `vstd::hash<T>`.
`vstd::Hash::binary_hash(ptr, size)` uses xxHash64.

Specialized for primitives, `std::pair`, `std::tuple`, `string`, `wstring`, `MD5`, `Guid`.

`vstd::compare` uses `memcmp` for non-arithmetic non-enum types by default. `std::pair` and `std::tuple` compare lexicographically.

---

## Function Reference

`#include <luisa/vstl/functional.h>`

Lightweight non-owning callable wrapper:

```cpp
vstd::FuncRef<void(int)> cb = vstd::make_func_ref(lambda);
cb(42);

vstd::FuncRef<int(double)> cb2 = &my_c_function;
```

---

## Ranges

`#include <luisa/vstl/ranges.h>`

One-shot ranges (can only begin once in debug builds):

```cpp
for (int64_t i : vstd::range(begin, end, step)) { }
for (int64_t i : vstd::range(end)) { }           // 0..end-1

for (T& x : vstd::ptr_range(ptr, count)) { }
for (auto& x : vstd::ite_range(container)) { }

// Chain: filter + transform
auto r = vstd::make_ite_range(container)
    | filter_range([](auto& x){ return x.active; })
    | transform_range([](auto& x){ return x.value; });
```

`i_range()` converts a local range to an erased heap range (`IRangeImpl`) for polymorphic passing.

---

## Scope Guard

`#include <luisa/vstl/meta_lib.h>`

```cpp
auto guard = vstd::scope_exit([&]{ cleanup(); });
```

---

## Macros

```cpp
KILL_COPY_CONSTRUCT(ClassName)
KILL_MOVE_CONSTRUCT(ClassName)
VSTD_TRIVIAL_COMPARABLE(ClassName)   // memcmp-based == != > <
```

---

## Allocation

`#include <luisa/vstl/memory.h>`

```cpp
void* p = vengine_malloc(size);
void* p = vengine_realloc(old, size);
vengine_free(p);

T* obj = vengine_new<T>(args...);
T* arr = vengine_new_array<T>(count, args...);
vengine_delete(obj);
```

`vengine_malloc` routes to `luisa::detail::allocator_allocate`. `VAllocHandle<VEngine>` / `VAllocHandle<Default>` provide typed allocator interfaces.

---

## Guid & MD5

`#include <luisa/vstl/v_guid.h>` / `<luisa/vstl/md5.h>`

```cpp
vstd::Guid g(true);               // generate new
vstd::Guid g("12345678-...");
vstd::optional<vstd::Guid> og = vstd::Guid::TryParseGuid(str);

vstd::MD5 md5(str);
vstd::string s = md5.to_string(upper=true);
```

Both have `vstd::hash` and `vstd::compare` specializations.
