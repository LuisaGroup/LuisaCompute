---
name: lc_core
description: LuisaCompute core library — basic traits/types, binary I/O, clock, dynamic module, allocators, logging, math, fiber, and STL containers
---

# LuisaCompute Core Library (lc_core)

Based on test cases in `src/tests/for_agent/*.cpp`.

## Basic Traits

**Header**: `<luisa/core/basic_traits.h>`

### Type Predicates
```cpp
luisa::always_false_v<T...>           // always false (for static_assert)
luisa::always_true_v<T...>            // always true
// Scalars
luisa::is_integral_v<T>               // is_boolean_v, is_floating_point_v, is_signed_v, is_unsigned_v
luisa::is_signed_integral_v<T>        // is_unsigned_integral_v, is_scalar_v
// Vectors
luisa::is_vector_v<T> / is_vector_v<T,N> / is_vector2_v / is_vector3_v / is_vector4_v
luisa::is_boolean_vector_v<T>         // is_floating_point_vector_v, is_integral_vector_v
luisa::is_signed_integral_vector_v<T> // is_unsigned_integral_vector_v
// Matrices
luisa::is_matrix_v<T> / is_matrix_v<T,N> / is_matrix2_v / is_matrix3_v / is_matrix4_v
// Combined
luisa::is_basic_v<T>                  // scalar||vector||matrix
luisa::is_boolean_or_vector_v<T>      // is_floating_point_or_vector_v, is_integral_or_vector_v
luisa::is_signed_integral_or_vector_v<T> // is_unsigned_integral_or_vector_v
luisa::is_vector_same_dimension_v<T1, T2, ...>
```

### Type Transformations
```cpp
using Elem = luisa::vector_element_t<VecType>;
using Elem = luisa::matrix_element_t<MatType>;
auto val = luisa::to_underlying(Enum::Value);
constexpr size_t dim = luisa::vector_dimension_v<T>;  // 1 for scalars
constexpr size_t dim = luisa::matrix_dimension_v<T>;  // 1 for scalars
```

## Basic Types

**Header**: `<luisa/core/basic_types.h>`

### Aliases
```cpp
// Scalars
luisa::byte(int8_t), ubyte(uint8_t), ushort(uint16_t), uint(uint32_t), ulong(uint64_t), slong(int64_t), half(16-bit)
// Vectors (2/3/4): bool, short, ushort, byte, ubyte, int, uint, slong, ulong, half, float, double
// Matrices (2x2/3x3/4x4): float, double, half
```

### Construction
```cpp
float2 f(1.0f);                // broadcast
int2 i(1, 2);                  // component-wise
auto z = float2::zero();       // (0,0)
auto o = float2::one();        // (1,1)

// Matrix (default = identity)
float2x2 m2;                   // identity
float2x2 m2c(float2(1,2), float2(3,4)); // from cols
auto eye = float2x2::eye(2.0f); // 2*identity
auto fill = float2x2::fill(3.0f);
```

### Element Access
```cpp
float3 v(1,2,3); v.x, v.y, v.z; v[0]; v[1] = 5.0f;
float3x3 m; m[0], m[1], m[2];  // column access
float e = m[col][row];
```

### Operators
```cpp
a+b, a-b, a*b, a/b            // component-wise
a*2.0f, 3.0f*a                // scalar
-a, +a                         // unary
~i, i<<1                       // bitwise (integral)
a==b, a<b                      // comparison → bool vector
b1||b2, b1&&b2                 // bool logic
any(b), all(b), none(b)        // bool vector reduce

// Matrix
m*2.0f, 3.0f*m, m/2.0f        // scalar
m * v                          // matrix-vector
a * b                          // matrix-matrix
a + b, a - b                   // element-wise
```

### Make Functions
```cpp
make_float2(1.0f);                        // broadcast
make_float2(1.0f, 2.0f);                 // components
make_float2(float3(1,2,3));              // from larger vec
make_float3(float2(1,2), 3.0f);          // vec + scalar
make_float3(1.0f, float2(2,3));          // scalar + vec
make_float4(float2(1,2), float2(3,4));   // vec + vec

make_float2x2(2.0f);                      // diagonal fill
make_float2x2(1,2,3,4);                  // row-major
make_float2x2(float2(1,2), float2(3,4)); // columns
make_float3x3(1,2,3, 4,5,6, 7,8,9);     // row-major
make_float4x4(1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16);
```

## Binary File Stream

**Header**: `<luisa/core/binary_file_stream.h>`. Use instead of `std::ifstream` for binary I/O.

```cpp
luisa::BinaryFileStream stream("file.bin");
if (stream.valid()) { /* or operator bool */ }
size_t len = stream.length(), pos = stream.pos();
stream.set_pos(128); stream.set_pos(0);
stream.read(luisa::span<std::byte>(buf.data(), buf.size()));
stream.close();
luisa::BinaryFileStream s2(std::move(stream));  // move semantics
```

## Binary IO (BinaryBlob)

**Header**: `<luisa/core/binary_io.h>`

```cpp
luisa::BinaryBlob blob{ptr, size, [](void* p) { ::operator delete(p); }};
luisa::BinaryBlob empty;
std::byte* d = blob.data(); size_t sz = blob.size(); bool e = blob.empty();
luisa::span<std::byte> sp = static_cast<luisa::span<std::byte>>(blob);
luisa::BinaryBlob b2(std::move(blob)); b3 = std::move(b2);
void* raw = blob.release();  // blob becomes empty; manual delete required
```

## Clock

**Header**: `<luisa/core/clock.h>`

```cpp
luisa::Clock clock;           // starts timing on construction
clock.tic();                  // reset
double ms = clock.toc();      // elapsed ms since tic (does NOT reset)
double t1 = clock.toc();      // cumulative
```

## Dynamic Module

**Header**: `<luisa/core/dynamic_module.h>`

```cpp
auto mod = luisa::DynamicModule::load("name");           // platform-specific
auto mod = luisa::DynamicModule::load("/path", "name");  // from dir
auto mod = luisa::DynamicModule::load_exact("/path/lib.so");
if (mod) { void* h = mod.handle(); }
void* addr = mod.address("fn");
auto* fn = mod.function<MyFunc>("fn");
void* raw = mod.release(); luisa::dynamic_module_destroy(raw);
mod.reset();

// Search paths
luisa::DynamicModule::add_search_path(dir);
luisa::DynamicModule::remove_search_path(dir);
```

## First Fit Allocator

**Header**: `<luisa/core/first_fit.h>`

```cpp
luisa::FirstFit alloc(1024, 8);          // size, alignment
luisa::FirstFit::Node* n = alloc.allocate(100);
if (n) { size_t off = n->offset(), sz = n->size(); }
n = alloc.allocate_best_fit(100);
alloc.free(n);
size_t sz = alloc.size(), align = alloc.alignment();
luisa::string fl = alloc.dump_free_list();
```

## Logging

**Header**: `<luisa/core/logging.h>`

```cpp
luisa::log_level_verbose();  // log_level_info(), log_level_warning(), log_level_error()
luisa::log_flush();

// Function-style
luisa::log_verbose("msg"); luisa::log_info("v: {}", 42); luisa::log_warning("warn");

// Macro-style (recommended)
LUISA_VERBOSE("msg"); LUISA_INFO("v: {}, {}", 1, 2); LUISA_WARNING("warn");
LUISA_VERBOSE_WITH_LOCATION("dbg: {}", val);
LUISA_INFO_WITH_LOCATION("proc: {}", name);
LUISA_WARNING_WITH_LOCATION("deprecated: {}", api);
```

Format: `{}` (default), `{:x}` (hex), `{:b}` (binary), `{:e}` (scientific), `{:.2f}` (fixed precision).

## Mathematics

**Header**: `<luisa/core/mathematics.h>`

### Scalar
```cpp
luisa::next_pow2(100u);          // 128
luisa::fract(3.7f);              // 0.7 (-3.7f → 0.3)
luisa::radians(180.0f);          // pi
luisa::degrees(constants::pi);   // 180
luisa::sin(x), cos(x), sqrt(x), abs(x), min(a,b), max(a,b)
```

### Vector (component-wise)
```cpp
luisa::sin(v2), cos(v2), sqrt(v2), abs(v2), floor(v2), ceil(v2), fract(v2)
luisa::min(a, b), max(a, b), pow(a, b), atan2(y, x), fmod(a, b)
luisa::min(2.0f, a)              // scalar-vector
luisa::isnan(v2), isinf(v2)

luisa::dot(a, b);                // scalar result
luisa::length(a);                // sqrt(dot(a,a))
luisa::distance(a, b);
luisa::normalize(a);             // a/length(a)
luisa::cross(c, d);              // 3D only
```

### Matrix
```cpp
luisa::transpose(m);             // 2x2/3x3/4x4
luisa::inverse(m);
luisa::determinant(m);

// Transformations (float4x4)
luisa::translation(1,2,3);      // or translation(float3)
luisa::scaling(2,3,4);          // non-uniform; scaling(5.0f) = uniform
luisa::rotation(axis_float3, angle_rad);
```

### Interpolation & Selection
```cpp
luisa::select(false_val, true_val, cond);          // scalar/vector
luisa::lerp(a, b, t);                               // a+(b-a)*t (scalar/vector)
luisa::clamp(v, lo, hi);                            // scalar/vector
luisa::sign(x);                                     // 1.0/-1.0 (scalar/vector, float/int)
luisa::fma(a, b, c);                                // a*b+c
```

### Constants
```cpp
luisa::constants::pi, pi_over_2, pi_over_4, two_pi, inv_pi, e
```

## Pool Allocator

**Header**: `<luisa/core/pool.h>`

```cpp
luisa::Pool<MyClass> pool;                  // thread-safe
luisa::Pool<MyClass, false> pool_nt;        // non-thread-safe
MyClass* obj = pool.allocate();             // raw (no ctor)
pool.deallocate(obj);
MyClass* obj2 = pool.create();              // default-construct
MyClass* obj3 = pool.create(args...);       // construct with args
pool.destroy(obj2);
```

## Fiber

**Header**: `<luisa/core/fiber.h>`. Built on marl.

### Scheduler
```cpp
luisa::fiber::scheduler sched;     // all cores
luisa::fiber::scheduler sched(4);  // fixed threads
// RAII: binds on construction, unbinds on destruction
```

### Sync Primitives
```cpp
luisa::fiber::event evt(luisa::fiber::event::Mode::Manual, false);
evt.signal(); evt.clear(); evt.wait(); bool r = evt.test()/evt.is_signalled();

luisa::fiber::counter cnt(3);
cnt.add(2); cnt.done(); cnt.wait();

luisa::fiber::mutex mtx;
luisa::fiber::lock lck(mtx);
luisa::fiber::condition_variable cv;
```

### Tasks
```cpp
luisa::fiber::schedule([]() noexcept { /* work */ });
auto evt = luisa::fiber::async([]() noexcept { return 42; }); evt.wait();
```

### Parallel For
```cpp
// Blocking
luisa::fiber::parallel(100, [](uint32_t i) noexcept {});
luisa::fiber::parallel(100, [](uint32_t begin, uint32_t end) noexcept {});
// Async
auto cnt = luisa::fiber::async_parallel(100, [](uint32_t i) noexcept {}); cnt.wait();
// Iterator
luisa::fiber::parallel(data.begin(), data.end(), 64, [](auto l, auto r) {});
auto cnt = luisa::fiber::async_parallel(data.begin(), data.end(), 64, [](auto l, auto r) {}); cnt.wait();
// With external counter
luisa::fiber::counter cnt(0); luisa::fiber::async_parallel(cnt, 100, [](uint32_t i){});
// Control batch size
luisa::fiber::parallel(1000, []{}, /*internal_jobs=*/10);

uint32_t n = luisa::fiber::worker_thread_count();
```

### Defer
```cpp
{ luisa_fiber_defer(printf("world\n")); printf("hello "); }
```

## STL Containers & Utilities

**Location**: `include/luisa/core/stl/`. Uses `std::` or EASTL based on `LUISA_USE_SYSTEM_STL`. All in `luisa` namespace.

### Memory
**Header**: `<luisa/core/stl/memory.h>`
```cpp
luisa::allocator<T> alloc;
auto sz1 = 64_k;  // 65536 (also 16_M, 2_G)
luisa::unique_ptr<T> up = luisa::make_unique<T>(args...);
luisa::shared_ptr<T> sp = luisa::make_shared<T>(args...);
luisa::weak_ptr<T> wp = sp;
luisa::span<T> s(data, count);
T* p = luisa::allocate_with_allocator<T>(n); luisa::deallocate_with_allocator(p);
T* obj = luisa::new_with_allocator<T>(args...); luisa::delete_with_allocator(obj);
auto u = luisa::bit_cast<uint32_t>(3.14f);
```

### Strings & Format
**Header**: `<luisa/core/stl/string.h>`, `<luisa/core/stl/format.h>`
```cpp
luisa::string s = "hello"; luisa::u8string u8s = u8"hello"; luisa::wstring ws = L"hello";
luisa::string_hash h; uint64_t hash = h("hello");
luisa::string s = luisa::format("Value: {}, {}", 42, 3.14);
auto s2 = luisa::to_string(float3(1,2,3));
auto hex = luisa::hash_to_string(0x1234ABCD);
```

### Containers
```cpp
// <stl/vector.h>
luisa::vector<int> vec = {1,2,3}; vec.push_back(4);
luisa::fixed_vector<int, 64> fvec;
luisa::bitvector bits(100);
auto* raw = luisa::enlarge_by(vec, 10);  // push 10 uninit, return ptr to first
luisa::vector_resize(vec, 100);
size_t bytes = luisa::size_bytes(vec);

// <stl/unordered_map.h> — dense hash map (faster than std)
luisa::unordered_map<string, int> map; map.emplace("key", 42);
luisa::unordered_set<int> set;

// <stl/map.h> — ordered
luisa::map<string,int> omap; luisa::set<int> oset; luisa::multimap<string,int> mm; luisa::multiset<int> ms;

// <stl/fixed_map.h> — fixed-capacity
luisa::fixed_map<int,string,64> fmap; luisa::fixed_set<int,64> fset;
luisa::fixed_unordered_map<int,string,64> fumap; luisa::fixed_unordered_set<int,64> fus;
luisa::fixed_multimap<int,string,64> fmm; luisa::fixed_multiset<int,64> fms;

// <stl/vector_map.h> — sorted-vector-based (better cache locality)
luisa::vector_map<int,string> vm; luisa::vector_set<int> vs;
luisa::vector_multimap<int,string> vmm; luisa::vector_multiset<int> vms;

// <stl/deque.h>, <stl/queue.h>, <stl/stack.h>, <stl/priority_queue.h>
luisa::deque<int> dq; luisa::queue<int> q; luisa::stack<int> st; luisa::priority_queue<int> pq;

// <stl/list.h>
luisa::list<int> lst; luisa::forward_list<int> flst;
luisa::fixed_list<int,64> fl; luisa::fixed_forward_list<int,64> ffl;

// <stl/ring_buffer.h> (EASTL only)
luisa::ring_buffer<int> rb; luisa::fixed_ring_buffer<int,64> frb;
```

### LRU Cache
**Header**: `<luisa/core/stl/lru_cache.h>`
```cpp
luisa::lru_cache<string, int> cache(100); cache.emplace("key", 42);
auto val = cache.at("key");  // luisa::optional<int>
cache.touch("key");
auto tc = luisa::LRUCache<string, int>::create(100);  // thread-safe
tc->set_delete_callback([](const int &v) {});
auto v = tc->fetch("key"); tc->update("key", 42);
```

### Optional & Variant
**Header**: `<luisa/core/stl/optional.h>`, `<luisa/core/stl/variant.h>`
```cpp
luisa::optional<int> opt = 42; if (opt) { int v = *opt; }
auto o = luisa::make_optional(3.14); auto n = luisa::nullopt;

luisa::variant<int,float,string> v = 3.14f;
if (luisa::holds_alternative<float>(v)) {}
auto f = luisa::get<float>(v);
auto p = luisa::get_if<int>(&v);
luisa::visit([](auto&& x){}, v);
```

### Functional
**Header**: `<luisa/core/stl/functional.h>`
```cpp
luisa::function<void(int)> fn = [](int){};
luisa::move_only_function<void(int)> mfn = [p=make_unique<int>()](int){};
luisa::less<> lt; luisa::equal_to<> eq; luisa::greater<> gt;
auto visitor = luisa::make_overloaded([](int i){ return "int"; }, [](float f){ return "float"; });
auto obj = luisa::lazy_construct([]{ return make_unique<Resource>(); });
auto guard = luisa::make_finally([]{ cleanup(); });  // EASTL only
```

### Hashing
**Header**: `<luisa/core/stl/hash.h>`
```cpp
uint64_t h = luisa::hash_value(42);
uint64_t hc = luisa::hash_combine({h1, h2, h3});
luisa::Hash128 h128 = luisa::hash128(data, size, seed);
luisa::string s = h128.to_string();
```

### Iterators & Algorithms
**Header**: `<luisa/core/stl/iterator.h>`, `<luisa/core/stl/algorithm.h>`
```cpp
for (auto i : luisa::range(10)) {}          // 0..9
for (auto i : luisa::range(2, 10)) {}       // 2..9
for (auto i : luisa::range(0, 10, 2)) {}    // 0,2,4,6,8

luisa::sort(vec.begin(), vec.end());         // pdqsort
luisa::sort(vec.begin(), vec.end(), luisa::greater<>{});
luisa::transform(a.begin(), a.end(), b.begin(), op);
bool found = luisa::binary_search(vec.begin(), vec.end(), val);

#include <luisa/core/stl/pdqsort.h>
pdqsort(vec.begin(), vec.end());
pdqsort_branchless(vec.begin(), vec.end());
```

### Other
```cpp
// <stl/filesystem.h>
luisa::filesystem::path p = "/some/path"; luisa::string s = luisa::to_string(p);
// <stl/sstream.h>
luisa::stringstream ss; ss << "value=" << 42; luisa::string s = ss.str();
luisa::ostringstream oss; luisa::istringstream iss("42 3.14");
```

## Summary

| Component | Header | Key |
|---|---|---|
| Basic Traits | `<luisa/core/basic_traits.h>` | `is_vector_v`, `is_matrix_v`, `vector_element_t`, `to_underlying` |
| Basic Types | `<luisa/core/basic_types.h>` | `float2/3/4`, `float2x2/3x3/4x4`, `make_float2/3/4`, `make_float2x2/3x3/4x4` |
| Binary File Stream | `<luisa/core/binary_file_stream.h>` | `BinaryFileStream` |
| Binary IO | `<luisa/core/binary_io.h>` | `BinaryBlob` |
| Clock | `<luisa/core/clock.h>` | `Clock::tic()`, `Clock::toc()` |
| Dynamic Module | `<luisa/core/dynamic_module.h>` | `DynamicModule::load()`, `address()`, `function<>()` |
| First Fit | `<luisa/core/first_fit.h>` | `FirstFit::allocate()`, `allocate_best_fit()`, `free()` |
| Logging | `<luisa/core/logging.h>` | `LUISA_INFO()`, `LUISA_WARNING()`, `log_level_info()` |
| Mathematics | `<luisa/core/mathematics.h>` | `sin()`, `dot()`, `normalize()`, `transpose()`, `inverse()`, `lerp()`, `clamp()` |
| Pool | `<luisa/core/pool.h>` | `Pool<T>::allocate()`, `create()`, `destroy()` |
| Fiber | `<luisa/core/fiber.h>` | `scheduler`, `schedule()`, `async()`, `parallel()`, `async_parallel()`, `event`, `counter` |
| STL Memory | `<luisa/core/stl/memory.h>` | `allocator`, `unique_ptr`, `shared_ptr`, `span`, `make_unique`, `make_shared` |
| STL String | `<luisa/core/stl/string.h>` | `string`, `string_hash`, `u8string`, `wstring` |
| STL Format | `<luisa/core/stl/format.h>` | `format()`, `to_string()`, `hash_to_string()` |
| STL Vector | `<luisa/core/stl/vector.h>` | `vector`, `fixed_vector`, `bitvector`, `enlarge_by`, `vector_resize` |
| STL Map | `<luisa/core/stl/unordered_map.h>` | `unordered_map`, `unordered_set` |
| STL Algorithm | `<luisa/core/stl/algorithm.h>` | `sort`, `transform`, `binary_search`, `pdqsort` |
| STL Functional | `<luisa/core/stl/functional.h>` | `function`, `move_only_function`, `overloaded`, `lazy_construct` |
| STL Hash | `<luisa/core/stl/hash.h>` | `hash_value`, `hash_combine`, `Hash128` |
| STL LRU Cache | `<luisa/core/stl/lru_cache.h>` | `lru_cache`, `LRUCache` |
