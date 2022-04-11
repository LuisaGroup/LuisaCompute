#include <serialize/serialize.h>
#include <serialize/traits.h>
#include <dsl/var.h>
#include <dsl/stmt.h>
#include <dsl/func.h>
#include <dsl/builtin.h>
#include <dsl/syntax.h>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace luisa::compute;

enum En{
    Q,W,E,R
};

struct A;

struct A{
    int x,y,z;
    En e;

    enum Tag {
        Base, Derived
    };

    Tag tag = Base;

    virtual void serialize(Serializer& s){
        s.serialize(MAKE_NAME_PAIR(x),MAKE_NAME_PAIR(y),MAKE_NAME_PAIR(z), MAKE_NAME_PAIR(e), MAKE_NAME_PAIR(tag));
    }

    virtual void serialize(Deserializer& s){
        s.serialize(MAKE_NAME_PAIR(x),MAKE_NAME_PAIR(y),MAKE_NAME_PAIR(z), MAKE_NAME_PAIR(e), MAKE_NAME_PAIR(tag));
    }

    using is_polymorphically_serialized = int;
    using polymorphic_tag_type = Tag;

    static luisa::unique_ptr<A> create(Tag t);
};

struct D : public A{
    int ddd = 23333;

    D() {
        tag = Derived;
    }

    virtual void serialize(Serializer& s) override{
        A::serialize(s);
        s.serialize(MAKE_NAME_PAIR(ddd));
    }

    virtual void serialize(Deserializer& s) override{
        A::serialize(s);
        s.serialize(MAKE_NAME_PAIR(ddd));
    }

};

luisa::unique_ptr<A> A::create(Tag t) {
    if(t == Base) return luisa::make_unique<A>();
    else return luisa::make_unique<D>();
}

struct B{
    int m,n;
    luisa::unique_ptr<A> a;
    luisa::variant<int, std::string> q, w;
    luisa::vector<int> data;

    template<typename S>
    void serialize(S& s){
        s.serialize(
            MAKE_NAME_PAIR(m), 
            MAKE_NAME_PAIR(n), 
            MAKE_NAME_PAIR(a), 
            MAKE_NAME_PAIR(data), 
            MAKE_NAME_PAIR(q),
            MAKE_NAME_PAIR(w)
        );
    }
};

void testSerialize() {
    B b;
    b.m = 12;
    b.n = 15;
    b.a = std::move(luisa::make_unique<D>());
    b.a->x = 1;
    b.a->y = 13;
    b.a->z = 15;
    b.a->e = R;
    b.q = 900;
    b.w = "qwer";
    b.data.push_back(1);
    b.data.push_back(2);
    b.data.push_back(3);
    b.data.push_back(4);
    Serializer ser;
    ser.serialize(MAKE_NAME_PAIR(b));
    std::cout << ser.data().dump(4) << std::endl;
}

void testDeserialize() {
    B bb;
    nlohmann::json jj = R"({"b":{"a":{"tag":1,"ddd":233,"x":1,"y":13,"z":15,"e":2},"data":[1,2,3,4],"m":12,"n":15,"q":{"index":1,"value":"qqq"},"w":{"index":0,"value":12333}}})"_json;
    std::cout << jj << std::endl;
    Deserializer dser(jj);
    dser.serialize(KeyValuePair{"b", bb});
    Serializer sser;
    sser.serialize(MAKE_NAME_PAIR(bb));
    std::cout << sser.data().dump(4) << std::endl;
}

void testKernel(){
    Kernel1D kernel = [](Float3 linear) noexcept {
        auto t = thread_id();
        Constant<int> c = {1, 2, 3};
        auto yy = c[0];
        auto x = linear.xyz();
        auto srgb = make_uint3(
            round(saturate(
                      select(1.055f * pow(x, 1.0f / 2.4f) - 0.055f,
                             12.92f * x,
                             x <= 0.00031308f)) *
                  255.0f));
    };
    Serializer ser;
    ser.serialize(MAKE_NAME_PAIR(kernel));
    std::cout << ser.data() << std::endl;
    Kernel1D kernel2 = [](){};
    Deserializer dser(ser.data());
    dser.serialize(KeyValuePair{"kernel", kernel2});
    Serializer ser2;
    ser2.serialize(MAKE_NAME_PAIR(kernel2));
    std::cout << ser2.data() << std::endl;

}

int main() {
    // testDeserialize();
    // testSerialize();
    testKernel();
}