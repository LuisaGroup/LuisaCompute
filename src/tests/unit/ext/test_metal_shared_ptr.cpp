#include "ut/ut.hpp"

#include <functional>
#include <type_traits>
#include <utility>
#include "../../../backends/common/metal-cpp/Foundation/NSSharedPtr.hpp"

using namespace boost::ut;

namespace {
// A normal C++ pointee exposes null-this calls without requiring an Apple
// device or relying on Objective-C's unrelated nil-message semantics.
struct Counted {
    unsigned references{1u};
    Counted *retain() noexcept {
        references++;
        return this;
    }
    void release() noexcept { references--; }
};
struct Derived : Counted {};
struct Pipeline {
    NS::SharedPtr<Counted> optional;
    NS::SharedPtr<Counted> active;
};
}// namespace

int main() {
    "empty_shared_ptr_lifetimes"_test = [] {
        NS::SharedPtr<Counted> a;
        NS::SharedPtr<Counted> b{nullptr};
        NS::SharedPtr<Counted> copied{a};
        NS::SharedPtr<Counted> moved{std::move(b)};
        a = copied;
        copied = std::move(moved);
        a.reset();
        auto retained = NS::RetainPtr(static_cast<Counted *>(nullptr));
        expect(!a && !b && !copied && !moved && !retained);
    };
    "empty_shared_ptr_conversions"_test = [] {
        NS::SharedPtr<Derived> derived;
        NS::SharedPtr<Counted> copied{derived};
        NS::SharedPtr<Counted> moved{std::move(derived)};
        copied = derived;
        moved = std::move(derived);
        expect(!derived && !copied && !moved);
    };
    "shared_ptr_reference_accounting"_test = [] {
        Derived object;
        {
            auto owner = NS::TransferPtr(&object);
            NS::SharedPtr<Counted> copied = owner;
            expect(object.references == 2u);
            NS::SharedPtr<Counted> assigned;
            assigned = copied;
            expect(object.references == 3u);
            assigned = std::move(copied); // Same pointee, distinct owners.
            expect(object.references == 2u && !copied);
            assigned.reset();
            expect(object.references == 1u);
            assigned = owner;
            expect(object.references == 2u);
            assigned = std::move(owner); // Converting move, same pointee.
            expect(object.references == 1u && !owner);
            copied = assigned;
            assigned = NS::SharedPtr<Derived>{};
            expect(object.references == 1u && !assigned);
            copied = NS::SharedPtr<Counted>{};
            expect(object.references == 0u);
        }
        expect(object.references == 0u);
    };
    "optional_pipeline_aggregate_return"_test = [] {
        Counted object;
        {
            auto pipeline = [&] {
                NS::SharedPtr<Counted> optional;
                auto active = NS::TransferPtr(&object);
                return Pipeline{std::move(optional), std::move(active)};
            }();
            auto copy = pipeline;
            expect(!copy.optional && object.references == 2u);
            auto moved = std::move(copy);
            expect(!copy.active && moved.active.get() == &object);
        }
        expect(object.references == 0u);
    };
}
