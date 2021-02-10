//
// Created by Mike Smith on 2021/2/10.
//

#include <iostream>
#include <sstream>

struct Var;

struct Expr {
    virtual void print(std::ostream &) const noexcept = 0;
};

std::ostream &operator<<(std::ostream &os, const Expr &expr) noexcept {
    expr.print(os);
    return os;
}

struct ValueExpr : public Expr {
    float value;
    explicit ValueExpr(float v) noexcept : value{v} {}
    void print(std::ostream &os) const noexcept override { os << "DSL(" << value << ")"; }
};

struct MulExpr : public Expr {
    const Var *lhs;
    const Var *rhs;
    MulExpr(const Var *lhs, const Var *rhs) noexcept : lhs{lhs}, rhs{rhs} {}
    void print(std::ostream &ostream) const noexcept override;
};

struct Var {
    
    const Expr *expr{nullptr};
    float value{};

    constexpr Var(float x) noexcept : value{x} {}
    Var(const MulExpr *expr) noexcept : expr{expr} {}
    
    [[nodiscard]] constexpr bool is_cpu() const noexcept { return expr == nullptr; }

    [[nodiscard]] constexpr auto operator*(const Var &rhs) const noexcept {
        if (is_cpu() && rhs.is_cpu()) { return Var{value * rhs.value}; }
        return Var{new MulExpr(this, &rhs)};
    }

    friend std::ostream &operator<<(std::ostream &os, const Var &v) noexcept {
        if (v.is_cpu()) { os << "CPU(" << v.value << ")"; }
        else { os << *v.expr; }
        return os;
    }
};

void MulExpr::print(std::ostream &os) const noexcept { os << "(" << *lhs << " * " << *rhs << ")"; }

auto foo(const Var &a, const Var &b, const Var &c) noexcept {
    return a * b * c;
}

int main() {
    float a, b, c;
    std::cin >> a >> b >> c;
    std::cout << "foo: " << foo(a, b, c) << std::endl;
}
