#include "ut/ut.hpp"
#include "test_osl_common.h"

using namespace boost::ut;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::test::osl_test;

namespace {

[[nodiscard]] osl::TranslationResult try_translate_single(
    luisa::string_view oso_text,
    osl::OSLTranslator &translator,
    const osl::TranslatorOptions &opts = {}) {
    auto s = osl::OSOParser::parse(oso_text);
    if (!s) {
        osl::TranslationResult r;
        r.errors.emplace_back(osl::TranslationError{.message = "parse failed"});
        return r;
    }
    xir::Module mod;
    auto *fn = mod.create_kernel();
    return translator.translate(*s, fn, opts);
}

}// namespace

int main(int argc, char *argv[]) {
    set_osl_test_args(argc, argv);

    "unsupported_opcode_fails_closed"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader bad_op
oparam	float	f0	0
const	float	$c1	1.0
code ___main___
	nonexistent_opcode	f0 $c1
end
)oso";
        TestOSLTranslator tr;
        auto res = try_translate_single(oso, tr);
        expect(!res.succeeded());
        expect(!res.errors.empty());
    };

    "non_enumerable_dynamic_string_fails_closed"_test = [] {
        // String lattice exceeds max_string_lattice_size=2 via repeated concat -> TOP -> transform fails closed!
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader top_string_fail
oparam	point	v0	0 0 0
global	point	P
global	float	u
local	string	s
local	int	c1
local	int	c2
const	string	$a	"object"
const	string	$b	"shader"
const	string	$c	"camera"
const	string	$d	"common"
const	float	$t1	0.3
const	float	$t2	0.6
code ___main___
	lt		c1 u $t1
	if		c1 4 9
	assign		s $a
	jump		9
	lt		c2 u $t2
	if		c2 8 9
	assign		s $b
	jump		9
	assign		s $c
	transform	v0 s $d P
end
)oso";
        TestOSLTranslator tr;
        osl::TranslatorOptions opts;
        opts.max_string_lattice_size = 2u;// 3 values exceed size 2 -> becomes TOP!
        auto res = try_translate_single(oso, tr, opts);
        expect(!res.succeeded());
    };

    "cartesian_switch_explosion_fails_closed"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader switch_explosion
oparam	point	v0	0 0 0
global	point	P
global	float	u
global	float	v
local	string	s1
local	string	s2
local	int	c1
local	int	c2
const	string	$a	"object"
const	string	$b	"shader"
const	string	$c	"camera"
const	string	$d	"common"
const	float	$half	0.5
code ___main___
	lt		c1 u $half
	if		c1 4 5
	assign		s1 $a
	jump		5
	assign		s1 $b
	lt		c2 v $half
	if		c2 9 10
	assign		s2 $c
	jump		10
	assign		s2 $d
	transform	v0 s1 s2 P
end
)oso";
        TestOSLTranslator tr;
        osl::TranslatorOptions opts;
        opts.max_string_lattice_size = 8u;
        opts.max_string_switch_cases = 2u;// 2 * 2 = 4 cases exceeds limit 2!
        auto res = try_translate_single(oso, tr, opts);
        expect(!res.succeeded());
    };

    "cyclic_group_and_mismatched_connection_fail_closed"_test = [] {
        static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader link_node
param	float	in_f	0
param	string	in_s	"a"
oparam	float	out_f	0
code ___main___
	assign		out_f in_f
end
)oso";
        auto s = osl::OSOParser::parse(oso);
        expect(s != nullptr);

        // Cycle: A -> B -> A
        {
            osl::ShaderGroup g;
            uint32_t a = g.add_layer("A", s.get());
            uint32_t b = g.add_layer("B", s.get());
            g.connect_shaders(a, "out_f", b, "in_f");
            g.connect_shaders(b, "out_f", a, "in_f");
            g.mark_output(b, "out_f");

            TestOSLTranslator tr;
            xir::Module mod;
            auto *fn = mod.create_kernel();
            auto res = tr.translate(g, fn);
            expect(!res.succeeded());
        }

        // Incompatible connection: string -> float
        {
            osl::ShaderGroup g;
            uint32_t a = g.add_layer("A", s.get());
            uint32_t b = g.add_layer("B", s.get());
            g.connect_shaders(a, "in_s", b, "in_f");
            g.mark_output(b, "out_f");

            TestOSLTranslator tr;
            xir::Module mod;
            auto *fn = mod.create_kernel();
            auto res = tr.translate(g, fn);
            expect(!res.succeeded());
        }
    };

    "readonly_global_write_and_unimplemented_base_hooks_fail_closed"_test = [] {
        // Writing read-only global I fails closed
        {
            static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader ro_global
global	vector	I
const	vector	$v	1 0 0
code ___main___
	assign		I $v
end
)oso";
            TestOSLTranslator tr;
            auto res = try_translate_single(oso, tr);
            expect(!res.succeeded());
        }

        // Bare base OSLTranslator fails closed on renderer hooks instead of returning 0
        {
            static constexpr auto oso_tex = R"oso(
OpenShadingLanguage 1.00
shader need_tex_hook
oparam	color	c	0 0 0
global	float	u
global	float	v
const	string	$fn	"ramp.tx"
code ___main___
	texture		c $fn u v
end
)oso";
            osl::OSLTranslator bare;
            auto res = try_translate_single(oso_tex, bare);
            expect(!res.succeeded());
        }
        {
            static constexpr auto oso_closure = R"oso(
OpenShadingLanguage 1.00
surface need_closure_hook
global	normal	N
global	closure color	Ci
const	string	$s	"diffuse"
code ___main___
	closure		Ci $s N
end
)oso";
            osl::OSLTranslator bare;
            auto res = try_translate_single(oso_closure, bare);
            expect(!res.succeeded());
        }
        {
            static constexpr auto oso_trace = R"oso(
OpenShadingLanguage 1.00
shader need_trace_hook
oparam	int	hit	0
global	point	P
global	vector	I
code ___main___
	trace		hit P I
end
)oso";
            osl::OSLTranslator bare;
            auto res = try_translate_single(oso_trace, bare);
            expect(!res.succeeded());
        }
    };

    "intern_collision_parent_child_struct_conn_and_spline_split_fail_closed"_test = [] {
        // 1. Subclass intern_string hash collision between reachable dynamic strings fails closed
        {
            static constexpr auto oso = R"oso(
OpenShadingLanguage 1.00
shader str_collide
oparam	point	v0	0 0 0
global	float	u
global	point	P
local	string	s
local	int	c
const	string	$a	"object"
const	string	$b	"camera"
const	string	$com	"common"
const	float	$half	0.5
code ___main___
	lt		c u $half
	if		c 3 4
	assign		s $a
	jump		4
	assign		s $b
	transform	v0 s $com P
end
)oso";
            class CollidingInternTranslator final : public TestOSLTranslator {
            public:
                [[nodiscard]] uint32_t intern_string(luisa::string_view) noexcept override {
                    return 42u;// intentional collision for all strings
                }
            };
            CollidingInternTranslator tr;
            auto res = try_translate_single(oso, tr);
            expect(!res.succeeded());
        }

        // 2. Conflicting parent struct + child field incoming connections fail closed
        {
            static constexpr auto oso_src = R"oso(
OpenShadingLanguage 1.00
shader s_src
oparam	struct Pair	out_p		%structfields{a,c}
oparam	float	out_p.a	1	%mystruct{out_p} %mystructfield{0}
oparam	color	out_p.c	1 0 0	%mystruct{out_p} %mystructfield{1}
oparam	float	out_f	2
code ___main___
end
)oso";
            static constexpr auto oso_dst = R"oso(
OpenShadingLanguage 1.00
shader s_dst
param	struct Pair	in_p		%structfields{a,c}
param	float	in_p.a	0	%mystruct{in_p} %mystructfield{0}
param	color	in_p.c	0 0 0	%mystruct{in_p} %mystructfield{1}
oparam	float	out_a	0
code ___main___
	assign		out_a in_p.a
end
)oso";
            auto ss = osl::OSOParser::parse(oso_src);
            auto sd = osl::OSOParser::parse(oso_dst);
            expect(ss != nullptr && sd != nullptr);
            if (ss && sd) {
                osl::ShaderGroup g;
                uint32_t ls = g.add_layer("src", ss.get());
                uint32_t ld = g.add_layer("dst", sd.get());
                g.connect_shaders(ls, "out_p", ld, "in_p");
                g.connect_shaders(ls, "out_f", ld, "in_p.a");// conflicting child connection!
                g.mark_output(ld, "out_a");
                TestOSLTranslator tr;
                xir::Module mod;
                auto *fn = mod.create_kernel();
                auto res = tr.translate(g, fn);
                expect(!res.succeeded());
            }
        }

        // 3. Unknown spline basis and invalid knot counts fail closed
        {
            static constexpr auto oso_bad_basis = R"oso(
OpenShadingLanguage 1.00
shader bad_basis
oparam	float	f0	0
global	float	u
const	string	$unk	"unknown_basis"
const	float	$k0	0
const	float	$k1	1
const	float	$k2	2
const	float	$k3	3
code ___main___
	spline		f0 $unk u $k0 $k1 $k2 $k3
end
)oso";
            TestOSLTranslator tr;
            auto res = try_translate_single(oso_bad_basis, tr);
            expect(!res.succeeded());
        }
        {
            static constexpr auto oso_bad_bezier = R"oso(
OpenShadingLanguage 1.00
shader bad_bezier
oparam	float	f0	0
global	float	u
const	string	$bez	"bezier"
const	float	$k	1
code ___main___
	spline		f0 $bez u $k $k $k $k $k
end
)oso";
            TestOSLTranslator tr;
            auto res = try_translate_single(oso_bad_bezier, tr);
            expect(!res.succeeded());
        }
        {
            static constexpr auto oso_dyn_split = R"oso(
OpenShadingLanguage 1.00
shader dyn_split
oparam	int	i0	0
global	float	u
local	string[4]	arr
local	int	ms
const	string	$src	"a,b,c"
const	string	$sep	","
const	float	$ten	10.0
local	float	tmp
code ___main___
	mul		tmp u $ten
	assign		ms tmp
	split		i0 $src arr $sep ms
end
)oso";
            TestOSLTranslator tr;
            auto res = try_translate_single(oso_dyn_split, tr);
            expect(!res.succeeded());
        }
    };

    return 0;
}
