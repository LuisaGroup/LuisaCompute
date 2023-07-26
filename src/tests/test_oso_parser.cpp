#include <cstdio>

#include <luisa/osl/shader.h>
#include <luisa/osl/oso_parser.h>

int main(int argc, char *argv[]) {

    luisa::unique_ptr<luisa::compute::osl::Shader> shader;
    if (argc < 2) {
        static constexpr auto code = R"oso(
OpenShadingLanguage 1.00
# Compiled by oslc 1.13.4.0dev
# options: -q -I/Users/mike/ClionProjects/OpenShadingLanguage/src/shaders -I/Users/mike/ClionProjects/OpenShadingLanguage/src/shaders -I/Users/mike/ClionProjects/OpenShadingLanguage/src/shaders -o /Users/mike/ClionProjects/OpenShadingLanguage/cmake-build-release/src/shaders/mandelbrot.oso
surface mandelbrot
param	point	center	0 0 0		%meta{string,help,"Center point where uv=(0.5,0.5)"}  %read{0,0} %write{2147483647,-1}
        param	float	scale	2		%meta{string,help,"Scale of mapping from uv to complex space"} %meta{float,min,9.99999997e-07} %meta{float,max,10}  %read{7,7} %write{2147483647,-1}
        param	int	iters	100		%meta{string,help,"Maximum iterations"} %meta{int,min,1} %meta{int,max,1000000}  %read{13,46} %write{2147483647,-1}
        param	int	julia	0		%meta{string,help,"If nonzero, compute Julia set"}  %read{23,23} %write{2147483647,-1}
        param	point	P_julia	0 0 0		%meta{string,help,"If julia is nonzero, this is the Julia offset"}  %read{30,30} %write{2147483647,-1}
        param	color[]	colormap	0 0 0.00999999978 0 0 0.00999999978 0 0 0.5 0.75 0.25 0 0.949999988 0.949999988 0 1 1 1 1 1 1		%meta{string,help,"Color map for visualizing result"}  %read{51,51} %write{2147483647,-1}
        oparam	float	fout	0		%meta{string,help,"Output: number of iterations"}  %read{2147483647,-1} %write{41,41}
        oparam	color	Cout	0 0 0		%meta{string,help,"Output: color mapped result"}  %read{2147483647,-1} %write{51,52}
        global	float	u	%read{1,1} %write{2147483647,-1}
        global	float	v	%read{3,3} %write{2147483647,-1}
        local	point	cent	%read{8,8} %write{0,0}
        local	point	c	%read{9,38} %write{8,8}
        local	point	z	%read{13,38} %write{9,38}
        local	point	iota	%read{2147483647,-1} %write{10,10}
        local	int	i	%read{13,42} %write{12,38}
        local	float	___345_x	%read{13,38} %write{13,38}
        local	float	___345_y	%read{13,38} %write{13,38}
        local	float	___348_f	%read{51,51} %write{50,50}
        temp	point	$tmp1	%read{7,7} %write{6,6}
    const	float	$const2	0.5		%read{1,4} %write{2147483647,-1}
        temp	float	$tmp2	%read{2,2} %write{1,1}
        temp	float	$tmp3	%read{6,6} %write{2,2}
    const	float	$const3	2		%read{2,34} %write{2147483647,-1}
    const	int	$const4	1		%read{12,38} %write{2147483647,-1}
        temp	float	$tmp4	%read{4,4} %write{3,3}
    const	float	$const5	1		%read{3,49} %write{2147483647,-1}
        temp	float	$tmp5	%read{5,5} %write{4,4}
        temp	float	$tmp6	%read{6,6} %write{5,5}
    const	int	$const6	0		%read{14,52} %write{2147483647,-1}
    const	float	$const7	0		%read{6,36} %write{2147483647,-1}
        temp	point	$tmp7	%read{8,8} %write{7,7}
    const	point	$const8	1 2 0		%read{10,10} %write{2147483647,-1}
        temp	int	$tmp8	%read{13,38} %write{13,38}
        temp	int	$tmp9	%read{13,38} %write{13,38}
        temp	float	$tmp10	%read{13,38} %write{13,38}
    const	float	$const9	4		%read{17,17} %write{2147483647,-1}
        temp	int	$tmp11	%read{13,38} %write{13,38}
        temp	int	$tmp12	%read{13,38} %write{13,38}
        temp	int	$tmp13	%read{11,38} %write{12,38}
        temp	point	$tmp14	%read{13,38} %write{13,38}
        temp	float	$tmp15	%read{13,38} %write{13,38}
        temp	float	$tmp16	%read{13,38} %write{13,38}
        temp	float	$tmp17	%read{13,38} %write{13,38}
        temp	float	$tmp18	%read{13,38} %write{13,38}
        temp	float	$tmp19	%read{13,38} %write{13,38}
        temp	point	$tmp20	%read{13,38} %write{13,38}
        temp	float	$tmp21	%read{13,38} %write{13,38}
        temp	float	$tmp22	%read{13,38} %write{13,38}
        temp	float	$tmp23	%read{13,38} %write{13,38}
        temp	float	$tmp24	%read{13,38} %write{13,38}
        temp	float	$tmp25	%read{13,38} %write{13,38}
        temp	int	$tmp26	%read{40,40} %write{39,39}
        temp	float	$tmp27	%read{45,45} %write{43,43}
        temp	float	$tmp28	%read{43,43} %write{42,42}
        temp	float	$tmp29	%read{50,50} %write{45,45}
        temp	float	$tmp30	%read{45,45} %write{44,44}
        temp	float	$tmp31	%read{49,49} %write{48,48}
        temp	float	$tmp32	%read{48,48} %write{47,47}
        temp	float	$tmp33	%read{47,47} %write{46,46}
        temp	float	$tmp34	%read{50,50} %write{49,49}
    const	string	$const10	"linear"		%read{51,51} %write{2147483647,-1}
        code ___main___
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:36
#     point cent = center;
            assign		cent center 	%filename{"/Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl"} %line{36} %argrw{"wr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:37
#     point c = scale * point(2*(u-0.5), 2*((1-v)-0.5), 0) + cent;
        sub		$tmp2 u $const2 	%line{37} %argrw{"wrr"}
        mul		$tmp3 $const3 $tmp2 	%argrw{"wrr"}
        sub		$tmp4 $const5 v 	%argrw{"wrr"}
        sub		$tmp5 $tmp4 $const2 	%argrw{"wrr"}
        mul		$tmp6 $const3 $tmp5 	%argrw{"wrr"}
        point		$tmp1 $tmp3 $tmp6 $const7 	%argrw{"wrrr"}
        mul		$tmp7 scale $tmp1 	%argrw{"wrr"}
        add		c $tmp7 cent 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:38
#     point z = c;
        assign		z c 	%line{38} %argrw{"wr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:39
#     point iota = point (1, 2, 0);
        assign		iota $const8 	%line{39} %argrw{"wr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:41
#     for (i = 1; i < iters && dot(z,z) < 4.0; ++i) {
    for		$tmp13 13 21 38 39 	%line{41} %argrw{"r"}
            assign		i $const4 	%argrw{"wr"}
            lt		$tmp8 i iters 	%argrw{"wrr"}
            neq		$tmp9 $tmp8 $const6 	%argrw{"wrr"}
    if		$tmp9 20 20 	%argrw{"r"}
            dot		$tmp10 z z 	%argrw{"wrr"}
            lt		$tmp11 $tmp10 $const9 	%argrw{"wrr"}
            neq		$tmp12 $tmp11 $const6 	%argrw{"wrr"}
            assign		$tmp9 $tmp12 	%argrw{"wr"}
            neq		$tmp13 $tmp9 $const6 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:47
#         float x = z[0], y = z[1];
            compref		___345_x z $const6 	%line{47} %argrw{"wrr"}
            compref		___345_y z $const4 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:48
#         if (julia) {
    if		julia 31 38 	%line{48} %argrw{"r"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:49
#             z = point (x*x - y*y, 2*x*y, 0) + P_julia;
            mul		$tmp15 ___345_x ___345_x 	%line{49} %argrw{"wrr"}
            mul		$tmp16 ___345_y ___345_y 	%argrw{"wrr"}
            sub		$tmp17 $tmp15 $tmp16 	%argrw{"wrr"}
            mul		$tmp18 $const3 ___345_x 	%argrw{"wrr"}
            mul		$tmp19 $tmp18 ___345_y 	%argrw{"wrr"}
            point		$tmp14 $tmp17 $tmp19 $const7 	%argrw{"wrrr"}
            add		z $tmp14 P_julia 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:51
#             z = point (x*x - y*y, 2*x*y, 0) + c;
            mul		$tmp21 ___345_x ___345_x 	%line{51} %argrw{"wrr"}
            mul		$tmp22 ___345_y ___345_y 	%argrw{"wrr"}
            sub		$tmp23 $tmp21 $tmp22 	%argrw{"wrr"}
            mul		$tmp24 $const3 ___345_x 	%argrw{"wrr"}
            mul		$tmp25 $tmp24 ___345_y 	%argrw{"wrr"}
            point		$tmp20 $tmp23 $tmp25 $const7 	%argrw{"wrrr"}
            add		z $tmp20 c 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:41
#     for (i = 1; i < iters && dot(z,z) < 4.0; ++i) {
            add		i i $const4 	%line{41} %argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:54
#     if (i < iters) {
            lt		$tmp26 i iters 	%line{54} %argrw{"wrr"}
    if		$tmp26 52 53 	%argrw{"r"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:55
#         fout = i;
	assign		fout i 	%line{55} %argrw{"wr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:56
#         float f = pow(float(i)/iters, 1/log10(float(iters)));
	assign		$tmp28 i 	%line{56} %argrw{"wr"}
	assign		$tmp27 $tmp28 	%argrw{"wr"}
	assign		$tmp30 iters 	%argrw{"wr"}
	div		$tmp29 $tmp27 $tmp30 	%argrw{"wrr"}
	assign		$tmp33 iters 	%argrw{"wr"}
	assign		$tmp32 $tmp33 	%argrw{"wr"}
	log10		$tmp31 $tmp32 	%argrw{"wr"}
	div		$tmp34 $const5 $tmp31 	%argrw{"wrr"}
	pow		___348_f $tmp29 $tmp34 	%argrw{"wrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:57
#         Cout = (color)spline ("linear", f, colormap);
	spline		Cout $const10 ___348_f colormap 	%line{57} %argrw{"wrrr"}
# /Users/mike/ClionProjects/OpenShadingLanguage/src/shaders/mandelbrot.osl:59
#         Cout = 0;
	assign		Cout $const6 	%line{59} %argrw{"wr"}
	end
)oso";
        shader = luisa::compute::osl::OSOParser::parse(code);
    } else {
        shader = luisa::compute::osl::OSOParser::parse_file(argv[1]);
    }

    puts(shader->dump().c_str());
}
