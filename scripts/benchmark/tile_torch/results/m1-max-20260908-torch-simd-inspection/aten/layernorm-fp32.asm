
/Users/mike/.cache/uv/archive-v0/sC3F2za3_-HoN0aE/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000001f81644 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const>:
 1f81644: d10383ff     	sub	sp, sp, #0xe0
 1f81648: 6d0723e9     	stp	d9, d8, [sp, #0x70]
 1f8164c: a9086ffc     	stp	x28, x27, [sp, #0x80]
 1f81650: a90967fa     	stp	x26, x25, [sp, #0x90]
 1f81654: a90a5ff8     	stp	x24, x23, [sp, #0xa0]
 1f81658: a90b57f6     	stp	x22, x21, [sp, #0xb0]
 1f8165c: a90c4ff4     	stp	x20, x19, [sp, #0xc0]
 1f81660: a90d7bfd     	stp	x29, x30, [sp, #0xd0]
 1f81664: 910343fd     	add	x29, sp, #0xd0
 1f81668: eb02003f     	cmp	x1, x2
 1f8166c: 9a82c028     	csel	x8, x1, x2, gt
 1f81670: a90423e1     	stp	x1, x8, [sp, #0x40]
 1f81674: 5400404a     	b.ge	0x1f81e7c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x838>
 1f81678: aa0003f4     	mov	x20, x0
 1f8167c: d2800018     	mov	x24, #0x0               ; =0
 1f81680: f94023fb     	ldr	x27, [sp, #0x40]
 1f81684: d37ef779     	lsl	x25, x27, #2
 1f81688: 1e2e1008     	fmov	s8, #1.00000000
 1f8168c: 6f00e409     	movi.2d	v9, #0000000000000000
 1f81690: 14000007     	b	0x1f816ac <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x68>
 1f81694: 9100077b     	add	x27, x27, #0x1
 1f81698: 91000718     	add	x24, x24, #0x1
 1f8169c: 91001339     	add	x25, x25, #0x4
 1f816a0: f94027e8     	ldr	x8, [sp, #0x48]
 1f816a4: eb08037f     	cmp	x27, x8
 1f816a8: 54003ea0     	b.eq	0x1f81e7c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x838>
 1f816ac: a9402688     	ldp	x8, x9, [x20]
 1f816b0: f940010a     	ldr	x10, [x8]
 1f816b4: f9400135     	ldr	x21, [x9]
 1f816b8: 9b1b7ea8     	mul	x8, x21, x27
 1f816bc: d37ef51c     	lsl	x28, x8, #2
 1f816c0: f9002bea     	str	x10, [sp, #0x50]
 1f816c4: 8b1c0156     	add	x22, x10, x28
 1f816c8: f9400a88     	ldr	x8, [x20, #0x10]
 1f816cc: f940011a     	ldr	x26, [x8]
 1f816d0: aa1603e0     	mov	x0, x22
 1f816d4: aa1503e1     	mov	x1, x21
 1f816d8: d2800002     	mov	x2, #0x0                ; =0
 1f816dc: 940001f1     	bl	0x1f81ea0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)>
 1f816e0: 4ea01c15     	mov.16b	v21, v0
 1f816e4: a941a688     	ldp	x8, x9, [x20, #0x18]
 1f816e8: bd400100     	ldr	s0, [x8]
 1f816ec: 1e202820     	fadd	s0, s1, s0
 1f816f0: 1e21c000     	fsqrt	s0, s0
 1f816f4: 1e201916     	fdiv	s22, s8, s0
 1f816f8: f9400137     	ldr	x23, [x9]
 1f816fc: f9401688     	ldr	x8, [x20, #0x28]
 1f81700: f9400113     	ldr	x19, [x8]
 1f81704: f9400688     	ldr	x8, [x20, #0x8]
 1f81708: f9400108     	ldr	x8, [x8]
 1f8170c: b4000457     	cbz	x23, 0x1f81794 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x150>
 1f81710: b4000433     	cbz	x19, 0x1f81794 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x150>
 1f81714: 1e2142a5     	fneg	s5, s21
 1f81718: eb0803e9     	negs	x9, x8
 1f8171c: 92400529     	and	x9, x9, #0x3
 1f81720: 9240050a     	and	x10, x8, #0x3
 1f81724: da894549     	csneg	x9, x10, x9, mi
 1f81728: cb09010a     	sub	x10, x8, x9
 1f8172c: f100055f     	cmp	x10, #0x1
 1f81730: 540006eb     	b.lt	0x1f8180c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1c8>
 1f81734: d2800009     	mov	x9, #0x0                ; =0
 1f81738: 9b197eac     	mul	x12, x21, x25
 1f8173c: 4e0404a0     	dup.4s	v0, v5[0]
 1f81740: 8b0c034b     	add	x11, x26, x12
 1f81744: f9402bed     	ldr	x13, [sp, #0x50]
 1f81748: 8b0c01ac     	add	x12, x13, x12
 1f8174c: aa1703ed     	mov	x13, x23
 1f81750: aa1303ee     	mov	x14, x19
 1f81754: 6f00e404     	movi.2d	v4, #0000000000000000
 1f81758: 3cc10581     	ldr	q1, [x12], #0x10
 1f8175c: 3cc105a2     	ldr	q2, [x13], #0x10
 1f81760: 3cc105c3     	ldr	q3, [x14], #0x10
 1f81764: 4e21d401     	fadd.4s	v1, v0, v1
 1f81768: 4f969021     	fmul.4s	v1, v1, v22[0]
 1f8176c: 6e21dc41     	fmul.4s	v1, v2, v1
 1f81770: 4e21d461     	fadd.4s	v1, v3, v1
 1f81774: 3c810561     	str	q1, [x11], #0x10
 1f81778: 91001129     	add	x9, x9, #0x4
 1f8177c: eb0a013f     	cmp	x9, x10
 1f81780: 54fffecb     	b.lt	0x1f81758 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x114>
 1f81784: cb090108     	sub	x8, x8, x9
 1f81788: f100051f     	cmp	x8, #0x1
 1f8178c: 540004aa     	b.ge	0x1f81820 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1dc>
 1f81790: 14000135     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81794: f100051f     	cmp	x8, #0x1
 1f81798: 6f00e405     	movi.2d	v5, #0000000000000000
 1f8179c: 5400264b     	b.lt	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f817a0: f94023e9     	ldr	x9, [sp, #0x40]
 1f817a4: 8b180129     	add	x9, x9, x24
 1f817a8: d37ef52a     	lsl	x10, x9, #2
 1f817ac: b4000ab7     	cbz	x23, 0x1f81900 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2bc>
 1f817b0: b4000d33     	cbz	x19, 0x1f81954 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x310>
 1f817b4: f1000d1f     	cmp	x8, #0x3
 1f817b8: 54000f68     	b.hi	0x1f819a4 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x360>
 1f817bc: d2800009     	mov	x9, #0x0                ; =0
 1f817c0: cb090108     	sub	x8, x8, x9
 1f817c4: 9b197eaa     	mul	x10, x21, x25
 1f817c8: d37ef52c     	lsl	x12, x9, #2
 1f817cc: 8b0c014a     	add	x10, x10, x12
 1f817d0: 8b0a0349     	add	x9, x26, x10
 1f817d4: f9402beb     	ldr	x11, [sp, #0x50]
 1f817d8: 8b0a016a     	add	x10, x11, x10
 1f817dc: 8b0c026b     	add	x11, x19, x12
 1f817e0: 8b0c02ec     	add	x12, x23, x12
 1f817e4: bc404580     	ldr	s0, [x12], #0x4
 1f817e8: bc404561     	ldr	s1, [x11], #0x4
 1f817ec: bc404542     	ldr	s2, [x10], #0x4
 1f817f0: 1e353842     	fsub	s2, s2, s21
 1f817f4: 1e220ac2     	fmul	s2, s22, s2
 1f817f8: 1f000440     	fmadd	s0, s2, s0, s1
 1f817fc: bc004520     	str	s0, [x9], #0x4
 1f81800: f1000508     	subs	x8, x8, #0x1
 1f81804: 54ffff01     	b.ne	0x1f817e4 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1a0>
 1f81808: 14000117     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f8180c: d2800009     	mov	x9, #0x0                ; =0
 1f81810: 6f00e404     	movi.2d	v4, #0000000000000000
 1f81814: cb090108     	sub	x8, x8, x9
 1f81818: f100051f     	cmp	x8, #0x1
 1f8181c: 5400224b     	b.lt	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81820: 8b1c035a     	add	x26, x26, x28
 1f81824: d37ef53c     	lsl	x28, x9, #2
 1f81828: 8b090ac1     	add	x1, x22, x9, lsl #2
 1f8182c: f100111f     	cmp	x8, #0x4
 1f81830: 54000161     	b.ne	0x1f8185c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x218>
 1f81834: 3dc00020     	ldr	q0, [x1]
 1f81838: 3cfc6ae1     	ldr	q1, [x23, x28]
 1f8183c: 3cfc6a62     	ldr	q2, [x19, x28]
 1f81840: 4e0404a3     	dup.4s	v3, v5[0]
 1f81844: 4e20d460     	fadd.4s	v0, v3, v0
 1f81848: 4f969000     	fmul.4s	v0, v0, v22[0]
 1f8184c: 6e20dc20     	fmul.4s	v0, v1, v0
 1f81850: 4e20d440     	fadd.4s	v0, v2, v0
 1f81854: 3cbc6b40     	str	q0, [x26, x28]
 1f81858: 14000103     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f8185c: ad0293f6     	stp	q22, q4, [sp, #0x50]
 1f81860: 52800089     	mov	w9, #0x4                ; =4
 1f81864: 9a893108     	csel	x8, x8, x9, lo
 1f81868: d37ef515     	lsl	x21, x8, #2
 1f8186c: 910183e0     	add	x0, sp, #0x60
 1f81870: aa1503e2     	mov	x2, x21
 1f81874: 3d800ff5     	str	q21, [sp, #0x30]
 1f81878: 3d8007e5     	str	q5, [sp, #0x10]
 1f8187c: 94c17fa8     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 1f81880: 3dc01be0     	ldr	q0, [sp, #0x60]
 1f81884: 3d800be0     	str	q0, [sp, #0x20]
 1f81888: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8188c: 3d801be0     	str	q0, [sp, #0x60]
 1f81890: 910183e0     	add	x0, sp, #0x60
 1f81894: 8b1c02e1     	add	x1, x23, x28
 1f81898: aa1503e2     	mov	x2, x21
 1f8189c: 94c17fa0     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 1f818a0: 3dc01be0     	ldr	q0, [sp, #0x60]
 1f818a4: 3d8003e0     	str	q0, [sp]
 1f818a8: 6f00e400     	movi.2d	v0, #0000000000000000
 1f818ac: 3d801be0     	str	q0, [sp, #0x60]
 1f818b0: 910183e0     	add	x0, sp, #0x60
 1f818b4: 8b1c0261     	add	x1, x19, x28
 1f818b8: aa1503e2     	mov	x2, x21
 1f818bc: 94c17f98     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 1f818c0: ad408be0     	ldp	q0, q2, [sp, #0x10]
 1f818c4: 4e040400     	dup.4s	v0, v0[0]
 1f818c8: 4e22d400     	fadd.4s	v0, v0, v2
 1f818cc: ad4287e2     	ldp	q2, q1, [sp, #0x50]
 1f818d0: 4f829000     	fmul.4s	v0, v0, v2[0]
 1f818d4: 3dc003e2     	ldr	q2, [sp]
 1f818d8: 6e22dc00     	fmul.4s	v0, v0, v2
 1f818dc: 4e21d400     	fadd.4s	v0, v0, v1
 1f818e0: 3d801be0     	str	q0, [sp, #0x60]
 1f818e4: 8b1c0340     	add	x0, x26, x28
 1f818e8: 910183e1     	add	x1, sp, #0x60
 1f818ec: aa1503e2     	mov	x2, x21
 1f818f0: 94c17f8b     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 1f818f4: 3dc017f6     	ldr	q22, [sp, #0x50]
 1f818f8: 3dc00ff5     	ldr	q21, [sp, #0x30]
 1f818fc: 140000da     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81900: b4000733     	cbz	x19, 0x1f819e4 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x3a0>
 1f81904: f1000d1f     	cmp	x8, #0x3
 1f81908: 54000848     	b.hi	0x1f81a10 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x3cc>
 1f8190c: d2800009     	mov	x9, #0x0                ; =0
 1f81910: cb090108     	sub	x8, x8, x9
 1f81914: 9b197eaa     	mul	x10, x21, x25
 1f81918: d37ef52b     	lsl	x11, x9, #2
 1f8191c: 8b0b014a     	add	x10, x10, x11
 1f81920: 8b0a0349     	add	x9, x26, x10
 1f81924: f9402bec     	ldr	x12, [sp, #0x50]
 1f81928: 8b0a018a     	add	x10, x12, x10
 1f8192c: 8b0b026b     	add	x11, x19, x11
 1f81930: bc404560     	ldr	s0, [x11], #0x4
 1f81934: bc404541     	ldr	s1, [x10], #0x4
 1f81938: 1e353821     	fsub	s1, s1, s21
 1f8193c: 1e210ac1     	fmul	s1, s22, s1
 1f81940: 1e212800     	fadd	s0, s0, s1
 1f81944: bc004520     	str	s0, [x9], #0x4
 1f81948: f1000508     	subs	x8, x8, #0x1
 1f8194c: 54ffff21     	b.ne	0x1f81930 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2ec>
 1f81950: 140000c5     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81954: f100111f     	cmp	x8, #0x4
 1f81958: 54000762     	b.hs	0x1f81a44 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x400>
 1f8195c: d2800009     	mov	x9, #0x0                ; =0
 1f81960: cb090108     	sub	x8, x8, x9
 1f81964: 9b197eaa     	mul	x10, x21, x25
 1f81968: d37ef52b     	lsl	x11, x9, #2
 1f8196c: 8b0b014a     	add	x10, x10, x11
 1f81970: 8b0a0349     	add	x9, x26, x10
 1f81974: f9402bec     	ldr	x12, [sp, #0x50]
 1f81978: 8b0a018a     	add	x10, x12, x10
 1f8197c: 8b0b02eb     	add	x11, x23, x11
 1f81980: bc404560     	ldr	s0, [x11], #0x4
 1f81984: bc404541     	ldr	s1, [x10], #0x4
 1f81988: 1e353821     	fsub	s1, s1, s21
 1f8198c: 1e210ac1     	fmul	s1, s22, s1
 1f81990: 1f002420     	fmadd	s0, s1, s0, s9
 1f81994: bc004520     	str	s0, [x9], #0x4
 1f81998: f1000508     	subs	x8, x8, #0x1
 1f8199c: 54ffff21     	b.ne	0x1f81980 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x33c>
 1f819a0: 140000b1     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f819a4: d2800009     	mov	x9, #0x0                ; =0
 1f819a8: 9b0a6aaa     	madd	x10, x21, x10, x26
 1f819ac: cb17014b     	sub	x11, x10, x23
 1f819b0: f101017f     	cmp	x11, #0x40
 1f819b4: 54fff063     	b.lo	0x1f817c0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x17c>
 1f819b8: cb13014a     	sub	x10, x10, x19
 1f819bc: f101015f     	cmp	x10, #0x40
 1f819c0: 54fff003     	b.lo	0x1f817c0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x17c>
 1f819c4: f9402bea     	ldr	x10, [sp, #0x50]
 1f819c8: cb0a034a     	sub	x10, x26, x10
 1f819cc: f101015f     	cmp	x10, #0x40
 1f819d0: 54ffef83     	b.lo	0x1f817c0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x17c>
 1f819d4: f100411f     	cmp	x8, #0x10
 1f819d8: 54000502     	b.hs	0x1f81a78 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x434>
 1f819dc: d2800009     	mov	x9, #0x0                ; =0
 1f819e0: 1400004b     	b	0x1f81b0c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x4c8>
 1f819e4: d2800009     	mov	x9, #0x0                ; =0
 1f819e8: f100111f     	cmp	x8, #0x4
 1f819ec: 54001223     	b.lo	0x1f81c30 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x5ec>
 1f819f0: f9402bea     	ldr	x10, [sp, #0x50]
 1f819f4: cb0a034a     	sub	x10, x26, x10
 1f819f8: f101015f     	cmp	x10, #0x40
 1f819fc: 540011a3     	b.lo	0x1f81c30 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x5ec>
 1f81a00: f100411f     	cmp	x8, #0x10
 1f81a04: 54000b42     	b.hs	0x1f81b6c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x528>
 1f81a08: d2800009     	mov	x9, #0x0                ; =0
 1f81a0c: 14000077     	b	0x1f81be8 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x5a4>
 1f81a10: d2800009     	mov	x9, #0x0                ; =0
 1f81a14: 9b0a6aaa     	madd	x10, x21, x10, x26
 1f81a18: cb13014a     	sub	x10, x10, x19
 1f81a1c: f101015f     	cmp	x10, #0x40
 1f81a20: 54fff783     	b.lo	0x1f81910 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2cc>
 1f81a24: f9402bea     	ldr	x10, [sp, #0x50]
 1f81a28: cb0a034a     	sub	x10, x26, x10
 1f81a2c: f101015f     	cmp	x10, #0x40
 1f81a30: 54fff703     	b.lo	0x1f81910 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2cc>
 1f81a34: f100411f     	cmp	x8, #0x10
 1f81a38: 54001382     	b.hs	0x1f81ca8 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x664>
 1f81a3c: d2800009     	mov	x9, #0x0                ; =0
 1f81a40: 140000bc     	b	0x1f81d30 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x6ec>
 1f81a44: d2800009     	mov	x9, #0x0                ; =0
 1f81a48: 9b0a6aaa     	madd	x10, x21, x10, x26
 1f81a4c: cb17014a     	sub	x10, x10, x23
 1f81a50: f101015f     	cmp	x10, #0x40
 1f81a54: 54fff863     	b.lo	0x1f81960 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x31c>
 1f81a58: f9402bea     	ldr	x10, [sp, #0x50]
 1f81a5c: cb0a034a     	sub	x10, x26, x10
 1f81a60: f101015f     	cmp	x10, #0x40
 1f81a64: 54fff7e3     	b.lo	0x1f81960 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x31c>
 1f81a68: f100411f     	cmp	x8, #0x10
 1f81a6c: 540018e2     	b.hs	0x1f81d88 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x744>
 1f81a70: d2800009     	mov	x9, #0x0                ; =0
 1f81a74: 140000eb     	b	0x1f81e20 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x7dc>
 1f81a78: 927ce909     	and	x9, x8, #0x7ffffffffffffff0
 1f81a7c: 4e0406a0     	dup.4s	v0, v21[0]
 1f81a80: 910082ea     	add	x10, x23, #0x20
 1f81a84: 9100826b     	add	x11, x19, #0x20
 1f81a88: 9b197ead     	mul	x13, x21, x25
 1f81a8c: 8b0d034c     	add	x12, x26, x13
 1f81a90: 9100818c     	add	x12, x12, #0x20
 1f81a94: f9402bee     	ldr	x14, [sp, #0x50]
 1f81a98: 8b0d01cd     	add	x13, x14, x13
 1f81a9c: 910081ad     	add	x13, x13, #0x20
 1f81aa0: aa0903ee     	mov	x14, x9
 1f81aa4: ad7f0941     	ldp	q1, q2, [x10, #-0x20]
 1f81aa8: acc21143     	ldp	q3, q4, [x10], #0x40
 1f81aac: ad7f1965     	ldp	q5, q6, [x11, #-0x20]
 1f81ab0: acc24167     	ldp	q7, q16, [x11], #0x40
 1f81ab4: ad7f49b1     	ldp	q17, q18, [x13, #-0x20]
 1f81ab8: acc251b3     	ldp	q19, q20, [x13], #0x40
 1f81abc: 4ea0d631     	fsub.4s	v17, v17, v0
 1f81ac0: 4ea0d652     	fsub.4s	v18, v18, v0
 1f81ac4: 4ea0d673     	fsub.4s	v19, v19, v0
 1f81ac8: 4ea0d694     	fsub.4s	v20, v20, v0
 1f81acc: 4f969231     	fmul.4s	v17, v17, v22[0]
 1f81ad0: 4f969252     	fmul.4s	v18, v18, v22[0]
 1f81ad4: 4f969273     	fmul.4s	v19, v19, v22[0]
 1f81ad8: 4f969294     	fmul.4s	v20, v20, v22[0]
 1f81adc: 4e31cc25     	fmla.4s	v5, v1, v17
 1f81ae0: 4e32cc46     	fmla.4s	v6, v2, v18
 1f81ae4: 4e33cc67     	fmla.4s	v7, v3, v19
 1f81ae8: 4e34cc90     	fmla.4s	v16, v4, v20
 1f81aec: ad3f1985     	stp	q5, q6, [x12, #-0x20]
 1f81af0: ac824187     	stp	q7, q16, [x12], #0x40
 1f81af4: f10041ce     	subs	x14, x14, #0x10
 1f81af8: 54fffd61     	b.ne	0x1f81aa4 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x460>
 1f81afc: eb09011f     	cmp	x8, x9
 1f81b00: 54000b20     	b.eq	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81b04: f27e051f     	tst	x8, #0xc
 1f81b08: 54ffe5c0     	b.eq	0x1f817c0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x17c>
 1f81b0c: aa0903eb     	mov	x11, x9
 1f81b10: 927ef109     	and	x9, x8, #0x7ffffffffffffffc
 1f81b14: 4e0406a0     	dup.4s	v0, v21[0]
 1f81b18: cb09016a     	sub	x10, x11, x9
 1f81b1c: 9b197eac     	mul	x12, x21, x25
 1f81b20: d37ef56e     	lsl	x14, x11, #2
 1f81b24: 8b0e018c     	add	x12, x12, x14
 1f81b28: 8b0c034b     	add	x11, x26, x12
 1f81b2c: f9402bed     	ldr	x13, [sp, #0x50]
 1f81b30: 8b0c01ac     	add	x12, x13, x12
 1f81b34: 8b0e026d     	add	x13, x19, x14
 1f81b38: 8b0e02ee     	add	x14, x23, x14
 1f81b3c: 3cc105c1     	ldr	q1, [x14], #0x10
 1f81b40: 3cc105a2     	ldr	q2, [x13], #0x10
 1f81b44: 3cc10583     	ldr	q3, [x12], #0x10
 1f81b48: 4ea0d463     	fsub.4s	v3, v3, v0
 1f81b4c: 4f969063     	fmul.4s	v3, v3, v22[0]
 1f81b50: 4e23cc22     	fmla.4s	v2, v1, v3
 1f81b54: 3c810562     	str	q2, [x11], #0x10
 1f81b58: b100114a     	adds	x10, x10, #0x4
 1f81b5c: 54ffff01     	b.ne	0x1f81b3c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x4f8>
 1f81b60: eb09011f     	cmp	x8, x9
 1f81b64: 54ffe2e1     	b.ne	0x1f817c0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x17c>
 1f81b68: 1400003f     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81b6c: 927ce909     	and	x9, x8, #0x7ffffffffffffff0
 1f81b70: 4e0406a0     	dup.4s	v0, v21[0]
 1f81b74: 9b197eab     	mul	x11, x21, x25
 1f81b78: f9402bea     	ldr	x10, [sp, #0x50]
 1f81b7c: 8b0b014a     	add	x10, x10, x11
 1f81b80: 9100814a     	add	x10, x10, #0x20
 1f81b84: 8b0b034b     	add	x11, x26, x11
 1f81b88: 9100816b     	add	x11, x11, #0x20
 1f81b8c: aa0903ec     	mov	x12, x9
 1f81b90: ad7f0941     	ldp	q1, q2, [x10, #-0x20]
 1f81b94: acc21143     	ldp	q3, q4, [x10], #0x40
 1f81b98: 4ea0d421     	fsub.4s	v1, v1, v0
 1f81b9c: 4ea0d442     	fsub.4s	v2, v2, v0
 1f81ba0: 4ea0d463     	fsub.4s	v3, v3, v0
 1f81ba4: 4ea0d484     	fsub.4s	v4, v4, v0
 1f81ba8: 4f969021     	fmul.4s	v1, v1, v22[0]
 1f81bac: 4f969042     	fmul.4s	v2, v2, v22[0]
 1f81bb0: 4f969063     	fmul.4s	v3, v3, v22[0]
 1f81bb4: 4f969084     	fmul.4s	v4, v4, v22[0]
 1f81bb8: 4e25d421     	fadd.4s	v1, v1, v5
 1f81bbc: 4e25d442     	fadd.4s	v2, v2, v5
 1f81bc0: 4e25d463     	fadd.4s	v3, v3, v5
 1f81bc4: ad3f0961     	stp	q1, q2, [x11, #-0x20]
 1f81bc8: 4e25d481     	fadd.4s	v1, v4, v5
 1f81bcc: ac820563     	stp	q3, q1, [x11], #0x40
 1f81bd0: f100418c     	subs	x12, x12, #0x10
 1f81bd4: 54fffde1     	b.ne	0x1f81b90 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x54c>
 1f81bd8: eb09011f     	cmp	x8, x9
 1f81bdc: 54000440     	b.eq	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81be0: f27e051f     	tst	x8, #0xc
 1f81be4: 54000260     	b.eq	0x1f81c30 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x5ec>
 1f81be8: aa0903eb     	mov	x11, x9
 1f81bec: 4e0406a0     	dup.4s	v0, v21[0]
 1f81bf0: 927ef109     	and	x9, x8, #0x7ffffffffffffffc
 1f81bf4: cb09016a     	sub	x10, x11, x9
 1f81bf8: 9b197eac     	mul	x12, x21, x25
 1f81bfc: 8b0b098c     	add	x12, x12, x11, lsl #2
 1f81c00: 8b0c034b     	add	x11, x26, x12
 1f81c04: f9402bed     	ldr	x13, [sp, #0x50]
 1f81c08: 8b0c01ac     	add	x12, x13, x12
 1f81c0c: 3cc10581     	ldr	q1, [x12], #0x10
 1f81c10: 4ea0d421     	fsub.4s	v1, v1, v0
 1f81c14: 4f969021     	fmul.4s	v1, v1, v22[0]
 1f81c18: 4e25d421     	fadd.4s	v1, v1, v5
 1f81c1c: 3c810561     	str	q1, [x11], #0x10
 1f81c20: b100114a     	adds	x10, x10, #0x4
 1f81c24: 54ffff41     	b.ne	0x1f81c0c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x5c8>
 1f81c28: eb09011f     	cmp	x8, x9
 1f81c2c: 540001c0     	b.eq	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81c30: cb090108     	sub	x8, x8, x9
 1f81c34: 9b197eaa     	mul	x10, x21, x25
 1f81c38: 8b09094a     	add	x10, x10, x9, lsl #2
 1f81c3c: 8b0a0349     	add	x9, x26, x10
 1f81c40: f9402beb     	ldr	x11, [sp, #0x50]
 1f81c44: 8b0a016a     	add	x10, x11, x10
 1f81c48: bc404540     	ldr	s0, [x10], #0x4
 1f81c4c: 1e353800     	fsub	s0, s0, s21
 1f81c50: 1e200ac0     	fmul	s0, s22, s0
 1f81c54: 1e292800     	fadd	s0, s0, s9
 1f81c58: bc004520     	str	s0, [x9], #0x4
 1f81c5c: f1000508     	subs	x8, x8, #0x1
 1f81c60: 54ffff41     	b.ne	0x1f81c48 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x604>
 1f81c64: f9401a88     	ldr	x8, [x20, #0x30]
 1f81c68: 39400108     	ldrb	w8, [x8]
 1f81c6c: 360000a8     	tbz	w8, #0x0, 0x1f81c80 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x63c>
 1f81c70: f9402288     	ldr	x8, [x20, #0x40]
 1f81c74: 39400108     	ldrb	w8, [x8]
 1f81c78: 3707d0e8     	tbnz	w8, #0x0, 0x1f81694 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x50>
 1f81c7c: 14000007     	b	0x1f81c98 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x654>
 1f81c80: f9401e88     	ldr	x8, [x20, #0x38]
 1f81c84: f9400108     	ldr	x8, [x8]
 1f81c88: bc3b7915     	str	s21, [x8, x27, lsl #2]
 1f81c8c: f9402288     	ldr	x8, [x20, #0x40]
 1f81c90: 39400108     	ldrb	w8, [x8]
 1f81c94: 3707d008     	tbnz	w8, #0x0, 0x1f81694 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x50>
 1f81c98: f9402688     	ldr	x8, [x20, #0x48]
 1f81c9c: f9400108     	ldr	x8, [x8]
 1f81ca0: bc3b7916     	str	s22, [x8, x27, lsl #2]
 1f81ca4: 17fffe7c     	b	0x1f81694 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x50>
 1f81ca8: 927ce909     	and	x9, x8, #0x7ffffffffffffff0
 1f81cac: 4e0406a0     	dup.4s	v0, v21[0]
 1f81cb0: 9100826a     	add	x10, x19, #0x20
 1f81cb4: 9b197eac     	mul	x12, x21, x25
 1f81cb8: f9402beb     	ldr	x11, [sp, #0x50]
 1f81cbc: 8b0c016b     	add	x11, x11, x12
 1f81cc0: 9100816b     	add	x11, x11, #0x20
 1f81cc4: 8b0c034c     	add	x12, x26, x12
 1f81cc8: 9100818c     	add	x12, x12, #0x20
 1f81ccc: aa0903ed     	mov	x13, x9
 1f81cd0: ad7f0941     	ldp	q1, q2, [x10, #-0x20]
 1f81cd4: acc21143     	ldp	q3, q4, [x10], #0x40
 1f81cd8: ad7f1965     	ldp	q5, q6, [x11, #-0x20]
 1f81cdc: acc24167     	ldp	q7, q16, [x11], #0x40
 1f81ce0: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f81ce4: 4ea0d4c6     	fsub.4s	v6, v6, v0
 1f81ce8: 4ea0d4e7     	fsub.4s	v7, v7, v0
 1f81cec: 4ea0d610     	fsub.4s	v16, v16, v0
 1f81cf0: 4f9690a5     	fmul.4s	v5, v5, v22[0]
 1f81cf4: 4f9690c6     	fmul.4s	v6, v6, v22[0]
 1f81cf8: 4f9690e7     	fmul.4s	v7, v7, v22[0]
 1f81cfc: 4f969210     	fmul.4s	v16, v16, v22[0]
 1f81d00: 4e25d421     	fadd.4s	v1, v1, v5
 1f81d04: 4e26d442     	fadd.4s	v2, v2, v6
 1f81d08: 4e27d463     	fadd.4s	v3, v3, v7
 1f81d0c: 4e30d484     	fadd.4s	v4, v4, v16
 1f81d10: ad3f0981     	stp	q1, q2, [x12, #-0x20]
 1f81d14: ac821183     	stp	q3, q4, [x12], #0x40
 1f81d18: f10041ad     	subs	x13, x13, #0x10
 1f81d1c: 54fffda1     	b.ne	0x1f81cd0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x68c>
 1f81d20: eb09011f     	cmp	x8, x9
 1f81d24: 54fffa00     	b.eq	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81d28: f27e051f     	tst	x8, #0xc
 1f81d2c: 54ffdf20     	b.eq	0x1f81910 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2cc>
 1f81d30: aa0903eb     	mov	x11, x9
 1f81d34: 927ef109     	and	x9, x8, #0x7ffffffffffffffc
 1f81d38: 4e0406a0     	dup.4s	v0, v21[0]
 1f81d3c: cb09016a     	sub	x10, x11, x9
 1f81d40: 9b197eac     	mul	x12, x21, x25
 1f81d44: d37ef56d     	lsl	x13, x11, #2
 1f81d48: 8b0d018c     	add	x12, x12, x13
 1f81d4c: 8b0c034b     	add	x11, x26, x12
 1f81d50: f9402bee     	ldr	x14, [sp, #0x50]
 1f81d54: 8b0c01cc     	add	x12, x14, x12
 1f81d58: 8b0d026d     	add	x13, x19, x13
 1f81d5c: 3cc105a1     	ldr	q1, [x13], #0x10
 1f81d60: 3cc10582     	ldr	q2, [x12], #0x10
 1f81d64: 4ea0d442     	fsub.4s	v2, v2, v0
 1f81d68: 4f969042     	fmul.4s	v2, v2, v22[0]
 1f81d6c: 4e22d421     	fadd.4s	v1, v1, v2
 1f81d70: 3c810561     	str	q1, [x11], #0x10
 1f81d74: b100114a     	adds	x10, x10, #0x4
 1f81d78: 54ffff21     	b.ne	0x1f81d5c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x718>
 1f81d7c: eb09011f     	cmp	x8, x9
 1f81d80: 54ffdc81     	b.ne	0x1f81910 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2cc>
 1f81d84: 17ffffb8     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81d88: 927ce909     	and	x9, x8, #0x7ffffffffffffff0
 1f81d8c: 4e0406a0     	dup.4s	v0, v21[0]
 1f81d90: 910082ea     	add	x10, x23, #0x20
 1f81d94: 9b197eac     	mul	x12, x21, x25
 1f81d98: f9402beb     	ldr	x11, [sp, #0x50]
 1f81d9c: 8b0c016b     	add	x11, x11, x12
 1f81da0: 9100816b     	add	x11, x11, #0x20
 1f81da4: 8b0c034c     	add	x12, x26, x12
 1f81da8: 9100818c     	add	x12, x12, #0x20
 1f81dac: aa0903ed     	mov	x13, x9
 1f81db0: ad7f0941     	ldp	q1, q2, [x10, #-0x20]
 1f81db4: acc21143     	ldp	q3, q4, [x10], #0x40
 1f81db8: ad7f1965     	ldp	q5, q6, [x11, #-0x20]
 1f81dbc: acc24167     	ldp	q7, q16, [x11], #0x40
 1f81dc0: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f81dc4: 4ea0d4c6     	fsub.4s	v6, v6, v0
 1f81dc8: 4ea0d4e7     	fsub.4s	v7, v7, v0
 1f81dcc: 4ea0d610     	fsub.4s	v16, v16, v0
 1f81dd0: 4f9690a5     	fmul.4s	v5, v5, v22[0]
 1f81dd4: 4f9690c6     	fmul.4s	v6, v6, v22[0]
 1f81dd8: 4f9690e7     	fmul.4s	v7, v7, v22[0]
 1f81ddc: 4f969210     	fmul.4s	v16, v16, v22[0]
 1f81de0: 6f00e411     	movi.2d	v17, #0000000000000000
 1f81de4: 4e25cc31     	fmla.4s	v17, v1, v5
 1f81de8: 6f00e401     	movi.2d	v1, #0000000000000000
 1f81dec: 4e26cc41     	fmla.4s	v1, v2, v6
 1f81df0: 6f00e402     	movi.2d	v2, #0000000000000000
 1f81df4: 4e27cc62     	fmla.4s	v2, v3, v7
 1f81df8: 6f00e403     	movi.2d	v3, #0000000000000000
 1f81dfc: 4e30cc83     	fmla.4s	v3, v4, v16
 1f81e00: ad3f0591     	stp	q17, q1, [x12, #-0x20]
 1f81e04: ac820d82     	stp	q2, q3, [x12], #0x40
 1f81e08: f10041ad     	subs	x13, x13, #0x10
 1f81e0c: 54fffd21     	b.ne	0x1f81db0 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x76c>
 1f81e10: eb09011f     	cmp	x8, x9
 1f81e14: 54fff280     	b.eq	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81e18: f27e051f     	tst	x8, #0xc
 1f81e1c: 54ffda20     	b.eq	0x1f81960 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x31c>
 1f81e20: aa0903eb     	mov	x11, x9
 1f81e24: 927ef109     	and	x9, x8, #0x7ffffffffffffffc
 1f81e28: 4e0406a0     	dup.4s	v0, v21[0]
 1f81e2c: cb09016a     	sub	x10, x11, x9
 1f81e30: 9b197eac     	mul	x12, x21, x25
 1f81e34: d37ef56d     	lsl	x13, x11, #2
 1f81e38: 8b0d018c     	add	x12, x12, x13
 1f81e3c: 8b0c034b     	add	x11, x26, x12
 1f81e40: f9402bee     	ldr	x14, [sp, #0x50]
 1f81e44: 8b0c01cc     	add	x12, x14, x12
 1f81e48: 8b0d02ed     	add	x13, x23, x13
 1f81e4c: 3cc105a1     	ldr	q1, [x13], #0x10
 1f81e50: 3cc10582     	ldr	q2, [x12], #0x10
 1f81e54: 4ea0d442     	fsub.4s	v2, v2, v0
 1f81e58: 4f969042     	fmul.4s	v2, v2, v22[0]
 1f81e5c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f81e60: 4e22cc23     	fmla.4s	v3, v1, v2
 1f81e64: 3c810563     	str	q3, [x11], #0x10
 1f81e68: b100114a     	adds	x10, x10, #0x4
 1f81e6c: 54ffff01     	b.ne	0x1f81e4c <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x808>
 1f81e70: eb09011f     	cmp	x8, x9
 1f81e74: 54ffd761     	b.ne	0x1f81960 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x31c>
 1f81e78: 17ffff7b     	b	0x1f81c64 <void at::native::(anonymous namespace)::LayerNormKernelImplInternal<float, 0>(at::Tensor const&, at::Tensor const&, at::Tensor const&, long long, long long, float, at::Tensor*, at::Tensor*, at::Tensor*)::'lambda'(long long, long long)::operator()(long long, long long) const+0x620>
 1f81e7c: a94d7bfd     	ldp	x29, x30, [sp, #0xd0]
 1f81e80: a94c4ff4     	ldp	x20, x19, [sp, #0xc0]
 1f81e84: a94b57f6     	ldp	x22, x21, [sp, #0xb0]
 1f81e88: a94a5ff8     	ldp	x24, x23, [sp, #0xa0]
 1f81e8c: a94967fa     	ldp	x26, x25, [sp, #0x90]
 1f81e90: a9486ffc     	ldp	x28, x27, [sp, #0x80]
 1f81e94: 6d4723e9     	ldp	d9, d8, [sp, #0x70]
 1f81e98: 910383ff     	add	sp, sp, #0xe0
 1f81e9c: d65f03c0     	ret

0000000001f81ea0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)>:
 1f81ea0: 91000c28     	add	x8, x1, #0x3
 1f81ea4: f100003f     	cmp	x1, #0x0
 1f81ea8: 9a81b108     	csel	x8, x8, x1, lt
 1f81eac: 9342fd08     	asr	x8, x8, #2
 1f81eb0: b1003d09     	adds	x9, x8, #0xf
 1f81eb4: 91007908     	add	x8, x8, #0x1e
 1f81eb8: 9a89b108     	csel	x8, x8, x9, lt
 1f81ebc: 9344fd08     	asr	x8, x8, #4
 1f81ec0: d1000508     	sub	x8, x8, #0x1
 1f81ec4: dac01108     	clz	x8, x8
 1f81ec8: d2401508     	eor	x8, x8, #0x3f
 1f81ecc: f102103f     	cmp	x1, #0x84
 1f81ed0: 52800029     	mov	w9, #0x1                ; =1
 1f81ed4: 9a88b528     	csinc	x8, x9, x8, lt
 1f81ed8: f100111f     	cmp	x8, #0x4
 1f81edc: 54000048     	b.hi	0x1f81ee4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)+0x44>
 1f81ee0: 1400000b     	b	0x1f81f0c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)>
 1f81ee4: f100211f     	cmp	x8, #0x8
 1f81ee8: 54000048     	b.hi	0x1f81ef0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)+0x50>
 1f81eec: 1400016f     	b	0x1f824a8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)>
 1f81ef0: f100411f     	cmp	x8, #0x10
 1f81ef4: 54000048     	b.hi	0x1f81efc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)+0x5c>
 1f81ef8: 140002d8     	b	0x1f82a58 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)>
 1f81efc: f100811f     	cmp	x8, #0x20
 1f81f00: 54000048     	b.hi	0x1f81f08 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMoments<float>(float const*, long long, long long)+0x68>
 1f81f04: 1400044b     	b	0x1f83030 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)>
 1f81f08: 140005d9     	b	0x1f8366c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)>

0000000001f81f0c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)>:
 1f81f0c: d104c3ff     	sub	sp, sp, #0x130
 1f81f10: a90f5ff8     	stp	x24, x23, [sp, #0xf0]
 1f81f14: a91057f6     	stp	x22, x21, [sp, #0x100]
 1f81f18: a9114ff4     	stp	x20, x19, [sp, #0x110]
 1f81f1c: a9127bfd     	stp	x29, x30, [sp, #0x120]
 1f81f20: 910483fd     	add	x29, sp, #0x120
 1f81f24: d280000a     	mov	x10, #0x0               ; =0
 1f81f28: 91000c28     	add	x8, x1, #0x3
 1f81f2c: f100003f     	cmp	x1, #0x0
 1f81f30: 9a81b108     	csel	x8, x8, x1, lt
 1f81f34: 9342fd08     	asr	x8, x8, #2
 1f81f38: b1003d09     	adds	x9, x8, #0xf
 1f81f3c: 9100790b     	add	x11, x8, #0x1e
 1f81f40: 9a89b169     	csel	x9, x11, x9, lt
 1f81f44: 9344fd2b     	asr	x11, x9, #4
 1f81f48: d1000569     	sub	x9, x11, #0x1
 1f81f4c: dac01129     	clz	x9, x9
 1f81f50: d2401529     	eor	x9, x9, #0x3f
 1f81f54: f102103f     	cmp	x1, #0x84
 1f81f58: 5280002c     	mov	w12, #0x1               ; =1
 1f81f5c: 9a89b589     	csinc	x9, x12, x9, lt
 1f81f60: 6f00e401     	movi.2d	v1, #0000000000000000
 1f81f64: ad3d87a1     	stp	q1, q1, [x29, #-0x50]
 1f81f68: ad0587e1     	stp	q1, q1, [sp, #0xb0]
 1f81f6c: ad0487e1     	stp	q1, q1, [sp, #0x90]
 1f81f70: ad0387e1     	stp	q1, q1, [sp, #0x70]
 1f81f74: ad0287e1     	stp	q1, q1, [sp, #0x50]
 1f81f78: b101ec3f     	cmn	x1, #0x7b
 1f81f7c: 540015cb     	b.lt	0x1f82234 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x328>
 1f81f80: d100050c     	sub	x12, x8, #0x1
 1f81f84: 6f00e400     	movi.2d	v0, #0000000000000000
 1f81f88: b100819f     	cmn	x12, #0x20
 1f81f8c: 540010c8     	b.hi	0x1f821a4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x298>
 1f81f90: d2800004     	mov	x4, #0x0                ; =0
 1f81f94: 910143ea     	add	x10, sp, #0x50
 1f81f98: 9100414a     	add	x10, x10, #0x10
 1f81f9c: 910243ec     	add	x12, sp, #0x90
 1f81fa0: 9100418c     	add	x12, x12, #0x10
 1f81fa4: d10143ad     	sub	x13, x29, #0x50
 1f81fa8: b27d01ad     	orr	x13, x13, #0x8
 1f81fac: 5280020f     	mov	w15, #0x10              ; =16
 1f81fb0: b00880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f81fb4: 911a41ce     	add	x14, x14, #0x690
 1f81fb8: 6f00e400     	movi.2d	v0, #0000000000000000
 1f81fbc: 6f00e406     	movi.2d	v6, #0000000000000000
 1f81fc0: aa0003f0     	mov	x16, x0
 1f81fc4: aa0803e3     	mov	x3, x8
 1f81fc8: b00880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f81fcc: 91164231     	add	x17, x17, #0x590
 1f81fd0: 14000004     	b	0x1f81fe0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0xd4>
 1f81fd4: 91040210     	add	x16, x16, #0x100
 1f81fd8: eb0b009f     	cmp	x4, x11
 1f81fdc: 54000de0     	b.eq	0x1f82198 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x28c>
 1f81fe0: aa0403e5     	mov	x5, x4
 1f81fe4: aa0303e4     	mov	x4, x3
 1f81fe8: f1004063     	subs	x3, x3, #0x10
 1f81fec: 9a8fb094     	csel	x20, x4, x15, lt
 1f81ff0: cb051115     	sub	x21, x8, x5, lsl #4
 1f81ff4: f10042bf     	cmp	x21, #0x10
 1f81ff8: 9a8fb2b3     	csel	x19, x21, x15, lt
 1f81ffc: 38bfc1c4     	ldaprb	w4, [x14]
 1f82000: 36000204     	tbz	w4, #0x0, 0x1f82040 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x134>
 1f82004: f10006bf     	cmp	x21, #0x1
 1f82008: 5400050b     	b.lt	0x1f820a8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x19c>
 1f8200c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82010: aa1003e4     	mov	x4, x16
 1f82014: aa1103e6     	mov	x6, x17
 1f82018: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8201c: 3cc10483     	ldr	q3, [x4], #0x10
 1f82020: 3cc104c4     	ldr	q4, [x6], #0x10
 1f82024: 4ea1d465     	fsub.4s	v5, v3, v1
 1f82028: 4e24cca1     	fmla.4s	v1, v5, v4
 1f8202c: 4ea1d463     	fsub.4s	v3, v3, v1
 1f82030: 4e25cc62     	fmla.4s	v2, v3, v5
 1f82034: f1000694     	subs	x20, x20, #0x1
 1f82038: 54ffff21     	b.ne	0x1f8201c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x110>
 1f8203c: 1400001d     	b	0x1f820b0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x1a4>
 1f82040: a9010be0     	stp	x0, x2, [sp, #0x10]
 1f82044: aa0103f6     	mov	x22, x1
 1f82048: aa0803f7     	mov	x23, x8
 1f8204c: a902a7ea     	stp	x10, x9, [sp, #0x28]
 1f82050: a9002fec     	stp	x12, x11, [sp]
 1f82054: f90013ed     	str	x13, [sp, #0x20]
 1f82058: a90443e3     	stp	x3, x16, [sp, #0x40]
 1f8205c: f9001fe5     	str	x5, [sp, #0x38]
 1f82060: 94c0e9ec     	bl	0x4fbc810 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long) (.cold.1)>
 1f82064: 6f00e406     	movi.2d	v6, #0000000000000000
 1f82068: a9438fe5     	ldp	x5, x3, [sp, #0x38]
 1f8206c: 900880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82070: 91164231     	add	x17, x17, #0x590
 1f82074: f94027f0     	ldr	x16, [sp, #0x48]
 1f82078: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8207c: 900880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82080: 911a41ce     	add	x14, x14, #0x690
 1f82084: 5280020f     	mov	w15, #0x10              ; =16
 1f82088: a9422bed     	ldp	x13, x10, [sp, #0x20]
 1f8208c: a9402fec     	ldp	x12, x11, [sp]
 1f82090: f9401be9     	ldr	x9, [sp, #0x30]
 1f82094: aa1703e8     	mov	x8, x23
 1f82098: a9410be0     	ldp	x0, x2, [sp, #0x10]
 1f8209c: aa1603e1     	mov	x1, x22
 1f820a0: f10006bf     	cmp	x21, #0x1
 1f820a4: 54fffb4a     	b.ge	0x1f8200c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x100>
 1f820a8: 6f00e402     	movi.2d	v2, #0000000000000000
 1f820ac: 6f00e401     	movi.2d	v1, #0000000000000000
 1f820b0: f85b03a4     	ldur	x4, [x29, #-0x50]
 1f820b4: ab130095     	adds	x21, x4, x19
 1f820b8: 9e220263     	scvtf	s3, x19
 1f820bc: 9e2202a4     	scvtf	s4, x21
 1f820c0: 1e241863     	fdiv	s3, s3, s4
 1f820c4: 1e230c03     	fcsel	s3, s0, s3, eq
 1f820c8: 3dc027e4     	ldr	q4, [sp, #0x90]
 1f820cc: 4ea4d425     	fsub.4s	v5, v1, v4
 1f820d0: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f820d4: 4e21d441     	fadd.4s	v1, v2, v1
 1f820d8: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f820dc: 9e220082     	scvtf	s2, x4
 1f820e0: 4f8290a5     	fmul.4s	v5, v5, v2[0]
 1f820e4: 4e23d482     	fadd.4s	v2, v4, v3
 1f820e8: 3d8027e2     	str	q2, [sp, #0x90]
 1f820ec: 4e25cc61     	fmla.4s	v1, v3, v5
 1f820f0: 3d8017e1     	str	q1, [sp, #0x50]
 1f820f4: f81b03b5     	stur	x21, [x29, #-0x50]
 1f820f8: 910004a4     	add	x4, x5, #0x1
 1f820fc: f100093f     	cmp	x9, #0x2
 1f82100: 54fff6a3     	b.lo	0x1f81fd4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0xc8>
 1f82104: 3607f685     	tbz	w5, #0x0, 0x1f81fd4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0xc8>
 1f82108: aa0d03e5     	mov	x5, x13
 1f8210c: aa0c03e6     	mov	x6, x12
 1f82110: aa0a03e7     	mov	x7, x10
 1f82114: 52800053     	mov	w19, #0x2               ; =2
 1f82118: aa0403f7     	mov	x23, x4
 1f8211c: aa1703f4     	mov	x20, x23
 1f82120: f94000b7     	ldr	x23, [x5]
 1f82124: ab1502f6     	adds	x22, x23, x21
 1f82128: 540000a0     	b.eq	0x1f8213c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x230>
 1f8212c: 9e2202a3     	scvtf	s3, x21
 1f82130: 9e2202c4     	scvtf	s4, x22
 1f82134: 1e241863     	fdiv	s3, s3, s4
 1f82138: 14000002     	b	0x1f82140 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x234>
 1f8213c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82140: 3dc000c4     	ldr	q4, [x6]
 1f82144: 4ea4d442     	fsub.4s	v2, v2, v4
 1f82148: 3dc000e5     	ldr	q5, [x7]
 1f8214c: 4e21d4a1     	fadd.4s	v1, v5, v1
 1f82150: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f82154: 9e2202e5     	scvtf	s5, x23
 1f82158: 4f859045     	fmul.4s	v5, v2, v5[0]
 1f8215c: 4e23d482     	fadd.4s	v2, v4, v3
 1f82160: 4e25cc61     	fmla.4s	v1, v3, v5
 1f82164: a93fd8bf     	stp	xzr, x22, [x5, #-0x8]
 1f82168: ad3f88c6     	stp	q6, q2, [x6, #-0x10]
 1f8216c: ad3f84e6     	stp	q6, q1, [x7, #-0x10]
 1f82170: eb09027f     	cmp	x19, x9
 1f82174: 54fff302     	b.hs	0x1f81fd4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0xc8>
 1f82178: d341fe97     	lsr	x23, x20, #1
 1f8217c: 91000673     	add	x19, x19, #0x1
 1f82180: 910040e7     	add	x7, x7, #0x10
 1f82184: 910040c6     	add	x6, x6, #0x10
 1f82188: 910020a5     	add	x5, x5, #0x8
 1f8218c: aa1603f5     	mov	x21, x22
 1f82190: 360ffc74     	tbz	w20, #0x1, 0x1f8211c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x210>
 1f82194: 17ffff90     	b	0x1f81fd4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0xc8>
 1f82198: f85b03aa     	ldur	x10, [x29, #-0x50]
 1f8219c: 3dc027e0     	ldr	q0, [sp, #0x90]
 1f821a0: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f821a4: f100052b     	subs	x11, x9, #0x1
 1f821a8: 540004c0     	b.eq	0x1f82240 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x334>
 1f821ac: d100092c     	sub	x12, x9, #0x2
 1f821b0: 92400569     	and	x9, x11, #0x3
 1f821b4: f1000d9f     	cmp	x12, #0x3
 1f821b8: 54000e62     	b.hs	0x1f82384 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x478>
 1f821bc: 5280002b     	mov	w11, #0x1               ; =1
 1f821c0: b4000409     	cbz	x9, 0x1f82240 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x334>
 1f821c4: d37ced6d     	lsl	x13, x11, #4
 1f821c8: 910143ec     	add	x12, sp, #0x50
 1f821cc: 8b0d018c     	add	x12, x12, x13
 1f821d0: 910243ee     	add	x14, sp, #0x90
 1f821d4: 8b0d01cd     	add	x13, x14, x13
 1f821d8: d10143ae     	sub	x14, x29, #0x50
 1f821dc: 8b0b0dcb     	add	x11, x14, x11, lsl #3
 1f821e0: 14000010     	b	0x1f82220 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x314>
 1f821e4: 9e2201e2     	scvtf	s2, x15
 1f821e8: 9e2201c3     	scvtf	s3, x14
 1f821ec: 1e231842     	fdiv	s2, s2, s3
 1f821f0: 3cc105a3     	ldr	q3, [x13], #0x10
 1f821f4: 4ea0d463     	fsub.4s	v3, v3, v0
 1f821f8: 3cc10584     	ldr	q4, [x12], #0x10
 1f821fc: 4e24d421     	fadd.4s	v1, v1, v4
 1f82200: 9e220144     	scvtf	s4, x10
 1f82204: 4f829062     	fmul.4s	v2, v3, v2[0]
 1f82208: 4f849063     	fmul.4s	v3, v3, v4[0]
 1f8220c: 4e22d400     	fadd.4s	v0, v0, v2
 1f82210: 4e23cc41     	fmla.4s	v1, v2, v3
 1f82214: aa0e03ea     	mov	x10, x14
 1f82218: f1000529     	subs	x9, x9, #0x1
 1f8221c: 54000120     	b.eq	0x1f82240 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x334>
 1f82220: f840856f     	ldr	x15, [x11], #0x8
 1f82224: ab0f014e     	adds	x14, x10, x15
 1f82228: 54fffde1     	b.ne	0x1f821e4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x2d8>
 1f8222c: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82230: 17fffff0     	b	0x1f821f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x2e4>
 1f82234: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82238: f100052b     	subs	x11, x9, #0x1
 1f8223c: 54fffb81     	b.ne	0x1f821ac <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x2a0>
 1f82240: d37ef509     	lsl	x9, x8, #2
 1f82244: 6f00e404     	movi.2d	v4, #0000000000000000
 1f82248: eb090029     	subs	x9, x1, x9
 1f8224c: 5400022d     	b.le	0x1f82290 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x384>
 1f82250: 8b08100a     	add	x10, x0, x8, lsl #4
 1f82254: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82258: 5280002b     	mov	w11, #0x1               ; =1
 1f8225c: aa0903ec     	mov	x12, x9
 1f82260: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82264: bc404545     	ldr	s5, [x10], #0x4
 1f82268: 1e2238a6     	fsub	s6, s5, s2
 1f8226c: 9e230167     	ucvtf	s7, x11
 1f82270: 1e2718c7     	fdiv	s7, s6, s7
 1f82274: 1e272842     	fadd	s2, s2, s7
 1f82278: 1e2238a5     	fsub	s5, s5, s2
 1f8227c: 1f050cc3     	fmadd	s3, s6, s5, s3
 1f82280: 9100056b     	add	x11, x11, #0x1
 1f82284: f100058c     	subs	x12, x12, #0x1
 1f82288: 54fffee1     	b.ne	0x1f82264 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x358>
 1f8228c: 14000004     	b	0x1f8229c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x390>
 1f82290: d2800009     	mov	x9, #0x0                ; =0
 1f82294: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82298: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8229c: 9e220106     	scvtf	s6, x8
 1f822a0: ab08012a     	adds	x10, x9, x8
 1f822a4: 9e220145     	scvtf	s5, x10
 1f822a8: 1e2518c7     	fdiv	s7, s6, s5
 1f822ac: 1e270c84     	fcsel	s4, s4, s7, eq
 1f822b0: ab08014a     	adds	x10, x10, x8
 1f822b4: 6f00e407     	movi.2d	v7, #0000000000000000
 1f822b8: 9e220150     	scvtf	s16, x10
 1f822bc: 1e3018d1     	fdiv	s17, s6, s16
 1f822c0: 1e310cf1     	fcsel	s17, s7, s17, eq
 1f822c4: ab08014a     	adds	x10, x10, x8
 1f822c8: 9e220152     	scvtf	s18, x10
 1f822cc: 1e3218d3     	fdiv	s19, s6, s18
 1f822d0: 1e330cf3     	fcsel	s19, s7, s19, eq
 1f822d4: ab080148     	adds	x8, x10, x8
 1f822d8: 9e220114     	scvtf	s20, x8
 1f822dc: 1e3418c6     	fdiv	s6, s6, s20
 1f822e0: 1e260ce6     	fcsel	s6, s7, s6, eq
 1f822e4: 1e223807     	fsub	s7, s0, s2
 1f822e8: 1e2708f4     	fmul	s20, s7, s7
 1f822ec: 1e340894     	fmul	s20, s4, s20
 1f822f0: 9e220135     	scvtf	s21, x9
 1f822f4: 1f150694     	fmadd	s20, s20, s21, s1
 1f822f8: 1e342863     	fadd	s3, s3, s20
 1f822fc: 5e0c0414     	mov	s20, v0[1]
 1f82300: 1f070882     	fmadd	s2, s4, s7, s2
 1f82304: 1e223a84     	fsub	s4, s20, s2
 1f82308: 1e240887     	fmul	s7, s4, s4
 1f8230c: 1e270a27     	fmul	s7, s17, s7
 1f82310: 5e0c0434     	mov	s20, v1[1]
 1f82314: 1f0550e5     	fmadd	s5, s7, s5, s20
 1f82318: 1e252863     	fadd	s3, s3, s5
 1f8231c: 5e140405     	mov	s5, v0[2]
 1f82320: 1f040a22     	fmadd	s2, s17, s4, s2
 1f82324: 1e2238a4     	fsub	s4, s5, s2
 1f82328: 1e240885     	fmul	s5, s4, s4
 1f8232c: 1e250a65     	fmul	s5, s19, s5
 1f82330: 5e140427     	mov	s7, v1[2]
 1f82334: 1f101ca5     	fmadd	s5, s5, s16, s7
 1f82338: 1e252863     	fadd	s3, s3, s5
 1f8233c: 1f040a62     	fmadd	s2, s19, s4, s2
 1f82340: 5e1c0400     	mov	s0, v0[3]
 1f82344: 1e223804     	fsub	s4, s0, s2
 1f82348: 1f0408c0     	fmadd	s0, s6, s4, s2
 1f8234c: 5e1c0421     	mov	s1, v1[3]
 1f82350: 1e240882     	fmul	s2, s4, s4
 1f82354: 1e2208c2     	fmul	s2, s6, s2
 1f82358: 1f120441     	fmadd	s1, s2, s18, s1
 1f8235c: 1e212861     	fadd	s1, s3, s1
 1f82360: cb020028     	sub	x8, x1, x2
 1f82364: 9e220102     	scvtf	s2, x8
 1f82368: 1e221821     	fdiv	s1, s1, s2
 1f8236c: a9527bfd     	ldp	x29, x30, [sp, #0x120]
 1f82370: a9514ff4     	ldp	x20, x19, [sp, #0x110]
 1f82374: a95057f6     	ldp	x22, x21, [sp, #0x100]
 1f82378: a94f5ff8     	ldp	x24, x23, [sp, #0xf0]
 1f8237c: 9104c3ff     	add	sp, sp, #0x130
 1f82380: d65f03c0     	ret
 1f82384: 927ef56c     	and	x12, x11, #0xfffffffffffffffc
 1f82388: 910243eb     	add	x11, sp, #0x90
 1f8238c: 9100816d     	add	x13, x11, #0x20
 1f82390: d10143ab     	sub	x11, x29, #0x50
 1f82394: 9100816e     	add	x14, x11, #0x20
 1f82398: 910143eb     	add	x11, sp, #0x50
 1f8239c: 9100816f     	add	x15, x11, #0x20
 1f823a0: 5280002b     	mov	w11, #0x1               ; =1
 1f823a4: 6f00e402     	movi.2d	v2, #0000000000000000
 1f823a8: 14000034     	b	0x1f82478 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x56c>
 1f823ac: 9e220064     	scvtf	s4, x3
 1f823b0: 9e220205     	scvtf	s5, x16
 1f823b4: 1e251884     	fdiv	s4, s4, s5
 1f823b8: ad7f99a5     	ldp	q5, q6, [x13, #-0x10]
 1f823bc: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f823c0: ad7fc1e7     	ldp	q7, q16, [x15, #-0x10]
 1f823c4: 4e27d421     	fadd.4s	v1, v1, v7
 1f823c8: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f823cc: 9e220147     	scvtf	s7, x10
 1f823d0: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f823d4: 4e23d400     	fadd.4s	v0, v0, v3
 1f823d8: 4e25cc61     	fmla.4s	v1, v3, v5
 1f823dc: 4ea0d4c3     	fsub.4s	v3, v6, v0
 1f823e0: 4e30d421     	fadd.4s	v1, v1, v16
 1f823e4: 4f849064     	fmul.4s	v4, v3, v4[0]
 1f823e8: 9e220225     	scvtf	s5, x17
 1f823ec: 4f859063     	fmul.4s	v3, v3, v5[0]
 1f823f0: 4e24d400     	fadd.4s	v0, v0, v4
 1f823f4: a97fc5ca     	ldp	x10, x17, [x14, #-0x8]
 1f823f8: 9e220145     	scvtf	s5, x10
 1f823fc: 4e23cc81     	fmla.4s	v1, v4, v3
 1f82400: ab0a020a     	adds	x10, x16, x10
 1f82404: 9e220143     	scvtf	s3, x10
 1f82408: 1e2318a4     	fdiv	s4, s5, s3
 1f8240c: 1e240c44     	fcsel	s4, s2, s4, eq
 1f82410: ad4099a5     	ldp	q5, q6, [x13, #0x10]
 1f82414: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f82418: ad40c1e7     	ldp	q7, q16, [x15, #0x10]
 1f8241c: 4e27d421     	fadd.4s	v1, v1, v7
 1f82420: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f82424: 9e220207     	scvtf	s7, x16
 1f82428: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f8242c: 4e24d400     	fadd.4s	v0, v0, v4
 1f82430: 4e25cc81     	fmla.4s	v1, v4, v5
 1f82434: ab11014a     	adds	x10, x10, x17
 1f82438: 9e220224     	scvtf	s4, x17
 1f8243c: 9e220145     	scvtf	s5, x10
 1f82440: 1e251884     	fdiv	s4, s4, s5
 1f82444: 1e240c44     	fcsel	s4, s2, s4, eq
 1f82448: 4ea0d4c5     	fsub.4s	v5, v6, v0
 1f8244c: 4e30d421     	fadd.4s	v1, v1, v16
 1f82450: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f82454: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f82458: 4e24d400     	fadd.4s	v0, v0, v4
 1f8245c: 9100116b     	add	x11, x11, #0x4
 1f82460: 4e23cc81     	fmla.4s	v1, v4, v3
 1f82464: 910101ad     	add	x13, x13, #0x40
 1f82468: 910081ce     	add	x14, x14, #0x20
 1f8246c: 910101ef     	add	x15, x15, #0x40
 1f82470: f100118c     	subs	x12, x12, #0x4
 1f82474: 54ffea60     	b.eq	0x1f821c0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x2b4>
 1f82478: f85e81d0     	ldur	x16, [x14, #-0x18]
 1f8247c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82480: ab100151     	adds	x17, x10, x16
 1f82484: 54000080     	b.eq	0x1f82494 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x588>
 1f82488: 9e220203     	scvtf	s3, x16
 1f8248c: 9e220224     	scvtf	s4, x17
 1f82490: 1e241863     	fdiv	s3, s3, s4
 1f82494: f85f01c3     	ldur	x3, [x14, #-0x10]
 1f82498: ab030230     	adds	x16, x17, x3
 1f8249c: 54fff881     	b.ne	0x1f823ac <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x4a0>
 1f824a0: 6f00e404     	movi.2d	v4, #0000000000000000
 1f824a4: 17ffffc5     	b	0x1f823b8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 4ll>(float const*, long long, long long)+0x4ac>

0000000001f824a8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)>:
 1f824a8: d10743ff     	sub	sp, sp, #0x1d0
 1f824ac: a9195ff8     	stp	x24, x23, [sp, #0x190]
 1f824b0: a91a57f6     	stp	x22, x21, [sp, #0x1a0]
 1f824b4: a91b4ff4     	stp	x20, x19, [sp, #0x1b0]
 1f824b8: a91c7bfd     	stp	x29, x30, [sp, #0x1c0]
 1f824bc: 910703fd     	add	x29, sp, #0x1c0
 1f824c0: d280000a     	mov	x10, #0x0               ; =0
 1f824c4: 91000c28     	add	x8, x1, #0x3
 1f824c8: f100003f     	cmp	x1, #0x0
 1f824cc: 9a81b108     	csel	x8, x8, x1, lt
 1f824d0: 9342fd08     	asr	x8, x8, #2
 1f824d4: b1003d09     	adds	x9, x8, #0xf
 1f824d8: 9100790b     	add	x11, x8, #0x1e
 1f824dc: 9a89b169     	csel	x9, x11, x9, lt
 1f824e0: 9344fd2b     	asr	x11, x9, #4
 1f824e4: d1000569     	sub	x9, x11, #0x1
 1f824e8: dac01129     	clz	x9, x9
 1f824ec: d2401529     	eor	x9, x9, #0x3f
 1f824f0: f102103f     	cmp	x1, #0x84
 1f824f4: 5280002c     	mov	w12, #0x1               ; =1
 1f824f8: 9a89b589     	csinc	x9, x12, x9, lt
 1f824fc: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82500: ad3d87a1     	stp	q1, q1, [x29, #-0x50]
 1f82504: ad3c87a1     	stp	q1, q1, [x29, #-0x70]
 1f82508: ad0987e1     	stp	q1, q1, [sp, #0x130]
 1f8250c: ad0887e1     	stp	q1, q1, [sp, #0x110]
 1f82510: ad0787e1     	stp	q1, q1, [sp, #0xf0]
 1f82514: ad0687e1     	stp	q1, q1, [sp, #0xd0]
 1f82518: ad0587e1     	stp	q1, q1, [sp, #0xb0]
 1f8251c: ad0487e1     	stp	q1, q1, [sp, #0x90]
 1f82520: ad0387e1     	stp	q1, q1, [sp, #0x70]
 1f82524: ad0287e1     	stp	q1, q1, [sp, #0x50]
 1f82528: b101ec3f     	cmn	x1, #0x7b
 1f8252c: 540015cb     	b.lt	0x1f827e4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x33c>
 1f82530: d100050c     	sub	x12, x8, #0x1
 1f82534: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82538: b100819f     	cmn	x12, #0x20
 1f8253c: 540010c8     	b.hi	0x1f82754 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2ac>
 1f82540: d2800004     	mov	x4, #0x0                ; =0
 1f82544: 910143ea     	add	x10, sp, #0x50
 1f82548: 9100414a     	add	x10, x10, #0x10
 1f8254c: 910343ec     	add	x12, sp, #0xd0
 1f82550: 9100418c     	add	x12, x12, #0x10
 1f82554: d101c3ad     	sub	x13, x29, #0x70
 1f82558: b27d01ad     	orr	x13, x13, #0x8
 1f8255c: 5280020f     	mov	w15, #0x10              ; =16
 1f82560: 900880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82564: 911e81ce     	add	x14, x14, #0x7a0
 1f82568: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8256c: 6f00e406     	movi.2d	v6, #0000000000000000
 1f82570: aa0003f0     	mov	x16, x0
 1f82574: aa0803e3     	mov	x3, x8
 1f82578: 900880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f8257c: 911a8231     	add	x17, x17, #0x6a0
 1f82580: 14000004     	b	0x1f82590 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0xe8>
 1f82584: 91040210     	add	x16, x16, #0x100
 1f82588: eb0b009f     	cmp	x4, x11
 1f8258c: 54000de0     	b.eq	0x1f82748 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2a0>
 1f82590: aa0403e5     	mov	x5, x4
 1f82594: aa0303e4     	mov	x4, x3
 1f82598: f1004063     	subs	x3, x3, #0x10
 1f8259c: 9a8fb094     	csel	x20, x4, x15, lt
 1f825a0: cb051115     	sub	x21, x8, x5, lsl #4
 1f825a4: f10042bf     	cmp	x21, #0x10
 1f825a8: 9a8fb2b3     	csel	x19, x21, x15, lt
 1f825ac: 38bfc1c4     	ldaprb	w4, [x14]
 1f825b0: 36000204     	tbz	w4, #0x0, 0x1f825f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x148>
 1f825b4: f10006bf     	cmp	x21, #0x1
 1f825b8: 5400050b     	b.lt	0x1f82658 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x1b0>
 1f825bc: 6f00e401     	movi.2d	v1, #0000000000000000
 1f825c0: aa1003e4     	mov	x4, x16
 1f825c4: aa1103e6     	mov	x6, x17
 1f825c8: 6f00e402     	movi.2d	v2, #0000000000000000
 1f825cc: 3cc10483     	ldr	q3, [x4], #0x10
 1f825d0: 3cc104c4     	ldr	q4, [x6], #0x10
 1f825d4: 4ea1d465     	fsub.4s	v5, v3, v1
 1f825d8: 4e24cca1     	fmla.4s	v1, v5, v4
 1f825dc: 4ea1d463     	fsub.4s	v3, v3, v1
 1f825e0: 4e25cc62     	fmla.4s	v2, v3, v5
 1f825e4: f1000694     	subs	x20, x20, #0x1
 1f825e8: 54ffff21     	b.ne	0x1f825cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x124>
 1f825ec: 1400001d     	b	0x1f82660 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x1b8>
 1f825f0: a9010be0     	stp	x0, x2, [sp, #0x10]
 1f825f4: aa0103f6     	mov	x22, x1
 1f825f8: aa0803f7     	mov	x23, x8
 1f825fc: a902a7ea     	stp	x10, x9, [sp, #0x28]
 1f82600: a9002fec     	stp	x12, x11, [sp]
 1f82604: f90013ed     	str	x13, [sp, #0x20]
 1f82608: a90443e3     	stp	x3, x16, [sp, #0x40]
 1f8260c: f9001fe5     	str	x5, [sp, #0x38]
 1f82610: 94c0e890     	bl	0x4fbc850 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long) (.cold.1)>
 1f82614: 6f00e406     	movi.2d	v6, #0000000000000000
 1f82618: a9438fe5     	ldp	x5, x3, [sp, #0x38]
 1f8261c: 900880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82620: 911a8231     	add	x17, x17, #0x6a0
 1f82624: f94027f0     	ldr	x16, [sp, #0x48]
 1f82628: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8262c: 900880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82630: 911e81ce     	add	x14, x14, #0x7a0
 1f82634: 5280020f     	mov	w15, #0x10              ; =16
 1f82638: a9422bed     	ldp	x13, x10, [sp, #0x20]
 1f8263c: a9402fec     	ldp	x12, x11, [sp]
 1f82640: f9401be9     	ldr	x9, [sp, #0x30]
 1f82644: aa1703e8     	mov	x8, x23
 1f82648: a9410be0     	ldp	x0, x2, [sp, #0x10]
 1f8264c: aa1603e1     	mov	x1, x22
 1f82650: f10006bf     	cmp	x21, #0x1
 1f82654: 54fffb4a     	b.ge	0x1f825bc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x114>
 1f82658: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8265c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82660: f85903a4     	ldur	x4, [x29, #-0x70]
 1f82664: ab130095     	adds	x21, x4, x19
 1f82668: 9e220263     	scvtf	s3, x19
 1f8266c: 9e2202a4     	scvtf	s4, x21
 1f82670: 1e241863     	fdiv	s3, s3, s4
 1f82674: 1e230c03     	fcsel	s3, s0, s3, eq
 1f82678: 3dc037e4     	ldr	q4, [sp, #0xd0]
 1f8267c: 4ea4d425     	fsub.4s	v5, v1, v4
 1f82680: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f82684: 4e21d441     	fadd.4s	v1, v2, v1
 1f82688: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f8268c: 9e220082     	scvtf	s2, x4
 1f82690: 4f8290a5     	fmul.4s	v5, v5, v2[0]
 1f82694: 4e23d482     	fadd.4s	v2, v4, v3
 1f82698: 3d8037e2     	str	q2, [sp, #0xd0]
 1f8269c: 4e25cc61     	fmla.4s	v1, v3, v5
 1f826a0: 3d8017e1     	str	q1, [sp, #0x50]
 1f826a4: f81903b5     	stur	x21, [x29, #-0x70]
 1f826a8: 910004a4     	add	x4, x5, #0x1
 1f826ac: f100093f     	cmp	x9, #0x2
 1f826b0: 54fff6a3     	b.lo	0x1f82584 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0xdc>
 1f826b4: 3607f685     	tbz	w5, #0x0, 0x1f82584 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0xdc>
 1f826b8: aa0d03e5     	mov	x5, x13
 1f826bc: aa0c03e6     	mov	x6, x12
 1f826c0: aa0a03e7     	mov	x7, x10
 1f826c4: 52800053     	mov	w19, #0x2               ; =2
 1f826c8: aa0403f7     	mov	x23, x4
 1f826cc: aa1703f4     	mov	x20, x23
 1f826d0: f94000b7     	ldr	x23, [x5]
 1f826d4: ab1502f6     	adds	x22, x23, x21
 1f826d8: 540000a0     	b.eq	0x1f826ec <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x244>
 1f826dc: 9e2202a3     	scvtf	s3, x21
 1f826e0: 9e2202c4     	scvtf	s4, x22
 1f826e4: 1e241863     	fdiv	s3, s3, s4
 1f826e8: 14000002     	b	0x1f826f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x248>
 1f826ec: 6f00e403     	movi.2d	v3, #0000000000000000
 1f826f0: 3dc000c4     	ldr	q4, [x6]
 1f826f4: 4ea4d442     	fsub.4s	v2, v2, v4
 1f826f8: 3dc000e5     	ldr	q5, [x7]
 1f826fc: 4e21d4a1     	fadd.4s	v1, v5, v1
 1f82700: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f82704: 9e2202e5     	scvtf	s5, x23
 1f82708: 4f859045     	fmul.4s	v5, v2, v5[0]
 1f8270c: 4e23d482     	fadd.4s	v2, v4, v3
 1f82710: 4e25cc61     	fmla.4s	v1, v3, v5
 1f82714: a93fd8bf     	stp	xzr, x22, [x5, #-0x8]
 1f82718: ad3f88c6     	stp	q6, q2, [x6, #-0x10]
 1f8271c: ad3f84e6     	stp	q6, q1, [x7, #-0x10]
 1f82720: eb09027f     	cmp	x19, x9
 1f82724: 54fff302     	b.hs	0x1f82584 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0xdc>
 1f82728: d341fe97     	lsr	x23, x20, #1
 1f8272c: 91000673     	add	x19, x19, #0x1
 1f82730: 910040e7     	add	x7, x7, #0x10
 1f82734: 910040c6     	add	x6, x6, #0x10
 1f82738: 910020a5     	add	x5, x5, #0x8
 1f8273c: aa1603f5     	mov	x21, x22
 1f82740: 360ffc74     	tbz	w20, #0x1, 0x1f826cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x224>
 1f82744: 17ffff90     	b	0x1f82584 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0xdc>
 1f82748: f85903aa     	ldur	x10, [x29, #-0x70]
 1f8274c: 3dc037e0     	ldr	q0, [sp, #0xd0]
 1f82750: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f82754: f100052b     	subs	x11, x9, #0x1
 1f82758: 540004c0     	b.eq	0x1f827f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x348>
 1f8275c: d100092c     	sub	x12, x9, #0x2
 1f82760: 92400569     	and	x9, x11, #0x3
 1f82764: f1000d9f     	cmp	x12, #0x3
 1f82768: 54000e62     	b.hs	0x1f82934 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x48c>
 1f8276c: 5280002b     	mov	w11, #0x1               ; =1
 1f82770: b4000409     	cbz	x9, 0x1f827f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x348>
 1f82774: d37ced6d     	lsl	x13, x11, #4
 1f82778: 910143ec     	add	x12, sp, #0x50
 1f8277c: 8b0d018c     	add	x12, x12, x13
 1f82780: 910343ee     	add	x14, sp, #0xd0
 1f82784: 8b0d01cd     	add	x13, x14, x13
 1f82788: d101c3ae     	sub	x14, x29, #0x70
 1f8278c: 8b0b0dcb     	add	x11, x14, x11, lsl #3
 1f82790: 14000010     	b	0x1f827d0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x328>
 1f82794: 9e2201e2     	scvtf	s2, x15
 1f82798: 9e2201c3     	scvtf	s3, x14
 1f8279c: 1e231842     	fdiv	s2, s2, s3
 1f827a0: 3cc105a3     	ldr	q3, [x13], #0x10
 1f827a4: 4ea0d463     	fsub.4s	v3, v3, v0
 1f827a8: 3cc10584     	ldr	q4, [x12], #0x10
 1f827ac: 4e24d421     	fadd.4s	v1, v1, v4
 1f827b0: 9e220144     	scvtf	s4, x10
 1f827b4: 4f829062     	fmul.4s	v2, v3, v2[0]
 1f827b8: 4f849063     	fmul.4s	v3, v3, v4[0]
 1f827bc: 4e22d400     	fadd.4s	v0, v0, v2
 1f827c0: 4e23cc41     	fmla.4s	v1, v2, v3
 1f827c4: aa0e03ea     	mov	x10, x14
 1f827c8: f1000529     	subs	x9, x9, #0x1
 1f827cc: 54000120     	b.eq	0x1f827f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x348>
 1f827d0: f840856f     	ldr	x15, [x11], #0x8
 1f827d4: ab0f014e     	adds	x14, x10, x15
 1f827d8: 54fffde1     	b.ne	0x1f82794 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2ec>
 1f827dc: 6f00e402     	movi.2d	v2, #0000000000000000
 1f827e0: 17fffff0     	b	0x1f827a0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2f8>
 1f827e4: 6f00e400     	movi.2d	v0, #0000000000000000
 1f827e8: f100052b     	subs	x11, x9, #0x1
 1f827ec: 54fffb81     	b.ne	0x1f8275c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2b4>
 1f827f0: d37ef509     	lsl	x9, x8, #2
 1f827f4: 6f00e404     	movi.2d	v4, #0000000000000000
 1f827f8: eb090029     	subs	x9, x1, x9
 1f827fc: 5400022d     	b.le	0x1f82840 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x398>
 1f82800: 8b08100a     	add	x10, x0, x8, lsl #4
 1f82804: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82808: 5280002b     	mov	w11, #0x1               ; =1
 1f8280c: aa0903ec     	mov	x12, x9
 1f82810: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82814: bc404545     	ldr	s5, [x10], #0x4
 1f82818: 1e2238a6     	fsub	s6, s5, s2
 1f8281c: 9e230167     	ucvtf	s7, x11
 1f82820: 1e2718c7     	fdiv	s7, s6, s7
 1f82824: 1e272842     	fadd	s2, s2, s7
 1f82828: 1e2238a5     	fsub	s5, s5, s2
 1f8282c: 1f050cc3     	fmadd	s3, s6, s5, s3
 1f82830: 9100056b     	add	x11, x11, #0x1
 1f82834: f100058c     	subs	x12, x12, #0x1
 1f82838: 54fffee1     	b.ne	0x1f82814 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x36c>
 1f8283c: 14000004     	b	0x1f8284c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x3a4>
 1f82840: d2800009     	mov	x9, #0x0                ; =0
 1f82844: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82848: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8284c: 9e220106     	scvtf	s6, x8
 1f82850: ab08012a     	adds	x10, x9, x8
 1f82854: 9e220145     	scvtf	s5, x10
 1f82858: 1e2518c7     	fdiv	s7, s6, s5
 1f8285c: 1e270c84     	fcsel	s4, s4, s7, eq
 1f82860: ab08014a     	adds	x10, x10, x8
 1f82864: 6f00e407     	movi.2d	v7, #0000000000000000
 1f82868: 9e220150     	scvtf	s16, x10
 1f8286c: 1e3018d1     	fdiv	s17, s6, s16
 1f82870: 1e310cf1     	fcsel	s17, s7, s17, eq
 1f82874: ab08014a     	adds	x10, x10, x8
 1f82878: 9e220152     	scvtf	s18, x10
 1f8287c: 1e3218d3     	fdiv	s19, s6, s18
 1f82880: 1e330cf3     	fcsel	s19, s7, s19, eq
 1f82884: ab080148     	adds	x8, x10, x8
 1f82888: 9e220114     	scvtf	s20, x8
 1f8288c: 1e3418c6     	fdiv	s6, s6, s20
 1f82890: 1e260ce6     	fcsel	s6, s7, s6, eq
 1f82894: 1e223807     	fsub	s7, s0, s2
 1f82898: 1e2708f4     	fmul	s20, s7, s7
 1f8289c: 1e340894     	fmul	s20, s4, s20
 1f828a0: 9e220135     	scvtf	s21, x9
 1f828a4: 1f150694     	fmadd	s20, s20, s21, s1
 1f828a8: 1e342863     	fadd	s3, s3, s20
 1f828ac: 5e0c0414     	mov	s20, v0[1]
 1f828b0: 1f070882     	fmadd	s2, s4, s7, s2
 1f828b4: 1e223a84     	fsub	s4, s20, s2
 1f828b8: 1e240887     	fmul	s7, s4, s4
 1f828bc: 1e270a27     	fmul	s7, s17, s7
 1f828c0: 5e0c0434     	mov	s20, v1[1]
 1f828c4: 1f0550e5     	fmadd	s5, s7, s5, s20
 1f828c8: 1e252863     	fadd	s3, s3, s5
 1f828cc: 5e140405     	mov	s5, v0[2]
 1f828d0: 1f040a22     	fmadd	s2, s17, s4, s2
 1f828d4: 1e2238a4     	fsub	s4, s5, s2
 1f828d8: 1e240885     	fmul	s5, s4, s4
 1f828dc: 1e250a65     	fmul	s5, s19, s5
 1f828e0: 5e140427     	mov	s7, v1[2]
 1f828e4: 1f101ca5     	fmadd	s5, s5, s16, s7
 1f828e8: 1e252863     	fadd	s3, s3, s5
 1f828ec: 1f040a62     	fmadd	s2, s19, s4, s2
 1f828f0: 5e1c0400     	mov	s0, v0[3]
 1f828f4: 1e223804     	fsub	s4, s0, s2
 1f828f8: 1f0408c0     	fmadd	s0, s6, s4, s2
 1f828fc: 5e1c0421     	mov	s1, v1[3]
 1f82900: 1e240882     	fmul	s2, s4, s4
 1f82904: 1e2208c2     	fmul	s2, s6, s2
 1f82908: 1f120441     	fmadd	s1, s2, s18, s1
 1f8290c: 1e212861     	fadd	s1, s3, s1
 1f82910: cb020028     	sub	x8, x1, x2
 1f82914: 9e220102     	scvtf	s2, x8
 1f82918: 1e221821     	fdiv	s1, s1, s2
 1f8291c: a95c7bfd     	ldp	x29, x30, [sp, #0x1c0]
 1f82920: a95b4ff4     	ldp	x20, x19, [sp, #0x1b0]
 1f82924: a95a57f6     	ldp	x22, x21, [sp, #0x1a0]
 1f82928: a9595ff8     	ldp	x24, x23, [sp, #0x190]
 1f8292c: 910743ff     	add	sp, sp, #0x1d0
 1f82930: d65f03c0     	ret
 1f82934: 927ef56c     	and	x12, x11, #0xfffffffffffffffc
 1f82938: 910343eb     	add	x11, sp, #0xd0
 1f8293c: 9100816d     	add	x13, x11, #0x20
 1f82940: d101c3ab     	sub	x11, x29, #0x70
 1f82944: 9100816e     	add	x14, x11, #0x20
 1f82948: 910143eb     	add	x11, sp, #0x50
 1f8294c: 9100816f     	add	x15, x11, #0x20
 1f82950: 5280002b     	mov	w11, #0x1               ; =1
 1f82954: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82958: 14000034     	b	0x1f82a28 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x580>
 1f8295c: 9e220064     	scvtf	s4, x3
 1f82960: 9e220205     	scvtf	s5, x16
 1f82964: 1e251884     	fdiv	s4, s4, s5
 1f82968: ad7f99a5     	ldp	q5, q6, [x13, #-0x10]
 1f8296c: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f82970: ad7fc1e7     	ldp	q7, q16, [x15, #-0x10]
 1f82974: 4e27d421     	fadd.4s	v1, v1, v7
 1f82978: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f8297c: 9e220147     	scvtf	s7, x10
 1f82980: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f82984: 4e23d400     	fadd.4s	v0, v0, v3
 1f82988: 4e25cc61     	fmla.4s	v1, v3, v5
 1f8298c: 4ea0d4c3     	fsub.4s	v3, v6, v0
 1f82990: 4e30d421     	fadd.4s	v1, v1, v16
 1f82994: 4f849064     	fmul.4s	v4, v3, v4[0]
 1f82998: 9e220225     	scvtf	s5, x17
 1f8299c: 4f859063     	fmul.4s	v3, v3, v5[0]
 1f829a0: 4e24d400     	fadd.4s	v0, v0, v4
 1f829a4: a97fc5ca     	ldp	x10, x17, [x14, #-0x8]
 1f829a8: 9e220145     	scvtf	s5, x10
 1f829ac: 4e23cc81     	fmla.4s	v1, v4, v3
 1f829b0: ab0a020a     	adds	x10, x16, x10
 1f829b4: 9e220143     	scvtf	s3, x10
 1f829b8: 1e2318a4     	fdiv	s4, s5, s3
 1f829bc: 1e240c44     	fcsel	s4, s2, s4, eq
 1f829c0: ad4099a5     	ldp	q5, q6, [x13, #0x10]
 1f829c4: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f829c8: ad40c1e7     	ldp	q7, q16, [x15, #0x10]
 1f829cc: 4e27d421     	fadd.4s	v1, v1, v7
 1f829d0: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f829d4: 9e220207     	scvtf	s7, x16
 1f829d8: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f829dc: 4e24d400     	fadd.4s	v0, v0, v4
 1f829e0: 4e25cc81     	fmla.4s	v1, v4, v5
 1f829e4: ab11014a     	adds	x10, x10, x17
 1f829e8: 9e220224     	scvtf	s4, x17
 1f829ec: 9e220145     	scvtf	s5, x10
 1f829f0: 1e251884     	fdiv	s4, s4, s5
 1f829f4: 1e240c44     	fcsel	s4, s2, s4, eq
 1f829f8: 4ea0d4c5     	fsub.4s	v5, v6, v0
 1f829fc: 4e30d421     	fadd.4s	v1, v1, v16
 1f82a00: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f82a04: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f82a08: 4e24d400     	fadd.4s	v0, v0, v4
 1f82a0c: 9100116b     	add	x11, x11, #0x4
 1f82a10: 4e23cc81     	fmla.4s	v1, v4, v3
 1f82a14: 910101ad     	add	x13, x13, #0x40
 1f82a18: 910081ce     	add	x14, x14, #0x20
 1f82a1c: 910101ef     	add	x15, x15, #0x40
 1f82a20: f100118c     	subs	x12, x12, #0x4
 1f82a24: 54ffea60     	b.eq	0x1f82770 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x2c8>
 1f82a28: f85e81d0     	ldur	x16, [x14, #-0x18]
 1f82a2c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82a30: ab100151     	adds	x17, x10, x16
 1f82a34: 54000080     	b.eq	0x1f82a44 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x59c>
 1f82a38: 9e220203     	scvtf	s3, x16
 1f82a3c: 9e220224     	scvtf	s4, x17
 1f82a40: 1e241863     	fdiv	s3, s3, s4
 1f82a44: f85f01c3     	ldur	x3, [x14, #-0x10]
 1f82a48: ab030230     	adds	x16, x17, x3
 1f82a4c: 54fff881     	b.ne	0x1f8295c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x4b4>
 1f82a50: 6f00e404     	movi.2d	v4, #0000000000000000
 1f82a54: 17ffffc5     	b	0x1f82968 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 8ll>(float const*, long long, long long)+0x4c0>

0000000001f82a58 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)>:
 1f82a58: a9bc5ff8     	stp	x24, x23, [sp, #-0x40]!
 1f82a5c: a90157f6     	stp	x22, x21, [sp, #0x10]
 1f82a60: a9024ff4     	stp	x20, x19, [sp, #0x20]
 1f82a64: a9037bfd     	stp	x29, x30, [sp, #0x30]
 1f82a68: 9100c3fd     	add	x29, sp, #0x30
 1f82a6c: d10b43ff     	sub	sp, sp, #0x2d0
 1f82a70: d2800009     	mov	x9, #0x0                ; =0
 1f82a74: 91000c28     	add	x8, x1, #0x3
 1f82a78: f100003f     	cmp	x1, #0x0
 1f82a7c: 9a81b108     	csel	x8, x8, x1, lt
 1f82a80: 9342fd08     	asr	x8, x8, #2
 1f82a84: b1003d0a     	adds	x10, x8, #0xf
 1f82a88: 9100790b     	add	x11, x8, #0x1e
 1f82a8c: 9a8ab16a     	csel	x10, x11, x10, lt
 1f82a90: 9344fd4b     	asr	x11, x10, #4
 1f82a94: d100056a     	sub	x10, x11, #0x1
 1f82a98: dac0114a     	clz	x10, x10
 1f82a9c: d240154a     	eor	x10, x10, #0x3f
 1f82aa0: f102103f     	cmp	x1, #0x84
 1f82aa4: 5280002c     	mov	w12, #0x1               ; =1
 1f82aa8: 9a8ab58a     	csinc	x10, x12, x10, lt
 1f82aac: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82ab0: ad3d87a1     	stp	q1, q1, [x29, #-0x50]
 1f82ab4: ad3c87a1     	stp	q1, q1, [x29, #-0x70]
 1f82ab8: ad3b87a1     	stp	q1, q1, [x29, #-0x90]
 1f82abc: ad3a87a1     	stp	q1, q1, [x29, #-0xb0]
 1f82ac0: ad1187e1     	stp	q1, q1, [sp, #0x230]
 1f82ac4: ad1087e1     	stp	q1, q1, [sp, #0x210]
 1f82ac8: ad0f87e1     	stp	q1, q1, [sp, #0x1f0]
 1f82acc: ad0e87e1     	stp	q1, q1, [sp, #0x1d0]
 1f82ad0: ad0d87e1     	stp	q1, q1, [sp, #0x1b0]
 1f82ad4: ad0c87e1     	stp	q1, q1, [sp, #0x190]
 1f82ad8: ad0b87e1     	stp	q1, q1, [sp, #0x170]
 1f82adc: ad0a87e1     	stp	q1, q1, [sp, #0x150]
 1f82ae0: ad0987e1     	stp	q1, q1, [sp, #0x130]
 1f82ae4: ad0887e1     	stp	q1, q1, [sp, #0x110]
 1f82ae8: ad0787e1     	stp	q1, q1, [sp, #0xf0]
 1f82aec: ad0687e1     	stp	q1, q1, [sp, #0xd0]
 1f82af0: ad0587e1     	stp	q1, q1, [sp, #0xb0]
 1f82af4: ad0487e1     	stp	q1, q1, [sp, #0x90]
 1f82af8: ad0387e1     	stp	q1, q1, [sp, #0x70]
 1f82afc: ad0287e1     	stp	q1, q1, [sp, #0x50]
 1f82b00: b101ec3f     	cmn	x1, #0x7b
 1f82b04: 540015cb     	b.lt	0x1f82dbc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x364>
 1f82b08: d100050c     	sub	x12, x8, #0x1
 1f82b0c: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82b10: b100819f     	cmn	x12, #0x20
 1f82b14: 540010c8     	b.hi	0x1f82d2c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x2d4>
 1f82b18: d2800004     	mov	x4, #0x0                ; =0
 1f82b1c: 910143e9     	add	x9, sp, #0x50
 1f82b20: 91004129     	add	x9, x9, #0x10
 1f82b24: 910543ec     	add	x12, sp, #0x150
 1f82b28: 9100418c     	add	x12, x12, #0x10
 1f82b2c: d102c3ad     	sub	x13, x29, #0xb0
 1f82b30: b27d01ad     	orr	x13, x13, #0x8
 1f82b34: 5280020f     	mov	w15, #0x10              ; =16
 1f82b38: 900880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82b3c: 9122c1ce     	add	x14, x14, #0x8b0
 1f82b40: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82b44: 6f00e406     	movi.2d	v6, #0000000000000000
 1f82b48: aa0003f0     	mov	x16, x0
 1f82b4c: aa0803e3     	mov	x3, x8
 1f82b50: 900880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82b54: 911ec231     	add	x17, x17, #0x7b0
 1f82b58: 14000004     	b	0x1f82b68 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x110>
 1f82b5c: 91040210     	add	x16, x16, #0x100
 1f82b60: eb0b009f     	cmp	x4, x11
 1f82b64: 54000de0     	b.eq	0x1f82d20 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x2c8>
 1f82b68: aa0403e5     	mov	x5, x4
 1f82b6c: aa0303e4     	mov	x4, x3
 1f82b70: f1004063     	subs	x3, x3, #0x10
 1f82b74: 9a8fb094     	csel	x20, x4, x15, lt
 1f82b78: cb051115     	sub	x21, x8, x5, lsl #4
 1f82b7c: f10042bf     	cmp	x21, #0x10
 1f82b80: 9a8fb2b3     	csel	x19, x21, x15, lt
 1f82b84: 38bfc1c4     	ldaprb	w4, [x14]
 1f82b88: 36000204     	tbz	w4, #0x0, 0x1f82bc8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x170>
 1f82b8c: f10006bf     	cmp	x21, #0x1
 1f82b90: 5400050b     	b.lt	0x1f82c30 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x1d8>
 1f82b94: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82b98: aa1003e4     	mov	x4, x16
 1f82b9c: aa1103e6     	mov	x6, x17
 1f82ba0: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82ba4: 3cc10483     	ldr	q3, [x4], #0x10
 1f82ba8: 3cc104c4     	ldr	q4, [x6], #0x10
 1f82bac: 4ea1d465     	fsub.4s	v5, v3, v1
 1f82bb0: 4e24cca1     	fmla.4s	v1, v5, v4
 1f82bb4: 4ea1d463     	fsub.4s	v3, v3, v1
 1f82bb8: 4e25cc62     	fmla.4s	v2, v3, v5
 1f82bbc: f1000694     	subs	x20, x20, #0x1
 1f82bc0: 54ffff21     	b.ne	0x1f82ba4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x14c>
 1f82bc4: 1400001d     	b	0x1f82c38 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x1e0>
 1f82bc8: a9010be0     	stp	x0, x2, [sp, #0x10]
 1f82bcc: aa0103f6     	mov	x22, x1
 1f82bd0: aa0803f7     	mov	x23, x8
 1f82bd4: a902abe9     	stp	x9, x10, [sp, #0x28]
 1f82bd8: a9002fec     	stp	x12, x11, [sp]
 1f82bdc: f90013ed     	str	x13, [sp, #0x20]
 1f82be0: a90443e3     	stp	x3, x16, [sp, #0x40]
 1f82be4: f9001fe5     	str	x5, [sp, #0x38]
 1f82be8: 94c0e72a     	bl	0x4fbc890 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long) (.cold.1)>
 1f82bec: 6f00e406     	movi.2d	v6, #0000000000000000
 1f82bf0: a9438fe5     	ldp	x5, x3, [sp, #0x38]
 1f82bf4: 900880b1     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82bf8: 911ec231     	add	x17, x17, #0x7b0
 1f82bfc: f94027f0     	ldr	x16, [sp, #0x48]
 1f82c00: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82c04: 900880ae     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f82c08: 9122c1ce     	add	x14, x14, #0x8b0
 1f82c0c: 5280020f     	mov	w15, #0x10              ; =16
 1f82c10: a94227ed     	ldp	x13, x9, [sp, #0x20]
 1f82c14: a9402fec     	ldp	x12, x11, [sp]
 1f82c18: f9401bea     	ldr	x10, [sp, #0x30]
 1f82c1c: aa1703e8     	mov	x8, x23
 1f82c20: a9410be0     	ldp	x0, x2, [sp, #0x10]
 1f82c24: aa1603e1     	mov	x1, x22
 1f82c28: f10006bf     	cmp	x21, #0x1
 1f82c2c: 54fffb4a     	b.ge	0x1f82b94 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x13c>
 1f82c30: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82c34: 6f00e401     	movi.2d	v1, #0000000000000000
 1f82c38: f85503a4     	ldur	x4, [x29, #-0xb0]
 1f82c3c: ab130095     	adds	x21, x4, x19
 1f82c40: 9e220263     	scvtf	s3, x19
 1f82c44: 9e2202a4     	scvtf	s4, x21
 1f82c48: 1e241863     	fdiv	s3, s3, s4
 1f82c4c: 1e230c03     	fcsel	s3, s0, s3, eq
 1f82c50: 3dc057e4     	ldr	q4, [sp, #0x150]
 1f82c54: 4ea4d425     	fsub.4s	v5, v1, v4
 1f82c58: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f82c5c: 4e21d441     	fadd.4s	v1, v2, v1
 1f82c60: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f82c64: 9e220082     	scvtf	s2, x4
 1f82c68: 4f8290a5     	fmul.4s	v5, v5, v2[0]
 1f82c6c: 4e23d482     	fadd.4s	v2, v4, v3
 1f82c70: 3d8057e2     	str	q2, [sp, #0x150]
 1f82c74: 4e25cc61     	fmla.4s	v1, v3, v5
 1f82c78: 3d8017e1     	str	q1, [sp, #0x50]
 1f82c7c: f81503b5     	stur	x21, [x29, #-0xb0]
 1f82c80: 910004a4     	add	x4, x5, #0x1
 1f82c84: f100095f     	cmp	x10, #0x2
 1f82c88: 54fff6a3     	b.lo	0x1f82b5c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x104>
 1f82c8c: 3607f685     	tbz	w5, #0x0, 0x1f82b5c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x104>
 1f82c90: aa0d03e5     	mov	x5, x13
 1f82c94: aa0c03e6     	mov	x6, x12
 1f82c98: aa0903e7     	mov	x7, x9
 1f82c9c: 52800053     	mov	w19, #0x2               ; =2
 1f82ca0: aa0403f7     	mov	x23, x4
 1f82ca4: aa1703f4     	mov	x20, x23
 1f82ca8: f94000b7     	ldr	x23, [x5]
 1f82cac: ab1502f6     	adds	x22, x23, x21
 1f82cb0: 540000a0     	b.eq	0x1f82cc4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x26c>
 1f82cb4: 9e2202a3     	scvtf	s3, x21
 1f82cb8: 9e2202c4     	scvtf	s4, x22
 1f82cbc: 1e241863     	fdiv	s3, s3, s4
 1f82cc0: 14000002     	b	0x1f82cc8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x270>
 1f82cc4: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82cc8: 3dc000c4     	ldr	q4, [x6]
 1f82ccc: 4ea4d442     	fsub.4s	v2, v2, v4
 1f82cd0: 3dc000e5     	ldr	q5, [x7]
 1f82cd4: 4e21d4a1     	fadd.4s	v1, v5, v1
 1f82cd8: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f82cdc: 9e2202e5     	scvtf	s5, x23
 1f82ce0: 4f859045     	fmul.4s	v5, v2, v5[0]
 1f82ce4: 4e23d482     	fadd.4s	v2, v4, v3
 1f82ce8: 4e25cc61     	fmla.4s	v1, v3, v5
 1f82cec: a93fd8bf     	stp	xzr, x22, [x5, #-0x8]
 1f82cf0: ad3f88c6     	stp	q6, q2, [x6, #-0x10]
 1f82cf4: ad3f84e6     	stp	q6, q1, [x7, #-0x10]
 1f82cf8: eb0a027f     	cmp	x19, x10
 1f82cfc: 54fff302     	b.hs	0x1f82b5c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x104>
 1f82d00: d341fe97     	lsr	x23, x20, #1
 1f82d04: 91000673     	add	x19, x19, #0x1
 1f82d08: 910040e7     	add	x7, x7, #0x10
 1f82d0c: 910040c6     	add	x6, x6, #0x10
 1f82d10: 910020a5     	add	x5, x5, #0x8
 1f82d14: aa1603f5     	mov	x21, x22
 1f82d18: 360ffc74     	tbz	w20, #0x1, 0x1f82ca4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x24c>
 1f82d1c: 17ffff90     	b	0x1f82b5c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x104>
 1f82d20: f85503a9     	ldur	x9, [x29, #-0xb0]
 1f82d24: 3dc057e0     	ldr	q0, [sp, #0x150]
 1f82d28: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f82d2c: f100054b     	subs	x11, x10, #0x1
 1f82d30: 540004c0     	b.eq	0x1f82dc8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x370>
 1f82d34: d100094c     	sub	x12, x10, #0x2
 1f82d38: 9240056a     	and	x10, x11, #0x3
 1f82d3c: f1000d9f     	cmp	x12, #0x3
 1f82d40: 54000e62     	b.hs	0x1f82f0c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x4b4>
 1f82d44: 5280002b     	mov	w11, #0x1               ; =1
 1f82d48: b400040a     	cbz	x10, 0x1f82dc8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x370>
 1f82d4c: d37ced6d     	lsl	x13, x11, #4
 1f82d50: 910143ec     	add	x12, sp, #0x50
 1f82d54: 8b0d018c     	add	x12, x12, x13
 1f82d58: 910543ee     	add	x14, sp, #0x150
 1f82d5c: 8b0d01cd     	add	x13, x14, x13
 1f82d60: d102c3ae     	sub	x14, x29, #0xb0
 1f82d64: 8b0b0dcb     	add	x11, x14, x11, lsl #3
 1f82d68: 14000010     	b	0x1f82da8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x350>
 1f82d6c: 9e2201e2     	scvtf	s2, x15
 1f82d70: 9e2201c3     	scvtf	s3, x14
 1f82d74: 1e231842     	fdiv	s2, s2, s3
 1f82d78: 3cc105a3     	ldr	q3, [x13], #0x10
 1f82d7c: 4ea0d463     	fsub.4s	v3, v3, v0
 1f82d80: 3cc10584     	ldr	q4, [x12], #0x10
 1f82d84: 4e24d421     	fadd.4s	v1, v1, v4
 1f82d88: 9e220124     	scvtf	s4, x9
 1f82d8c: 4f829062     	fmul.4s	v2, v3, v2[0]
 1f82d90: 4f849063     	fmul.4s	v3, v3, v4[0]
 1f82d94: 4e22d400     	fadd.4s	v0, v0, v2
 1f82d98: 4e23cc41     	fmla.4s	v1, v2, v3
 1f82d9c: aa0e03e9     	mov	x9, x14
 1f82da0: f100054a     	subs	x10, x10, #0x1
 1f82da4: 54000120     	b.eq	0x1f82dc8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x370>
 1f82da8: f840856f     	ldr	x15, [x11], #0x8
 1f82dac: ab0f012e     	adds	x14, x9, x15
 1f82db0: 54fffde1     	b.ne	0x1f82d6c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x314>
 1f82db4: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82db8: 17fffff0     	b	0x1f82d78 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x320>
 1f82dbc: 6f00e400     	movi.2d	v0, #0000000000000000
 1f82dc0: f100054b     	subs	x11, x10, #0x1
 1f82dc4: 54fffb81     	b.ne	0x1f82d34 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x2dc>
 1f82dc8: d37ef509     	lsl	x9, x8, #2
 1f82dcc: 6f00e404     	movi.2d	v4, #0000000000000000
 1f82dd0: eb090029     	subs	x9, x1, x9
 1f82dd4: 5400022d     	b.le	0x1f82e18 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x3c0>
 1f82dd8: 8b08100a     	add	x10, x0, x8, lsl #4
 1f82ddc: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82de0: 5280002b     	mov	w11, #0x1               ; =1
 1f82de4: aa0903ec     	mov	x12, x9
 1f82de8: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82dec: bc404545     	ldr	s5, [x10], #0x4
 1f82df0: 1e2238a6     	fsub	s6, s5, s2
 1f82df4: 9e230167     	ucvtf	s7, x11
 1f82df8: 1e2718c7     	fdiv	s7, s6, s7
 1f82dfc: 1e272842     	fadd	s2, s2, s7
 1f82e00: 1e2238a5     	fsub	s5, s5, s2
 1f82e04: 1f050cc3     	fmadd	s3, s6, s5, s3
 1f82e08: 9100056b     	add	x11, x11, #0x1
 1f82e0c: f100058c     	subs	x12, x12, #0x1
 1f82e10: 54fffee1     	b.ne	0x1f82dec <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x394>
 1f82e14: 14000004     	b	0x1f82e24 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x3cc>
 1f82e18: d2800009     	mov	x9, #0x0                ; =0
 1f82e1c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f82e20: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82e24: 9e220106     	scvtf	s6, x8
 1f82e28: ab08012a     	adds	x10, x9, x8
 1f82e2c: 9e220145     	scvtf	s5, x10
 1f82e30: 1e2518c7     	fdiv	s7, s6, s5
 1f82e34: 1e270c84     	fcsel	s4, s4, s7, eq
 1f82e38: ab08014a     	adds	x10, x10, x8
 1f82e3c: 6f00e407     	movi.2d	v7, #0000000000000000
 1f82e40: 9e220150     	scvtf	s16, x10
 1f82e44: 1e3018d1     	fdiv	s17, s6, s16
 1f82e48: 1e310cf1     	fcsel	s17, s7, s17, eq
 1f82e4c: ab08014a     	adds	x10, x10, x8
 1f82e50: 9e220152     	scvtf	s18, x10
 1f82e54: 1e3218d3     	fdiv	s19, s6, s18
 1f82e58: 1e330cf3     	fcsel	s19, s7, s19, eq
 1f82e5c: ab080148     	adds	x8, x10, x8
 1f82e60: 9e220114     	scvtf	s20, x8
 1f82e64: 1e3418c6     	fdiv	s6, s6, s20
 1f82e68: 1e260ce6     	fcsel	s6, s7, s6, eq
 1f82e6c: 1e223807     	fsub	s7, s0, s2
 1f82e70: 1e2708f4     	fmul	s20, s7, s7
 1f82e74: 1e340894     	fmul	s20, s4, s20
 1f82e78: 9e220135     	scvtf	s21, x9
 1f82e7c: 1f150694     	fmadd	s20, s20, s21, s1
 1f82e80: 1e342863     	fadd	s3, s3, s20
 1f82e84: 5e0c0414     	mov	s20, v0[1]
 1f82e88: 1f070882     	fmadd	s2, s4, s7, s2
 1f82e8c: 1e223a84     	fsub	s4, s20, s2
 1f82e90: 1e240887     	fmul	s7, s4, s4
 1f82e94: 1e270a27     	fmul	s7, s17, s7
 1f82e98: 5e0c0434     	mov	s20, v1[1]
 1f82e9c: 1f0550e5     	fmadd	s5, s7, s5, s20
 1f82ea0: 1e252863     	fadd	s3, s3, s5
 1f82ea4: 5e140405     	mov	s5, v0[2]
 1f82ea8: 1f040a22     	fmadd	s2, s17, s4, s2
 1f82eac: 1e2238a4     	fsub	s4, s5, s2
 1f82eb0: 1e240885     	fmul	s5, s4, s4
 1f82eb4: 1e250a65     	fmul	s5, s19, s5
 1f82eb8: 5e140427     	mov	s7, v1[2]
 1f82ebc: 1f101ca5     	fmadd	s5, s5, s16, s7
 1f82ec0: 1e252863     	fadd	s3, s3, s5
 1f82ec4: 1f040a62     	fmadd	s2, s19, s4, s2
 1f82ec8: 5e1c0400     	mov	s0, v0[3]
 1f82ecc: 1e223804     	fsub	s4, s0, s2
 1f82ed0: 1f0408c0     	fmadd	s0, s6, s4, s2
 1f82ed4: 5e1c0421     	mov	s1, v1[3]
 1f82ed8: 1e240882     	fmul	s2, s4, s4
 1f82edc: 1e2208c2     	fmul	s2, s6, s2
 1f82ee0: 1f120441     	fmadd	s1, s2, s18, s1
 1f82ee4: 1e212861     	fadd	s1, s3, s1
 1f82ee8: cb020028     	sub	x8, x1, x2
 1f82eec: 9e220102     	scvtf	s2, x8
 1f82ef0: 1e221821     	fdiv	s1, s1, s2
 1f82ef4: 910b43ff     	add	sp, sp, #0x2d0
 1f82ef8: a9437bfd     	ldp	x29, x30, [sp, #0x30]
 1f82efc: a9424ff4     	ldp	x20, x19, [sp, #0x20]
 1f82f00: a94157f6     	ldp	x22, x21, [sp, #0x10]
 1f82f04: a8c45ff8     	ldp	x24, x23, [sp], #0x40
 1f82f08: d65f03c0     	ret
 1f82f0c: 927ef56c     	and	x12, x11, #0xfffffffffffffffc
 1f82f10: 910543eb     	add	x11, sp, #0x150
 1f82f14: 9100816d     	add	x13, x11, #0x20
 1f82f18: d102c3ab     	sub	x11, x29, #0xb0
 1f82f1c: 9100816e     	add	x14, x11, #0x20
 1f82f20: 910143eb     	add	x11, sp, #0x50
 1f82f24: 9100816f     	add	x15, x11, #0x20
 1f82f28: 5280002b     	mov	w11, #0x1               ; =1
 1f82f2c: 6f00e402     	movi.2d	v2, #0000000000000000
 1f82f30: 14000034     	b	0x1f83000 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x5a8>
 1f82f34: 9e220064     	scvtf	s4, x3
 1f82f38: 9e220205     	scvtf	s5, x16
 1f82f3c: 1e251884     	fdiv	s4, s4, s5
 1f82f40: ad7f99a5     	ldp	q5, q6, [x13, #-0x10]
 1f82f44: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f82f48: ad7fc1e7     	ldp	q7, q16, [x15, #-0x10]
 1f82f4c: 4e27d421     	fadd.4s	v1, v1, v7
 1f82f50: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f82f54: 9e220127     	scvtf	s7, x9
 1f82f58: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f82f5c: 4e23d400     	fadd.4s	v0, v0, v3
 1f82f60: 4e25cc61     	fmla.4s	v1, v3, v5
 1f82f64: 4ea0d4c3     	fsub.4s	v3, v6, v0
 1f82f68: 4e30d421     	fadd.4s	v1, v1, v16
 1f82f6c: 4f849064     	fmul.4s	v4, v3, v4[0]
 1f82f70: 9e220225     	scvtf	s5, x17
 1f82f74: 4f859063     	fmul.4s	v3, v3, v5[0]
 1f82f78: 4e24d400     	fadd.4s	v0, v0, v4
 1f82f7c: a97fc5c9     	ldp	x9, x17, [x14, #-0x8]
 1f82f80: 9e220125     	scvtf	s5, x9
 1f82f84: 4e23cc81     	fmla.4s	v1, v4, v3
 1f82f88: ab090209     	adds	x9, x16, x9
 1f82f8c: 9e220123     	scvtf	s3, x9
 1f82f90: 1e2318a4     	fdiv	s4, s5, s3
 1f82f94: 1e240c44     	fcsel	s4, s2, s4, eq
 1f82f98: ad4099a5     	ldp	q5, q6, [x13, #0x10]
 1f82f9c: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f82fa0: ad40c1e7     	ldp	q7, q16, [x15, #0x10]
 1f82fa4: 4e27d421     	fadd.4s	v1, v1, v7
 1f82fa8: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f82fac: 9e220207     	scvtf	s7, x16
 1f82fb0: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f82fb4: 4e24d400     	fadd.4s	v0, v0, v4
 1f82fb8: 4e25cc81     	fmla.4s	v1, v4, v5
 1f82fbc: ab110129     	adds	x9, x9, x17
 1f82fc0: 9e220224     	scvtf	s4, x17
 1f82fc4: 9e220125     	scvtf	s5, x9
 1f82fc8: 1e251884     	fdiv	s4, s4, s5
 1f82fcc: 1e240c44     	fcsel	s4, s2, s4, eq
 1f82fd0: 4ea0d4c5     	fsub.4s	v5, v6, v0
 1f82fd4: 4e30d421     	fadd.4s	v1, v1, v16
 1f82fd8: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f82fdc: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f82fe0: 4e24d400     	fadd.4s	v0, v0, v4
 1f82fe4: 9100116b     	add	x11, x11, #0x4
 1f82fe8: 4e23cc81     	fmla.4s	v1, v4, v3
 1f82fec: 910101ad     	add	x13, x13, #0x40
 1f82ff0: 910081ce     	add	x14, x14, #0x20
 1f82ff4: 910101ef     	add	x15, x15, #0x40
 1f82ff8: f100118c     	subs	x12, x12, #0x4
 1f82ffc: 54ffea60     	b.eq	0x1f82d48 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x2f0>
 1f83000: f85e81d0     	ldur	x16, [x14, #-0x18]
 1f83004: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83008: ab100131     	adds	x17, x9, x16
 1f8300c: 54000080     	b.eq	0x1f8301c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x5c4>
 1f83010: 9e220203     	scvtf	s3, x16
 1f83014: 9e220224     	scvtf	s4, x17
 1f83018: 1e241863     	fdiv	s3, s3, s4
 1f8301c: f85f01c3     	ldur	x3, [x14, #-0x10]
 1f83020: ab030230     	adds	x16, x17, x3
 1f83024: 54fff881     	b.ne	0x1f82f34 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x4dc>
 1f83028: 6f00e404     	movi.2d	v4, #0000000000000000
 1f8302c: 17ffffc5     	b	0x1f82f40 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 16ll>(float const*, long long, long long)+0x4e8>

0000000001f83030 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)>:
 1f83030: a9bc5ff8     	stp	x24, x23, [sp, #-0x40]!
 1f83034: a90157f6     	stp	x22, x21, [sp, #0x10]
 1f83038: a9024ff4     	stp	x20, x19, [sp, #0x20]
 1f8303c: a9037bfd     	stp	x29, x30, [sp, #0x30]
 1f83040: 9100c3fd     	add	x29, sp, #0x30
 1f83044: d11543ff     	sub	sp, sp, #0x550
 1f83048: 6f00e401     	movi.2d	v1, #0000000000000000
 1f8304c: 3d8153e1     	str	q1, [sp, #0x540]
 1f83050: 3d814fe1     	str	q1, [sp, #0x530]
 1f83054: 91000c28     	add	x8, x1, #0x3
 1f83058: f100003f     	cmp	x1, #0x0
 1f8305c: 9a81b108     	csel	x8, x8, x1, lt
 1f83060: 3d814be1     	str	q1, [sp, #0x520]
 1f83064: 3d8147e1     	str	q1, [sp, #0x510]
 1f83068: 9342fd08     	asr	x8, x8, #2
 1f8306c: b1003d09     	adds	x9, x8, #0xf
 1f83070: 9100790a     	add	x10, x8, #0x1e
 1f83074: 9a89b149     	csel	x9, x10, x9, lt
 1f83078: 3d8143e1     	str	q1, [sp, #0x500]
 1f8307c: 3d813fe1     	str	q1, [sp, #0x4f0]
 1f83080: 9344fd2a     	asr	x10, x9, #4
 1f83084: d1000549     	sub	x9, x10, #0x1
 1f83088: dac01129     	clz	x9, x9
 1f8308c: d2401529     	eor	x9, x9, #0x3f
 1f83090: 3d813be1     	str	q1, [sp, #0x4e0]
 1f83094: 3d8137e1     	str	q1, [sp, #0x4d0]
 1f83098: f102103f     	cmp	x1, #0x84
 1f8309c: 5280002b     	mov	w11, #0x1               ; =1
 1f830a0: 9a89b569     	csinc	x9, x11, x9, lt
 1f830a4: 3d8133e1     	str	q1, [sp, #0x4c0]
 1f830a8: 3d812fe1     	str	q1, [sp, #0x4b0]
 1f830ac: 3d812be1     	str	q1, [sp, #0x4a0]
 1f830b0: 3d8127e1     	str	q1, [sp, #0x490]
 1f830b4: 3d8123e1     	str	q1, [sp, #0x480]
 1f830b8: 3d811fe1     	str	q1, [sp, #0x470]
 1f830bc: 3d811be1     	str	q1, [sp, #0x460]
 1f830c0: 3d8117e1     	str	q1, [sp, #0x450]
 1f830c4: 3d8113e1     	str	q1, [sp, #0x440]
 1f830c8: 3d810fe1     	str	q1, [sp, #0x430]
 1f830cc: 3d810be1     	str	q1, [sp, #0x420]
 1f830d0: 3d8107e1     	str	q1, [sp, #0x410]
 1f830d4: ad1f87e1     	stp	q1, q1, [sp, #0x3f0]
 1f830d8: ad1e87e1     	stp	q1, q1, [sp, #0x3d0]
 1f830dc: ad1d87e1     	stp	q1, q1, [sp, #0x3b0]
 1f830e0: ad1c87e1     	stp	q1, q1, [sp, #0x390]
 1f830e4: ad1b87e1     	stp	q1, q1, [sp, #0x370]
 1f830e8: ad1a87e1     	stp	q1, q1, [sp, #0x350]
 1f830ec: ad1987e1     	stp	q1, q1, [sp, #0x330]
 1f830f0: ad1887e1     	stp	q1, q1, [sp, #0x310]
 1f830f4: ad1787e1     	stp	q1, q1, [sp, #0x2f0]
 1f830f8: ad1687e1     	stp	q1, q1, [sp, #0x2d0]
 1f830fc: ad1587e1     	stp	q1, q1, [sp, #0x2b0]
 1f83100: ad1487e1     	stp	q1, q1, [sp, #0x290]
 1f83104: ad1387e1     	stp	q1, q1, [sp, #0x270]
 1f83108: ad1287e1     	stp	q1, q1, [sp, #0x250]
 1f8310c: ad1187e1     	stp	q1, q1, [sp, #0x230]
 1f83110: ad1087e1     	stp	q1, q1, [sp, #0x210]
 1f83114: ad0f87e1     	stp	q1, q1, [sp, #0x1f0]
 1f83118: ad0e87e1     	stp	q1, q1, [sp, #0x1d0]
 1f8311c: ad0d87e1     	stp	q1, q1, [sp, #0x1b0]
 1f83120: ad0c87e1     	stp	q1, q1, [sp, #0x190]
 1f83124: ad0b87e1     	stp	q1, q1, [sp, #0x170]
 1f83128: ad0a87e1     	stp	q1, q1, [sp, #0x150]
 1f8312c: ad0987e1     	stp	q1, q1, [sp, #0x130]
 1f83130: ad0887e1     	stp	q1, q1, [sp, #0x110]
 1f83134: ad0787e1     	stp	q1, q1, [sp, #0xf0]
 1f83138: ad0687e1     	stp	q1, q1, [sp, #0xd0]
 1f8313c: ad0587e1     	stp	q1, q1, [sp, #0xb0]
 1f83140: ad0487e1     	stp	q1, q1, [sp, #0x90]
 1f83144: ad0387e1     	stp	q1, q1, [sp, #0x70]
 1f83148: ad0287e1     	stp	q1, q1, [sp, #0x50]
 1f8314c: b101ec3f     	cmn	x1, #0x7b
 1f83150: 540010eb     	b.lt	0x1f8336c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x33c>
 1f83154: d100050b     	sub	x11, x8, #0x1
 1f83158: b100817f     	cmn	x11, #0x20
 1f8315c: 54001088     	b.hi	0x1f8336c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x33c>
 1f83160: d2800004     	mov	x4, #0x0                ; =0
 1f83164: 910143eb     	add	x11, sp, #0x50
 1f83168: 9100416b     	add	x11, x11, #0x10
 1f8316c: 910943ec     	add	x12, sp, #0x250
 1f83170: 9100418c     	add	x12, x12, #0x10
 1f83174: 911143ed     	add	x13, sp, #0x450
 1f83178: b27d01ad     	orr	x13, x13, #0x8
 1f8317c: 5280020f     	mov	w15, #0x10              ; =16
 1f83180: f008808e     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f83184: 912701ce     	add	x14, x14, #0x9c0
 1f83188: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8318c: 6f00e406     	movi.2d	v6, #0000000000000000
 1f83190: aa0003f0     	mov	x16, x0
 1f83194: aa0803e3     	mov	x3, x8
 1f83198: f0088091     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f8319c: 91230231     	add	x17, x17, #0x8c0
 1f831a0: 14000004     	b	0x1f831b0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x180>
 1f831a4: 91040210     	add	x16, x16, #0x100
 1f831a8: eb0a009f     	cmp	x4, x10
 1f831ac: 54000de0     	b.eq	0x1f83368 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x338>
 1f831b0: aa0403e5     	mov	x5, x4
 1f831b4: aa0303e4     	mov	x4, x3
 1f831b8: f1004063     	subs	x3, x3, #0x10
 1f831bc: 9a8fb094     	csel	x20, x4, x15, lt
 1f831c0: cb051115     	sub	x21, x8, x5, lsl #4
 1f831c4: f10042bf     	cmp	x21, #0x10
 1f831c8: 9a8fb2b3     	csel	x19, x21, x15, lt
 1f831cc: 38bfc1c4     	ldaprb	w4, [x14]
 1f831d0: 36000204     	tbz	w4, #0x0, 0x1f83210 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x1e0>
 1f831d4: f10006bf     	cmp	x21, #0x1
 1f831d8: 5400050b     	b.lt	0x1f83278 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x248>
 1f831dc: 6f00e401     	movi.2d	v1, #0000000000000000
 1f831e0: aa1003e4     	mov	x4, x16
 1f831e4: aa1103e6     	mov	x6, x17
 1f831e8: 6f00e402     	movi.2d	v2, #0000000000000000
 1f831ec: 3cc10483     	ldr	q3, [x4], #0x10
 1f831f0: 3cc104c4     	ldr	q4, [x6], #0x10
 1f831f4: 4ea1d465     	fsub.4s	v5, v3, v1
 1f831f8: 4e24cca1     	fmla.4s	v1, v5, v4
 1f831fc: 4ea1d463     	fsub.4s	v3, v3, v1
 1f83200: 4e25cc62     	fmla.4s	v2, v3, v5
 1f83204: f1000694     	subs	x20, x20, #0x1
 1f83208: 54ffff21     	b.ne	0x1f831ec <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x1bc>
 1f8320c: 1400001d     	b	0x1f83280 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x250>
 1f83210: a9010be0     	stp	x0, x2, [sp, #0x10]
 1f83214: aa0103f6     	mov	x22, x1
 1f83218: aa0803f7     	mov	x23, x8
 1f8321c: a902a7eb     	stp	x11, x9, [sp, #0x28]
 1f83220: a9002bec     	stp	x12, x10, [sp]
 1f83224: f90013ed     	str	x13, [sp, #0x20]
 1f83228: a90443e3     	stp	x3, x16, [sp, #0x40]
 1f8322c: f9001fe5     	str	x5, [sp, #0x38]
 1f83230: 94c0e5a8     	bl	0x4fbc8d0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long) (.cold.1)>
 1f83234: 6f00e406     	movi.2d	v6, #0000000000000000
 1f83238: a9438fe5     	ldp	x5, x3, [sp, #0x38]
 1f8323c: f0088091     	adrp	x17, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f83240: 91230231     	add	x17, x17, #0x8c0
 1f83244: f94027f0     	ldr	x16, [sp, #0x48]
 1f83248: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8324c: f008808e     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f83250: 912701ce     	add	x14, x14, #0x9c0
 1f83254: 5280020f     	mov	w15, #0x10              ; =16
 1f83258: a9422fed     	ldp	x13, x11, [sp, #0x20]
 1f8325c: a9402bec     	ldp	x12, x10, [sp]
 1f83260: f9401be9     	ldr	x9, [sp, #0x30]
 1f83264: aa1703e8     	mov	x8, x23
 1f83268: a9410be0     	ldp	x0, x2, [sp, #0x10]
 1f8326c: aa1603e1     	mov	x1, x22
 1f83270: f10006bf     	cmp	x21, #0x1
 1f83274: 54fffb4a     	b.ge	0x1f831dc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x1ac>
 1f83278: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8327c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83280: f9422be4     	ldr	x4, [sp, #0x450]
 1f83284: ab130095     	adds	x21, x4, x19
 1f83288: 9e220263     	scvtf	s3, x19
 1f8328c: 9e2202a4     	scvtf	s4, x21
 1f83290: 1e241863     	fdiv	s3, s3, s4
 1f83294: 1e230c03     	fcsel	s3, s0, s3, eq
 1f83298: 3dc097e4     	ldr	q4, [sp, #0x250]
 1f8329c: 4ea4d425     	fsub.4s	v5, v1, v4
 1f832a0: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f832a4: 4e21d441     	fadd.4s	v1, v2, v1
 1f832a8: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f832ac: 9e220082     	scvtf	s2, x4
 1f832b0: 4f8290a5     	fmul.4s	v5, v5, v2[0]
 1f832b4: 4e23d482     	fadd.4s	v2, v4, v3
 1f832b8: 3d8097e2     	str	q2, [sp, #0x250]
 1f832bc: 4e25cc61     	fmla.4s	v1, v3, v5
 1f832c0: 3d8017e1     	str	q1, [sp, #0x50]
 1f832c4: f9022bf5     	str	x21, [sp, #0x450]
 1f832c8: 910004a4     	add	x4, x5, #0x1
 1f832cc: f100093f     	cmp	x9, #0x2
 1f832d0: 54fff6a3     	b.lo	0x1f831a4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x174>
 1f832d4: 3607f685     	tbz	w5, #0x0, 0x1f831a4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x174>
 1f832d8: aa0d03e5     	mov	x5, x13
 1f832dc: aa0c03e6     	mov	x6, x12
 1f832e0: aa0b03e7     	mov	x7, x11
 1f832e4: 52800053     	mov	w19, #0x2               ; =2
 1f832e8: aa0403f7     	mov	x23, x4
 1f832ec: aa1703f4     	mov	x20, x23
 1f832f0: f94000b7     	ldr	x23, [x5]
 1f832f4: ab1502f6     	adds	x22, x23, x21
 1f832f8: 540000a0     	b.eq	0x1f8330c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x2dc>
 1f832fc: 9e2202a3     	scvtf	s3, x21
 1f83300: 9e2202c4     	scvtf	s4, x22
 1f83304: 1e241863     	fdiv	s3, s3, s4
 1f83308: 14000002     	b	0x1f83310 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x2e0>
 1f8330c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83310: 3dc000c4     	ldr	q4, [x6]
 1f83314: 4ea4d442     	fsub.4s	v2, v2, v4
 1f83318: 3dc000e5     	ldr	q5, [x7]
 1f8331c: 4e21d4a1     	fadd.4s	v1, v5, v1
 1f83320: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f83324: 9e2202e5     	scvtf	s5, x23
 1f83328: 4f859045     	fmul.4s	v5, v2, v5[0]
 1f8332c: 4e23d482     	fadd.4s	v2, v4, v3
 1f83330: 4e25cc61     	fmla.4s	v1, v3, v5
 1f83334: a93fd8bf     	stp	xzr, x22, [x5, #-0x8]
 1f83338: ad3f88c6     	stp	q6, q2, [x6, #-0x10]
 1f8333c: ad3f84e6     	stp	q6, q1, [x7, #-0x10]
 1f83340: eb09027f     	cmp	x19, x9
 1f83344: 54fff302     	b.hs	0x1f831a4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x174>
 1f83348: d341fe97     	lsr	x23, x20, #1
 1f8334c: 91000673     	add	x19, x19, #0x1
 1f83350: 910040e7     	add	x7, x7, #0x10
 1f83354: 910040c6     	add	x6, x6, #0x10
 1f83358: 910020a5     	add	x5, x5, #0x8
 1f8335c: aa1603f5     	mov	x21, x22
 1f83360: 360ffc74     	tbz	w20, #0x1, 0x1f832ec <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x2bc>
 1f83364: 17ffff90     	b	0x1f831a4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x174>
 1f83368: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f8336c: 3dc097e0     	ldr	q0, [sp, #0x250]
 1f83370: f100052b     	subs	x11, x9, #0x1
 1f83374: 54000480     	b.eq	0x1f83404 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x3d4>
 1f83378: f9422bea     	ldr	x10, [sp, #0x450]
 1f8337c: d100092c     	sub	x12, x9, #0x2
 1f83380: 92400569     	and	x9, x11, #0x3
 1f83384: f1000d9f     	cmp	x12, #0x3
 1f83388: 54000e02     	b.hs	0x1f83548 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x518>
 1f8338c: 5280002b     	mov	w11, #0x1               ; =1
 1f83390: b40003a9     	cbz	x9, 0x1f83404 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x3d4>
 1f83394: d37ced6d     	lsl	x13, x11, #4
 1f83398: 910143ec     	add	x12, sp, #0x50
 1f8339c: 8b0d018c     	add	x12, x12, x13
 1f833a0: 910943ee     	add	x14, sp, #0x250
 1f833a4: 8b0d01cd     	add	x13, x14, x13
 1f833a8: 911143ee     	add	x14, sp, #0x450
 1f833ac: 8b0b0dcb     	add	x11, x14, x11, lsl #3
 1f833b0: 14000010     	b	0x1f833f0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x3c0>
 1f833b4: 9e2201e2     	scvtf	s2, x15
 1f833b8: 9e2201c3     	scvtf	s3, x14
 1f833bc: 1e231842     	fdiv	s2, s2, s3
 1f833c0: 3cc105a3     	ldr	q3, [x13], #0x10
 1f833c4: 4ea0d463     	fsub.4s	v3, v3, v0
 1f833c8: 3cc10584     	ldr	q4, [x12], #0x10
 1f833cc: 4e24d421     	fadd.4s	v1, v1, v4
 1f833d0: 9e220144     	scvtf	s4, x10
 1f833d4: 4f829062     	fmul.4s	v2, v3, v2[0]
 1f833d8: 4f849063     	fmul.4s	v3, v3, v4[0]
 1f833dc: 4e22d400     	fadd.4s	v0, v0, v2
 1f833e0: 4e23cc41     	fmla.4s	v1, v2, v3
 1f833e4: aa0e03ea     	mov	x10, x14
 1f833e8: f1000529     	subs	x9, x9, #0x1
 1f833ec: 540000c0     	b.eq	0x1f83404 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x3d4>
 1f833f0: f840856f     	ldr	x15, [x11], #0x8
 1f833f4: ab0f014e     	adds	x14, x10, x15
 1f833f8: 54fffde1     	b.ne	0x1f833b4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x384>
 1f833fc: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83400: 17fffff0     	b	0x1f833c0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x390>
 1f83404: d37ef509     	lsl	x9, x8, #2
 1f83408: 6f00e404     	movi.2d	v4, #0000000000000000
 1f8340c: eb090029     	subs	x9, x1, x9
 1f83410: 5400022d     	b.le	0x1f83454 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x424>
 1f83414: 8b08100a     	add	x10, x0, x8, lsl #4
 1f83418: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8341c: 5280002b     	mov	w11, #0x1               ; =1
 1f83420: aa0903ec     	mov	x12, x9
 1f83424: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83428: bc404545     	ldr	s5, [x10], #0x4
 1f8342c: 1e2238a6     	fsub	s6, s5, s2
 1f83430: 9e230167     	ucvtf	s7, x11
 1f83434: 1e2718c7     	fdiv	s7, s6, s7
 1f83438: 1e272842     	fadd	s2, s2, s7
 1f8343c: 1e2238a5     	fsub	s5, s5, s2
 1f83440: 1f050cc3     	fmadd	s3, s6, s5, s3
 1f83444: 9100056b     	add	x11, x11, #0x1
 1f83448: f100058c     	subs	x12, x12, #0x1
 1f8344c: 54fffee1     	b.ne	0x1f83428 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x3f8>
 1f83450: 14000004     	b	0x1f83460 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x430>
 1f83454: d2800009     	mov	x9, #0x0                ; =0
 1f83458: 6f00e403     	movi.2d	v3, #0000000000000000
 1f8345c: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83460: 9e220106     	scvtf	s6, x8
 1f83464: ab08012a     	adds	x10, x9, x8
 1f83468: 9e220145     	scvtf	s5, x10
 1f8346c: 1e2518c7     	fdiv	s7, s6, s5
 1f83470: 1e270c84     	fcsel	s4, s4, s7, eq
 1f83474: ab08014a     	adds	x10, x10, x8
 1f83478: 6f00e407     	movi.2d	v7, #0000000000000000
 1f8347c: 9e220150     	scvtf	s16, x10
 1f83480: 1e3018d1     	fdiv	s17, s6, s16
 1f83484: 1e310cf1     	fcsel	s17, s7, s17, eq
 1f83488: ab08014a     	adds	x10, x10, x8
 1f8348c: 9e220152     	scvtf	s18, x10
 1f83490: 1e3218d3     	fdiv	s19, s6, s18
 1f83494: 1e330cf3     	fcsel	s19, s7, s19, eq
 1f83498: ab080148     	adds	x8, x10, x8
 1f8349c: 9e220114     	scvtf	s20, x8
 1f834a0: 1e3418c6     	fdiv	s6, s6, s20
 1f834a4: 1e260ce6     	fcsel	s6, s7, s6, eq
 1f834a8: 1e223807     	fsub	s7, s0, s2
 1f834ac: 1e2708f4     	fmul	s20, s7, s7
 1f834b0: 1e340894     	fmul	s20, s4, s20
 1f834b4: 9e220135     	scvtf	s21, x9
 1f834b8: 1f150694     	fmadd	s20, s20, s21, s1
 1f834bc: 1e342863     	fadd	s3, s3, s20
 1f834c0: 5e0c0414     	mov	s20, v0[1]
 1f834c4: 1f070882     	fmadd	s2, s4, s7, s2
 1f834c8: 1e223a84     	fsub	s4, s20, s2
 1f834cc: 1e240887     	fmul	s7, s4, s4
 1f834d0: 1e270a27     	fmul	s7, s17, s7
 1f834d4: 5e0c0434     	mov	s20, v1[1]
 1f834d8: 1f0550e5     	fmadd	s5, s7, s5, s20
 1f834dc: 1e252863     	fadd	s3, s3, s5
 1f834e0: 5e140405     	mov	s5, v0[2]
 1f834e4: 1f040a22     	fmadd	s2, s17, s4, s2
 1f834e8: 1e2238a4     	fsub	s4, s5, s2
 1f834ec: 1e240885     	fmul	s5, s4, s4
 1f834f0: 1e250a65     	fmul	s5, s19, s5
 1f834f4: 5e140427     	mov	s7, v1[2]
 1f834f8: 1f101ca5     	fmadd	s5, s5, s16, s7
 1f834fc: 1e252863     	fadd	s3, s3, s5
 1f83500: 1f040a62     	fmadd	s2, s19, s4, s2
 1f83504: 5e1c0400     	mov	s0, v0[3]
 1f83508: 1e223804     	fsub	s4, s0, s2
 1f8350c: 1f0408c0     	fmadd	s0, s6, s4, s2
 1f83510: 5e1c0421     	mov	s1, v1[3]
 1f83514: 1e240882     	fmul	s2, s4, s4
 1f83518: 1e2208c2     	fmul	s2, s6, s2
 1f8351c: 1f120441     	fmadd	s1, s2, s18, s1
 1f83520: 1e212861     	fadd	s1, s3, s1
 1f83524: cb020028     	sub	x8, x1, x2
 1f83528: 9e220102     	scvtf	s2, x8
 1f8352c: 1e221821     	fdiv	s1, s1, s2
 1f83530: 911543ff     	add	sp, sp, #0x550
 1f83534: a9437bfd     	ldp	x29, x30, [sp, #0x30]
 1f83538: a9424ff4     	ldp	x20, x19, [sp, #0x20]
 1f8353c: a94157f6     	ldp	x22, x21, [sp, #0x10]
 1f83540: a8c45ff8     	ldp	x24, x23, [sp], #0x40
 1f83544: d65f03c0     	ret
 1f83548: 927ef56c     	and	x12, x11, #0xfffffffffffffffc
 1f8354c: 910943eb     	add	x11, sp, #0x250
 1f83550: 9100816d     	add	x13, x11, #0x20
 1f83554: 911143eb     	add	x11, sp, #0x450
 1f83558: 9100816e     	add	x14, x11, #0x20
 1f8355c: 910143eb     	add	x11, sp, #0x50
 1f83560: 9100816f     	add	x15, x11, #0x20
 1f83564: 5280002b     	mov	w11, #0x1               ; =1
 1f83568: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8356c: 14000034     	b	0x1f8363c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x60c>
 1f83570: 9e220064     	scvtf	s4, x3
 1f83574: 9e220205     	scvtf	s5, x16
 1f83578: 1e251884     	fdiv	s4, s4, s5
 1f8357c: ad7f99a5     	ldp	q5, q6, [x13, #-0x10]
 1f83580: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f83584: ad7fc1e7     	ldp	q7, q16, [x15, #-0x10]
 1f83588: 4e27d421     	fadd.4s	v1, v1, v7
 1f8358c: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f83590: 9e220147     	scvtf	s7, x10
 1f83594: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f83598: 4e23d400     	fadd.4s	v0, v0, v3
 1f8359c: 4e25cc61     	fmla.4s	v1, v3, v5
 1f835a0: 4ea0d4c3     	fsub.4s	v3, v6, v0
 1f835a4: 4e30d421     	fadd.4s	v1, v1, v16
 1f835a8: 4f849064     	fmul.4s	v4, v3, v4[0]
 1f835ac: 9e220225     	scvtf	s5, x17
 1f835b0: 4f859063     	fmul.4s	v3, v3, v5[0]
 1f835b4: 4e24d400     	fadd.4s	v0, v0, v4
 1f835b8: a97fc5ca     	ldp	x10, x17, [x14, #-0x8]
 1f835bc: 9e220145     	scvtf	s5, x10
 1f835c0: 4e23cc81     	fmla.4s	v1, v4, v3
 1f835c4: ab0a020a     	adds	x10, x16, x10
 1f835c8: 9e220143     	scvtf	s3, x10
 1f835cc: 1e2318a4     	fdiv	s4, s5, s3
 1f835d0: 1e240c44     	fcsel	s4, s2, s4, eq
 1f835d4: ad4099a5     	ldp	q5, q6, [x13, #0x10]
 1f835d8: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f835dc: ad40c1e7     	ldp	q7, q16, [x15, #0x10]
 1f835e0: 4e27d421     	fadd.4s	v1, v1, v7
 1f835e4: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f835e8: 9e220207     	scvtf	s7, x16
 1f835ec: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f835f0: 4e24d400     	fadd.4s	v0, v0, v4
 1f835f4: 4e25cc81     	fmla.4s	v1, v4, v5
 1f835f8: ab11014a     	adds	x10, x10, x17
 1f835fc: 9e220224     	scvtf	s4, x17
 1f83600: 9e220145     	scvtf	s5, x10
 1f83604: 1e251884     	fdiv	s4, s4, s5
 1f83608: 1e240c44     	fcsel	s4, s2, s4, eq
 1f8360c: 4ea0d4c5     	fsub.4s	v5, v6, v0
 1f83610: 4e30d421     	fadd.4s	v1, v1, v16
 1f83614: 4f8490a4     	fmul.4s	v4, v5, v4[0]
 1f83618: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f8361c: 4e24d400     	fadd.4s	v0, v0, v4
 1f83620: 9100116b     	add	x11, x11, #0x4
 1f83624: 4e23cc81     	fmla.4s	v1, v4, v3
 1f83628: 910101ad     	add	x13, x13, #0x40
 1f8362c: 910081ce     	add	x14, x14, #0x20
 1f83630: 910101ef     	add	x15, x15, #0x40
 1f83634: f100118c     	subs	x12, x12, #0x4
 1f83638: 54ffeac0     	b.eq	0x1f83390 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x360>
 1f8363c: f85e81d0     	ldur	x16, [x14, #-0x18]
 1f83640: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83644: ab100151     	adds	x17, x10, x16
 1f83648: 54000080     	b.eq	0x1f83658 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x628>
 1f8364c: 9e220203     	scvtf	s3, x16
 1f83650: 9e220224     	scvtf	s4, x17
 1f83654: 1e241863     	fdiv	s3, s3, s4
 1f83658: f85f01c3     	ldur	x3, [x14, #-0x10]
 1f8365c: ab030230     	adds	x16, x17, x3
 1f83660: 54fff881     	b.ne	0x1f83570 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x540>
 1f83664: 6f00e404     	movi.2d	v4, #0000000000000000
 1f83668: 17ffffc5     	b	0x1f8357c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 32ll>(float const*, long long, long long)+0x54c>

0000000001f8366c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)>:
 1f8366c: a9bb6ffc     	stp	x28, x27, [sp, #-0x50]!
 1f83670: a9015ff8     	stp	x24, x23, [sp, #0x10]
 1f83674: a90257f6     	stp	x22, x21, [sp, #0x20]
 1f83678: a9034ff4     	stp	x20, x19, [sp, #0x30]
 1f8367c: a9047bfd     	stp	x29, x30, [sp, #0x40]
 1f83680: 910103fd     	add	x29, sp, #0x40
 1f83684: d12943ff     	sub	sp, sp, #0xa50
 1f83688: aa0203f3     	mov	x19, x2
 1f8368c: aa0103f4     	mov	x20, x1
 1f83690: aa0003f5     	mov	x21, x0
 1f83694: 6f00e400     	movi.2d	v0, #0000000000000000
 1f83698: 3d8293e0     	str	q0, [sp, #0xa40]
 1f8369c: 3d828fe0     	str	q0, [sp, #0xa30]
 1f836a0: 3d828be0     	str	q0, [sp, #0xa20]
 1f836a4: 3d8287e0     	str	q0, [sp, #0xa10]
 1f836a8: 3d8283e0     	str	q0, [sp, #0xa00]
 1f836ac: 3d827fe0     	str	q0, [sp, #0x9f0]
 1f836b0: 3d827be0     	str	q0, [sp, #0x9e0]
 1f836b4: 3d8277e0     	str	q0, [sp, #0x9d0]
 1f836b8: 3d8273e0     	str	q0, [sp, #0x9c0]
 1f836bc: 3d826fe0     	str	q0, [sp, #0x9b0]
 1f836c0: 3d826be0     	str	q0, [sp, #0x9a0]
 1f836c4: 3d8267e0     	str	q0, [sp, #0x990]
 1f836c8: 3d8263e0     	str	q0, [sp, #0x980]
 1f836cc: 3d825fe0     	str	q0, [sp, #0x970]
 1f836d0: 3d825be0     	str	q0, [sp, #0x960]
 1f836d4: 3d8257e0     	str	q0, [sp, #0x950]
 1f836d8: 3d8253e0     	str	q0, [sp, #0x940]
 1f836dc: 3d824fe0     	str	q0, [sp, #0x930]
 1f836e0: 3d824be0     	str	q0, [sp, #0x920]
 1f836e4: 3d8247e0     	str	q0, [sp, #0x910]
 1f836e8: 3d8243e0     	str	q0, [sp, #0x900]
 1f836ec: 3d823fe0     	str	q0, [sp, #0x8f0]
 1f836f0: 3d823be0     	str	q0, [sp, #0x8e0]
 1f836f4: 3d8237e0     	str	q0, [sp, #0x8d0]
 1f836f8: 3d8233e0     	str	q0, [sp, #0x8c0]
 1f836fc: 3d822fe0     	str	q0, [sp, #0x8b0]
 1f83700: 3d822be0     	str	q0, [sp, #0x8a0]
 1f83704: 3d8227e0     	str	q0, [sp, #0x890]
 1f83708: 3d8223e0     	str	q0, [sp, #0x880]
 1f8370c: 3d821fe0     	str	q0, [sp, #0x870]
 1f83710: 3d821be0     	str	q0, [sp, #0x860]
 1f83714: 3d8217e0     	str	q0, [sp, #0x850]
 1f83718: 911143e0     	add	x0, sp, #0x450
 1f8371c: 52808001     	mov	w1, #0x400              ; =1024
 1f83720: 94c175c8     	bl	0x4fe0e40 <_zunmqr_+0x4fe0e40>
 1f83724: 91000e88     	add	x8, x20, #0x3
 1f83728: f100029f     	cmp	x20, #0x0
 1f8372c: 9a94b108     	csel	x8, x8, x20, lt
 1f83730: 9342fd16     	asr	x22, x8, #2
 1f83734: b1003ec8     	adds	x8, x22, #0xf
 1f83738: 91007ac9     	add	x9, x22, #0x1e
 1f8373c: 9a88b128     	csel	x8, x9, x8, lt
 1f83740: 9344fd18     	asr	x24, x8, #4
 1f83744: d1000708     	sub	x8, x24, #0x1
 1f83748: dac01108     	clz	x8, x8
 1f8374c: d2401508     	eor	x8, x8, #0x3f
 1f83750: f102129f     	cmp	x20, #0x84
 1f83754: 52800029     	mov	w9, #0x1                ; =1
 1f83758: 9a88b537     	csinc	x23, x9, x8, lt
 1f8375c: 910143e0     	add	x0, sp, #0x50
 1f83760: 52808001     	mov	w1, #0x400              ; =1024
 1f83764: 94c175b7     	bl	0x4fe0e40 <_zunmqr_+0x4fe0e40>
 1f83768: d2800009     	mov	x9, #0x0                ; =0
 1f8376c: b101ee9f     	cmn	x20, #0x7b
 1f83770: 5400152b     	b.lt	0x1f83a14 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x3a8>
 1f83774: d10006c8     	sub	x8, x22, #0x1
 1f83778: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8377c: b100811f     	cmn	x8, #0x20
 1f83780: 6f00e415     	movi.2d	v21, #0000000000000000
 1f83784: 54001008     	b.hi	0x1f83984 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x318>
 1f83788: d2800010     	mov	x16, #0x0               ; =0
 1f8378c: 910143e8     	add	x8, sp, #0x50
 1f83790: 91004108     	add	x8, x8, #0x10
 1f83794: 911143e9     	add	x9, sp, #0x450
 1f83798: 91004129     	add	x9, x9, #0x10
 1f8379c: 912143ea     	add	x10, sp, #0x850
 1f837a0: b27d014a     	orr	x10, x10, #0x8
 1f837a4: 5280020b     	mov	w11, #0x10              ; =16
 1f837a8: f008808c     	adrp	x12, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f837ac: 912b418c     	add	x12, x12, #0xad0
 1f837b0: 6f00e400     	movi.2d	v0, #0000000000000000
 1f837b4: 6f00e406     	movi.2d	v6, #0000000000000000
 1f837b8: aa1503ed     	mov	x13, x21
 1f837bc: aa1603ef     	mov	x15, x22
 1f837c0: f008808e     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f837c4: 912741ce     	add	x14, x14, #0x9d0
 1f837c8: 14000004     	b	0x1f837d8 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x16c>
 1f837cc: 910401ad     	add	x13, x13, #0x100
 1f837d0: eb18021f     	cmp	x16, x24
 1f837d4: 54000d20     	b.eq	0x1f83978 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x30c>
 1f837d8: aa1003f1     	mov	x17, x16
 1f837dc: aa0f03f0     	mov	x16, x15
 1f837e0: f10041ef     	subs	x15, x15, #0x10
 1f837e4: 9a8bb200     	csel	x0, x16, x11, lt
 1f837e8: cb1112c1     	sub	x1, x22, x17, lsl #4
 1f837ec: f100403f     	cmp	x1, #0x10
 1f837f0: 9a8bb030     	csel	x16, x1, x11, lt
 1f837f4: 38bfc182     	ldaprb	w2, [x12]
 1f837f8: 36000202     	tbz	w2, #0x0, 0x1f83838 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x1cc>
 1f837fc: f100043f     	cmp	x1, #0x1
 1f83800: 5400044b     	b.lt	0x1f83888 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x21c>
 1f83804: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83808: aa0d03e1     	mov	x1, x13
 1f8380c: aa0e03e2     	mov	x2, x14
 1f83810: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83814: 3cc10423     	ldr	q3, [x1], #0x10
 1f83818: 3cc10444     	ldr	q4, [x2], #0x10
 1f8381c: 4ea1d465     	fsub.4s	v5, v3, v1
 1f83820: 4e24cca1     	fmla.4s	v1, v5, v4
 1f83824: 4ea1d463     	fsub.4s	v3, v3, v1
 1f83828: 4e25cc62     	fmla.4s	v2, v3, v5
 1f8382c: f1000400     	subs	x0, x0, #0x1
 1f83830: 54ffff21     	b.ne	0x1f83814 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x1a8>
 1f83834: 14000017     	b	0x1f83890 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x224>
 1f83838: a90123ea     	stp	x10, x8, [sp, #0x10]
 1f8383c: f90007e9     	str	x9, [sp, #0x8]
 1f83840: a90437ef     	stp	x15, x13, [sp, #0x40]
 1f83844: a90247e0     	stp	x0, x17, [sp, #0x20]
 1f83848: a90343e1     	stp	x1, x16, [sp, #0x30]
 1f8384c: 94c0e431     	bl	0x4fbc910 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long) (.cold.1)>
 1f83850: 6f00e406     	movi.2d	v6, #0000000000000000
 1f83854: a94343e1     	ldp	x1, x16, [sp, #0x30]
 1f83858: a94247e0     	ldp	x0, x17, [sp, #0x20]
 1f8385c: a94437ef     	ldp	x15, x13, [sp, #0x40]
 1f83860: f008808e     	adrp	x14, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f83864: 912741ce     	add	x14, x14, #0x9d0
 1f83868: 6f00e400     	movi.2d	v0, #0000000000000000
 1f8386c: f008808c     	adrp	x12, 0x12f96000 <at::native::DispatchStub<void (*)(at::Tensor&, at::Tensor const&, double, long long, at::Tensor const&, double, long long, double, long long), at::native::qadd_tensor_cpu_stub_DECLARE_DISPATCH_type>::DEFAULT>
 1f83870: 912b418c     	add	x12, x12, #0xad0
 1f83874: 5280020b     	mov	w11, #0x10              ; =16
 1f83878: a940abe9     	ldp	x9, x10, [sp, #0x8]
 1f8387c: f9400fe8     	ldr	x8, [sp, #0x18]
 1f83880: f100043f     	cmp	x1, #0x1
 1f83884: 54fffc0a     	b.ge	0x1f83804 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x198>
 1f83888: 6f00e402     	movi.2d	v2, #0000000000000000
 1f8388c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83890: f9442be0     	ldr	x0, [sp, #0x850]
 1f83894: ab100004     	adds	x4, x0, x16
 1f83898: 9e220203     	scvtf	s3, x16
 1f8389c: 9e220084     	scvtf	s4, x4
 1f838a0: 1e241863     	fdiv	s3, s3, s4
 1f838a4: 1e230c03     	fcsel	s3, s0, s3, eq
 1f838a8: 3dc117e4     	ldr	q4, [sp, #0x450]
 1f838ac: 4ea4d425     	fsub.4s	v5, v1, v4
 1f838b0: 3dc017e1     	ldr	q1, [sp, #0x50]
 1f838b4: 4e21d441     	fadd.4s	v1, v2, v1
 1f838b8: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f838bc: 9e220002     	scvtf	s2, x0
 1f838c0: 4f8290a5     	fmul.4s	v5, v5, v2[0]
 1f838c4: 4e23d482     	fadd.4s	v2, v4, v3
 1f838c8: 3d8117e2     	str	q2, [sp, #0x450]
 1f838cc: 4e25cc61     	fmla.4s	v1, v3, v5
 1f838d0: 3d8017e1     	str	q1, [sp, #0x50]
 1f838d4: f9042be4     	str	x4, [sp, #0x850]
 1f838d8: 91000630     	add	x16, x17, #0x1
 1f838dc: f1000aff     	cmp	x23, #0x2
 1f838e0: 54fff763     	b.lo	0x1f837cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x160>
 1f838e4: 3607f751     	tbz	w17, #0x0, 0x1f837cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x160>
 1f838e8: aa0a03f1     	mov	x17, x10
 1f838ec: aa0903e0     	mov	x0, x9
 1f838f0: aa0803e1     	mov	x1, x8
 1f838f4: 52800042     	mov	w2, #0x2                ; =2
 1f838f8: aa1003e6     	mov	x6, x16
 1f838fc: aa0603e3     	mov	x3, x6
 1f83900: f9400226     	ldr	x6, [x17]
 1f83904: ab0400c5     	adds	x5, x6, x4
 1f83908: 540000a0     	b.eq	0x1f8391c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x2b0>
 1f8390c: 9e220083     	scvtf	s3, x4
 1f83910: 9e2200a4     	scvtf	s4, x5
 1f83914: 1e241863     	fdiv	s3, s3, s4
 1f83918: 14000002     	b	0x1f83920 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x2b4>
 1f8391c: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83920: 3dc00004     	ldr	q4, [x0]
 1f83924: 4ea4d442     	fsub.4s	v2, v2, v4
 1f83928: 3dc00025     	ldr	q5, [x1]
 1f8392c: 4e21d4a1     	fadd.4s	v1, v5, v1
 1f83930: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f83934: 9e2200c5     	scvtf	s5, x6
 1f83938: 4f859045     	fmul.4s	v5, v2, v5[0]
 1f8393c: 4e23d482     	fadd.4s	v2, v4, v3
 1f83940: 4e25cc61     	fmla.4s	v1, v3, v5
 1f83944: a93f963f     	stp	xzr, x5, [x17, #-0x8]
 1f83948: ad3f8806     	stp	q6, q2, [x0, #-0x10]
 1f8394c: ad3f8426     	stp	q6, q1, [x1, #-0x10]
 1f83950: eb17005f     	cmp	x2, x23
 1f83954: 54fff3c2     	b.hs	0x1f837cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x160>
 1f83958: d341fc66     	lsr	x6, x3, #1
 1f8395c: 91000442     	add	x2, x2, #0x1
 1f83960: 91004021     	add	x1, x1, #0x10
 1f83964: 91004000     	add	x0, x0, #0x10
 1f83968: 91002231     	add	x17, x17, #0x8
 1f8396c: aa0503e4     	mov	x4, x5
 1f83970: 360ffc63     	tbz	w3, #0x1, 0x1f838fc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x290>
 1f83974: 17ffff96     	b	0x1f837cc <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x160>
 1f83978: f9442be9     	ldr	x9, [sp, #0x850]
 1f8397c: 3dc117e0     	ldr	q0, [sp, #0x450]
 1f83980: 3dc017f5     	ldr	q21, [sp, #0x50]
 1f83984: f10006ea     	subs	x10, x23, #0x1
 1f83988: 540004e0     	b.eq	0x1f83a24 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x3b8>
 1f8398c: d1000aeb     	sub	x11, x23, #0x2
 1f83990: 92400548     	and	x8, x10, #0x3
 1f83994: f1000d7f     	cmp	x11, #0x3
 1f83998: 54000ea2     	b.hs	0x1f83b6c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x500>
 1f8399c: 5280002a     	mov	w10, #0x1               ; =1
 1f839a0: b4000428     	cbz	x8, 0x1f83a24 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x3b8>
 1f839a4: d37ced4c     	lsl	x12, x10, #4
 1f839a8: 910143eb     	add	x11, sp, #0x50
 1f839ac: 8b0c016b     	add	x11, x11, x12
 1f839b0: 911143ed     	add	x13, sp, #0x450
 1f839b4: 8b0c01ac     	add	x12, x13, x12
 1f839b8: 912143ed     	add	x13, sp, #0x850
 1f839bc: 8b0a0daa     	add	x10, x13, x10, lsl #3
 1f839c0: 14000010     	b	0x1f83a00 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x394>
 1f839c4: 9e2201c1     	scvtf	s1, x14
 1f839c8: 9e2201a2     	scvtf	s2, x13
 1f839cc: 1e221821     	fdiv	s1, s1, s2
 1f839d0: 3cc10582     	ldr	q2, [x12], #0x10
 1f839d4: 4ea0d442     	fsub.4s	v2, v2, v0
 1f839d8: 3cc10563     	ldr	q3, [x11], #0x10
 1f839dc: 4e23d6b5     	fadd.4s	v21, v21, v3
 1f839e0: 9e220123     	scvtf	s3, x9
 1f839e4: 4f819041     	fmul.4s	v1, v2, v1[0]
 1f839e8: 4f839042     	fmul.4s	v2, v2, v3[0]
 1f839ec: 4e21d400     	fadd.4s	v0, v0, v1
 1f839f0: 4e22cc35     	fmla.4s	v21, v1, v2
 1f839f4: aa0d03e9     	mov	x9, x13
 1f839f8: f1000508     	subs	x8, x8, #0x1
 1f839fc: 54000140     	b.eq	0x1f83a24 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x3b8>
 1f83a00: f840854e     	ldr	x14, [x10], #0x8
 1f83a04: ab0e012d     	adds	x13, x9, x14
 1f83a08: 54fffde1     	b.ne	0x1f839c4 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x358>
 1f83a0c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83a10: 17fffff0     	b	0x1f839d0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x364>
 1f83a14: 6f00e400     	movi.2d	v0, #0000000000000000
 1f83a18: 6f00e415     	movi.2d	v21, #0000000000000000
 1f83a1c: f10006ea     	subs	x10, x23, #0x1
 1f83a20: 54fffb61     	b.ne	0x1f8398c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x320>
 1f83a24: d37ef6c8     	lsl	x8, x22, #2
 1f83a28: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83a2c: eb080288     	subs	x8, x20, x8
 1f83a30: 5400022d     	b.le	0x1f83a74 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x408>
 1f83a34: 8b1612a9     	add	x9, x21, x22, lsl #4
 1f83a38: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83a3c: 5280002a     	mov	w10, #0x1               ; =1
 1f83a40: aa0803eb     	mov	x11, x8
 1f83a44: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83a48: bc404524     	ldr	s4, [x9], #0x4
 1f83a4c: 1e213885     	fsub	s5, s4, s1
 1f83a50: 9e230146     	ucvtf	s6, x10
 1f83a54: 1e2618a6     	fdiv	s6, s5, s6
 1f83a58: 1e262821     	fadd	s1, s1, s6
 1f83a5c: 1e213884     	fsub	s4, s4, s1
 1f83a60: 1f0408a2     	fmadd	s2, s5, s4, s2
 1f83a64: 9100054a     	add	x10, x10, #0x1
 1f83a68: f100056b     	subs	x11, x11, #0x1
 1f83a6c: 54fffee1     	b.ne	0x1f83a48 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x3dc>
 1f83a70: 14000004     	b	0x1f83a80 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x414>
 1f83a74: d2800008     	mov	x8, #0x0                ; =0
 1f83a78: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83a7c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83a80: 9e2202c5     	scvtf	s5, x22
 1f83a84: ab160109     	adds	x9, x8, x22
 1f83a88: 9e220124     	scvtf	s4, x9
 1f83a8c: 1e2418a6     	fdiv	s6, s5, s4
 1f83a90: 1e260c63     	fcsel	s3, s3, s6, eq
 1f83a94: ab160129     	adds	x9, x9, x22
 1f83a98: 6f00e406     	movi.2d	v6, #0000000000000000
 1f83a9c: 9e220127     	scvtf	s7, x9
 1f83aa0: 1e2718b0     	fdiv	s16, s5, s7
 1f83aa4: 1e300cd0     	fcsel	s16, s6, s16, eq
 1f83aa8: ab160129     	adds	x9, x9, x22
 1f83aac: 9e220131     	scvtf	s17, x9
 1f83ab0: 1e3118b2     	fdiv	s18, s5, s17
 1f83ab4: 1e320cd2     	fcsel	s18, s6, s18, eq
 1f83ab8: ab160129     	adds	x9, x9, x22
 1f83abc: 9e220133     	scvtf	s19, x9
 1f83ac0: 1e3318a5     	fdiv	s5, s5, s19
 1f83ac4: 1e250cc5     	fcsel	s5, s6, s5, eq
 1f83ac8: 1e213806     	fsub	s6, s0, s1
 1f83acc: 1e2608d3     	fmul	s19, s6, s6
 1f83ad0: 1e330873     	fmul	s19, s3, s19
 1f83ad4: 9e220114     	scvtf	s20, x8
 1f83ad8: 1f145673     	fmadd	s19, s19, s20, s21
 1f83adc: 1e332842     	fadd	s2, s2, s19
 1f83ae0: 5e0c0413     	mov	s19, v0[1]
 1f83ae4: 1f060461     	fmadd	s1, s3, s6, s1
 1f83ae8: 1e213a63     	fsub	s3, s19, s1
 1f83aec: 1e230866     	fmul	s6, s3, s3
 1f83af0: 1e260a06     	fmul	s6, s16, s6
 1f83af4: 5e0c06b3     	mov	s19, v21[1]
 1f83af8: 1f044cc4     	fmadd	s4, s6, s4, s19
 1f83afc: 1e242842     	fadd	s2, s2, s4
 1f83b00: 5e140404     	mov	s4, v0[2]
 1f83b04: 1f030601     	fmadd	s1, s16, s3, s1
 1f83b08: 1e213883     	fsub	s3, s4, s1
 1f83b0c: 1e230864     	fmul	s4, s3, s3
 1f83b10: 1e240a44     	fmul	s4, s18, s4
 1f83b14: 5e1406a6     	mov	s6, v21[2]
 1f83b18: 1f071884     	fmadd	s4, s4, s7, s6
 1f83b1c: 1e242842     	fadd	s2, s2, s4
 1f83b20: 1f030641     	fmadd	s1, s18, s3, s1
 1f83b24: 5e1c0400     	mov	s0, v0[3]
 1f83b28: 1e213803     	fsub	s3, s0, s1
 1f83b2c: 1f0304a0     	fmadd	s0, s5, s3, s1
 1f83b30: 5e1c06a1     	mov	s1, v21[3]
 1f83b34: 1e230863     	fmul	s3, s3, s3
 1f83b38: 1e2308a3     	fmul	s3, s5, s3
 1f83b3c: 1f110461     	fmadd	s1, s3, s17, s1
 1f83b40: 1e212841     	fadd	s1, s2, s1
 1f83b44: cb130288     	sub	x8, x20, x19
 1f83b48: 9e220102     	scvtf	s2, x8
 1f83b4c: 1e221821     	fdiv	s1, s1, s2
 1f83b50: 912943ff     	add	sp, sp, #0xa50
 1f83b54: a9447bfd     	ldp	x29, x30, [sp, #0x40]
 1f83b58: a9434ff4     	ldp	x20, x19, [sp, #0x30]
 1f83b5c: a94257f6     	ldp	x22, x21, [sp, #0x20]
 1f83b60: a9415ff8     	ldp	x24, x23, [sp, #0x10]
 1f83b64: a8c56ffc     	ldp	x28, x27, [sp], #0x50
 1f83b68: d65f03c0     	ret
 1f83b6c: 927ef54b     	and	x11, x10, #0xfffffffffffffffc
 1f83b70: 911143ea     	add	x10, sp, #0x450
 1f83b74: 9100814c     	add	x12, x10, #0x20
 1f83b78: 912143ea     	add	x10, sp, #0x850
 1f83b7c: 9100814d     	add	x13, x10, #0x20
 1f83b80: 910143ea     	add	x10, sp, #0x50
 1f83b84: 9100814e     	add	x14, x10, #0x20
 1f83b88: 5280002a     	mov	w10, #0x1               ; =1
 1f83b8c: 6f00e401     	movi.2d	v1, #0000000000000000
 1f83b90: 14000034     	b	0x1f83c60 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x5f4>
 1f83b94: 9e220223     	scvtf	s3, x17
 1f83b98: 9e2201e4     	scvtf	s4, x15
 1f83b9c: 1e241863     	fdiv	s3, s3, s4
 1f83ba0: ad7f9584     	ldp	q4, q5, [x12, #-0x10]
 1f83ba4: 4ea0d484     	fsub.4s	v4, v4, v0
 1f83ba8: ad7f9dc6     	ldp	q6, q7, [x14, #-0x10]
 1f83bac: 4e26d6a6     	fadd.4s	v6, v21, v6
 1f83bb0: 4f829082     	fmul.4s	v2, v4, v2[0]
 1f83bb4: 9e220130     	scvtf	s16, x9
 1f83bb8: 4f909084     	fmul.4s	v4, v4, v16[0]
 1f83bbc: 4e22d400     	fadd.4s	v0, v0, v2
 1f83bc0: 4e24cc46     	fmla.4s	v6, v2, v4
 1f83bc4: 4ea0d4a2     	fsub.4s	v2, v5, v0
 1f83bc8: 4e27d4c4     	fadd.4s	v4, v6, v7
 1f83bcc: 4f839043     	fmul.4s	v3, v2, v3[0]
 1f83bd0: 9e220205     	scvtf	s5, x16
 1f83bd4: 4f859042     	fmul.4s	v2, v2, v5[0]
 1f83bd8: 4e23d400     	fadd.4s	v0, v0, v3
 1f83bdc: a97fc1a9     	ldp	x9, x16, [x13, #-0x8]
 1f83be0: 9e220125     	scvtf	s5, x9
 1f83be4: 4e22cc64     	fmla.4s	v4, v3, v2
 1f83be8: ab0901e9     	adds	x9, x15, x9
 1f83bec: 9e220122     	scvtf	s2, x9
 1f83bf0: 1e2218a3     	fdiv	s3, s5, s2
 1f83bf4: 1e230c23     	fcsel	s3, s1, s3, eq
 1f83bf8: ad409985     	ldp	q5, q6, [x12, #0x10]
 1f83bfc: 4ea0d4a5     	fsub.4s	v5, v5, v0
 1f83c00: ad40c1c7     	ldp	q7, q16, [x14, #0x10]
 1f83c04: 4e27d484     	fadd.4s	v4, v4, v7
 1f83c08: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f83c0c: 9e2201e7     	scvtf	s7, x15
 1f83c10: 4f8790a5     	fmul.4s	v5, v5, v7[0]
 1f83c14: 4e23d400     	fadd.4s	v0, v0, v3
 1f83c18: 4e25cc64     	fmla.4s	v4, v3, v5
 1f83c1c: ab100129     	adds	x9, x9, x16
 1f83c20: 9e220203     	scvtf	s3, x16
 1f83c24: 9e220125     	scvtf	s5, x9
 1f83c28: 1e251863     	fdiv	s3, s3, s5
 1f83c2c: 1e230c23     	fcsel	s3, s1, s3, eq
 1f83c30: 4ea0d4c5     	fsub.4s	v5, v6, v0
 1f83c34: 4e30d495     	fadd.4s	v21, v4, v16
 1f83c38: 4f8390a3     	fmul.4s	v3, v5, v3[0]
 1f83c3c: 4f8290a2     	fmul.4s	v2, v5, v2[0]
 1f83c40: 4e23d400     	fadd.4s	v0, v0, v3
 1f83c44: 9100114a     	add	x10, x10, #0x4
 1f83c48: 4e22cc75     	fmla.4s	v21, v3, v2
 1f83c4c: 9101018c     	add	x12, x12, #0x40
 1f83c50: 910081ad     	add	x13, x13, #0x20
 1f83c54: 910101ce     	add	x14, x14, #0x40
 1f83c58: f100116b     	subs	x11, x11, #0x4
 1f83c5c: 54ffea20     	b.eq	0x1f839a0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x334>
 1f83c60: f85e81af     	ldur	x15, [x13, #-0x18]
 1f83c64: 6f00e402     	movi.2d	v2, #0000000000000000
 1f83c68: ab0f0130     	adds	x16, x9, x15
 1f83c6c: 54000080     	b.eq	0x1f83c7c <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x610>
 1f83c70: 9e2201e2     	scvtf	s2, x15
 1f83c74: 9e220203     	scvtf	s3, x16
 1f83c78: 1e231842     	fdiv	s2, s2, s3
 1f83c7c: f85f01b1     	ldur	x17, [x13, #-0x10]
 1f83c80: ab11020f     	adds	x15, x16, x17
 1f83c84: 54fff881     	b.ne	0x1f83b94 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x528>
 1f83c88: 6f00e403     	movi.2d	v3, #0000000000000000
 1f83c8c: 17ffffc5     	b	0x1f83ba0 <std::__1::pair<at::OpMathType<float>::type, at::OpMathType<float>::type> at::native::DEFAULT::RowwiseMomentsImpl<float, 64ll>(float const*, long long, long long)+0x534>
