
/Users/mike/.cache/uv/archive-v0/sC3F2za3_-HoN0aE/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000002335904 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const>:
 2335904: d10343ff     	sub	sp, sp, #0xd0
 2335908: 6d0623e9     	stp	d9, d8, [sp, #0x60]
 233590c: a9076ffc     	stp	x28, x27, [sp, #0x70]
 2335910: a90867fa     	stp	x26, x25, [sp, #0x80]
 2335914: a9095ff8     	stp	x24, x23, [sp, #0x90]
 2335918: a90a57f6     	stp	x22, x21, [sp, #0xa0]
 233591c: a90b4ff4     	stp	x20, x19, [sp, #0xb0]
 2335920: a90c7bfd     	stp	x29, x30, [sp, #0xc0]
 2335924: 910303fd     	add	x29, sp, #0xc0
 2335928: eb02003f     	cmp	x1, x2
 233592c: 9a82c028     	csel	x8, x1, x2, gt
 2335930: f9000fe8     	str	x8, [sp, #0x18]
 2335934: 5400248a     	b.ge	0x2335dc4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x4c0>
 2335938: aa0103f3     	mov	x19, x1
 233593c: aa0003fb     	mov	x27, x0
 2335940: d37ef438     	lsl	x24, x1, #2
 2335944: 52800048     	mov	w8, #0x2                ; =2
 2335948: 9e670100     	fmov	d0, x8
 233594c: 3d8003e0     	str	q0, [sp]
 2335950: 1e2e1008     	fmov	s8, #1.00000000
 2335954: 14000017     	b	0x23359b0 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0xac>
 2335958: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 233595c: f100113f     	cmp	x9, #0x4
 2335960: 52800088     	mov	w8, #0x4                ; =4
 2335964: 9a883128     	csel	x8, x9, x8, lo
 2335968: d37ef516     	lsl	x22, x8, #2
 233596c: 910143e0     	add	x0, sp, #0x50
 2335970: aa1503e1     	mov	x1, x21
 2335974: aa1603e2     	mov	x2, x22
 2335978: 3d8013e1     	str	q1, [sp, #0x40]
 233597c: 94b2af68     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335980: ad4203e1     	ldp	q1, q0, [sp, #0x40]
 2335984: 4f819000     	fmul.4s	v0, v0, v1[0]
 2335988: 3d8017e0     	str	q0, [sp, #0x50]
 233598c: 910143e1     	add	x1, sp, #0x50
 2335990: aa1503e0     	mov	x0, x21
 2335994: aa1603e2     	mov	x2, x22
 2335998: 94b2af61     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 233599c: 91000673     	add	x19, x19, #0x1
 23359a0: 91001318     	add	x24, x24, #0x4
 23359a4: f9400fe8     	ldr	x8, [sp, #0x18]
 23359a8: eb08027f     	cmp	x19, x8
 23359ac: 540020c0     	b.eq	0x2335dc4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x4c0>
 23359b0: a9402768     	ldp	x8, x9, [x27]
 23359b4: f940011c     	ldr	x28, [x8]
 23359b8: f940013a     	ldr	x26, [x9]
 23359bc: 9b137f48     	mul	x8, x26, x19
 23359c0: d37ef515     	lsl	x21, x8, #2
 23359c4: 8b150396     	add	x22, x28, x21
 23359c8: f9400b68     	ldr	x8, [x27, #0x10]
 23359cc: f9400108     	ldr	x8, [x8]
 23359d0: f9001fe8     	str	x8, [sp, #0x38]
 23359d4: f1000f5f     	cmp	x26, #0x3
 23359d8: 5400020c     	b.gt	0x2335a18 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x114>
 23359dc: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 23359e0: d37ef742     	lsl	x2, x26, #2
 23359e4: 910143e0     	add	x0, sp, #0x50
 23359e8: aa1603e1     	mov	x1, x22
 23359ec: 94b2af4c     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 23359f0: 3dc017e2     	ldr	q2, [sp, #0x50]
 23359f4: f100075f     	cmp	x26, #0x1
 23359f8: 540008ad     	b.le	0x2335b0c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x208>
 23359fc: 6f00e400     	movi.2d	v0, #0000000000000000
 2335a00: 6e042440     	mov.s	v0[0], v2[1]
 2335a04: 4e20f440     	fmax.4s	v0, v2, v0
 2335a08: f1000b5f     	cmp	x26, #0x2
 2335a0c: 54000641     	b.ne	0x2335ad4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1d0>
 2335a10: 4ea01c02     	mov.16b	v2, v0
 2335a14: 1400003e     	b	0x2335b0c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x208>
 2335a18: 3dc002c1     	ldr	q1, [x22]
 2335a1c: 927ef349     	and	x9, x26, #0x7ffffffffffffffc
 2335a20: f1001528     	subs	x8, x9, #0x5
 2335a24: 540001e3     	b.lo	0x2335a60 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x15c>
 2335a28: 9b18734a     	madd	x10, x26, x24, x28
 2335a2c: 9100414a     	add	x10, x10, #0x10
 2335a30: 5280008b     	mov	w11, #0x4               ; =4
 2335a34: 3cc10540     	ldr	q0, [x10], #0x10
 2335a38: 4e20f421     	fmax.4s	v1, v1, v0
 2335a3c: 9100116b     	add	x11, x11, #0x4
 2335a40: eb09017f     	cmp	x11, x9
 2335a44: 54ffff83     	b.lo	0x2335a34 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x130>
 2335a48: 927ef508     	and	x8, x8, #0xfffffffffffffffc
 2335a4c: 91002108     	add	x8, x8, #0x8
 2335a50: cb080357     	sub	x23, x26, x8
 2335a54: f10006ff     	cmp	x23, #0x1
 2335a58: 540000ca     	b.ge	0x2335a70 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x16c>
 2335a5c: 14000028     	b	0x2335afc <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f8>
 2335a60: 52800088     	mov	w8, #0x4                ; =4
 2335a64: cb080357     	sub	x23, x26, x8
 2335a68: f10006ff     	cmp	x23, #0x1
 2335a6c: 5400048b     	b.lt	0x2335afc <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f8>
 2335a70: 8b080ac1     	add	x1, x22, x8, lsl #2
 2335a74: f10012ff     	cmp	x23, #0x4
 2335a78: 54000081     	b.ne	0x2335a88 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x184>
 2335a7c: 3dc00020     	ldr	q0, [x1]
 2335a80: 4e20f420     	fmax.4s	v0, v1, v0
 2335a84: 1400001d     	b	0x2335af8 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f4>
 2335a88: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 2335a8c: f10012ff     	cmp	x23, #0x4
 2335a90: 52800088     	mov	w8, #0x4                ; =4
 2335a94: 9a8832e8     	csel	x8, x23, x8, lo
 2335a98: d37ef502     	lsl	x2, x8, #2
 2335a9c: 910143e0     	add	x0, sp, #0x50
 2335aa0: 3d8013e1     	str	q1, [sp, #0x40]
 2335aa4: 94b2af1e     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335aa8: ad4203e2     	ldp	q2, q0, [sp, #0x40]
 2335aac: 4e20f440     	fmax.4s	v0, v2, v0
 2335ab0: f1000eff     	cmp	x23, #0x3
 2335ab4: 540001c0     	b.eq	0x2335aec <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1e8>
 2335ab8: f1000aff     	cmp	x23, #0x2
 2335abc: 54000120     	b.eq	0x2335ae0 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1dc>
 2335ac0: f10006ff     	cmp	x23, #0x1
 2335ac4: 540001a1     	b.ne	0x2335af8 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f4>
 2335ac8: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335acc: 3dc16101     	ldr	q1, [x8, #0x580]
 2335ad0: 14000009     	b	0x2335af4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f0>
 2335ad4: 6f00e401     	movi.2d	v1, #0000000000000000
 2335ad8: 6e044441     	mov.s	v1[0], v2[2]
 2335adc: 1400000b     	b	0x2335b08 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x204>
 2335ae0: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335ae4: 3dc16501     	ldr	q1, [x8, #0x590]
 2335ae8: 14000003     	b	0x2335af4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x1f0>
 2335aec: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335af0: 3dc16901     	ldr	q1, [x8, #0x5a0]
 2335af4: 6ee11c40     	bif.16b	v0, v2, v1
 2335af8: 4ea01c01     	mov.16b	v1, v0
 2335afc: 6e014020     	ext.16b	v0, v1, v1, #0x8
 2335b00: 4e20f420     	fmax.4s	v0, v1, v0
 2335b04: 4ea00801     	rev64.4s	v1, v0
 2335b08: 4e21f402     	fmax.4s	v2, v0, v1
 2335b0c: eb1a03e8     	negs	x8, x26
 2335b10: 92400508     	and	x8, x8, #0x3
 2335b14: 92400749     	and	x9, x26, #0x3
 2335b18: da884528     	csneg	x8, x9, x8, mi
 2335b1c: cb080357     	sub	x23, x26, x8
 2335b20: f10006ff     	cmp	x23, #0x1
 2335b24: 3d800be2     	str	q2, [sp, #0x20]
 2335b28: d2800019     	mov	x25, #0x0               ; =0
 2335b2c: 540001eb     	b.lt	0x2335b68 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x264>
 2335b30: 4e040440     	dup.4s	v0, v2[0]
 2335b34: 3d8013e0     	str	q0, [sp, #0x40]
 2335b38: 9b187f48     	mul	x8, x26, x24
 2335b3c: f9401fe9     	ldr	x9, [sp, #0x38]
 2335b40: 8b080134     	add	x20, x9, x8
 2335b44: 8b08039c     	add	x28, x28, x8
 2335b48: 3cc10780     	ldr	q0, [x28], #0x10
 2335b4c: 3dc013e1     	ldr	q1, [sp, #0x40]
 2335b50: 4ea1d400     	fsub.4s	v0, v0, v1
 2335b54: 94a4d84c     	bl	0x4c6bc84 <_Sleef_expf4_u10>
 2335b58: 3c810680     	str	q0, [x20], #0x10
 2335b5c: 91001339     	add	x25, x25, #0x4
 2335b60: eb17033f     	cmp	x25, x23
 2335b64: 54ffff2b     	b.lt	0x2335b48 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x244>
 2335b68: f9401ffc     	ldr	x28, [sp, #0x38]
 2335b6c: 8b150395     	add	x21, x28, x21
 2335b70: cb190348     	sub	x8, x26, x25
 2335b74: f100051f     	cmp	x8, #0x1
 2335b78: 5400016b     	b.lt	0x2335ba4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2a0>
 2335b7c: 8b190ac1     	add	x1, x22, x25, lsl #2
 2335b80: f100111f     	cmp	x8, #0x4
 2335b84: 54000401     	b.ne	0x2335c04 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x300>
 2335b88: 3dc00020     	ldr	q0, [x1]
 2335b8c: 3dc00be1     	ldr	q1, [sp, #0x20]
 2335b90: 4e040421     	dup.4s	v1, v1[0]
 2335b94: 4ea1d400     	fsub.4s	v0, v0, v1
 2335b98: 94a4d83b     	bl	0x4c6bc84 <_Sleef_expf4_u10>
 2335b9c: d37ef728     	lsl	x8, x25, #2
 2335ba0: 3ca86aa0     	str	q0, [x21, x8]
 2335ba4: f9400768     	ldr	x8, [x27, #0x8]
 2335ba8: f9400116     	ldr	x22, [x8]
 2335bac: f1000edf     	cmp	x22, #0x3
 2335bb0: 5400056c     	b.gt	0x2335c5c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x358>
 2335bb4: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 2335bb8: d37ef6c2     	lsl	x2, x22, #2
 2335bbc: 910143e0     	add	x0, sp, #0x50
 2335bc0: aa1503e1     	mov	x1, x21
 2335bc4: 94b2aed6     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335bc8: 3dc017e0     	ldr	q0, [sp, #0x50]
 2335bcc: f10006df     	cmp	x22, #0x1
 2335bd0: 54000bcd     	b.le	0x2335d48 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x444>
 2335bd4: 6f00e401     	movi.2d	v1, #0000000000000000
 2335bd8: 6e042401     	mov.s	v1[0], v0[1]
 2335bdc: 9e6702c2     	fmov	d2, x22
 2335be0: 3dc003e3     	ldr	q3, [sp]
 2335be4: 6ee38c42     	cmeq.2d	v2, v2, v3
 2335be8: 4e080442     	dup.2d	v2, v2[0]
 2335bec: 4e21d401     	fadd.4s	v1, v0, v1
 2335bf0: 6f00e403     	movi.2d	v3, #0000000000000000
 2335bf4: 6e044403     	mov.s	v3[0], v0[2]
 2335bf8: 4e23d420     	fadd.4s	v0, v1, v3
 2335bfc: 6ea21c20     	bit.16b	v0, v1, v2
 2335c00: 14000052     	b	0x2335d48 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x444>
 2335c04: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 2335c08: f100111f     	cmp	x8, #0x4
 2335c0c: 52800089     	mov	w9, #0x4                ; =4
 2335c10: 9a893108     	csel	x8, x8, x9, lo
 2335c14: d37ef516     	lsl	x22, x8, #2
 2335c18: 910143e0     	add	x0, sp, #0x50
 2335c1c: aa1603e2     	mov	x2, x22
 2335c20: 94b2aebf     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335c24: 3dc017e0     	ldr	q0, [sp, #0x50]
 2335c28: 3dc00be1     	ldr	q1, [sp, #0x20]
 2335c2c: 4e040421     	dup.4s	v1, v1[0]
 2335c30: 4ea1d400     	fsub.4s	v0, v0, v1
 2335c34: 94a4d814     	bl	0x4c6bc84 <_Sleef_expf4_u10>
 2335c38: 8b190aa0     	add	x0, x21, x25, lsl #2
 2335c3c: 3d8017e0     	str	q0, [sp, #0x50]
 2335c40: 910143e1     	add	x1, sp, #0x50
 2335c44: aa1603e2     	mov	x2, x22
 2335c48: 94b2aeb5     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335c4c: f9400768     	ldr	x8, [x27, #0x8]
 2335c50: f9400116     	ldr	x22, [x8]
 2335c54: f1000edf     	cmp	x22, #0x3
 2335c58: 54fffaed     	b.le	0x2335bb4 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x2b0>
 2335c5c: 3dc002a1     	ldr	q1, [x21]
 2335c60: 927ef2c9     	and	x9, x22, #0x7ffffffffffffffc
 2335c64: f1001528     	subs	x8, x9, #0x5
 2335c68: 54000203     	b.lo	0x2335ca8 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x3a4>
 2335c6c: 9b187f4a     	mul	x10, x26, x24
 2335c70: 8b0a038a     	add	x10, x28, x10
 2335c74: 9100414a     	add	x10, x10, #0x10
 2335c78: 5280008b     	mov	w11, #0x4               ; =4
 2335c7c: 3cc10540     	ldr	q0, [x10], #0x10
 2335c80: 4e20d421     	fadd.4s	v1, v1, v0
 2335c84: 9100116b     	add	x11, x11, #0x4
 2335c88: eb09017f     	cmp	x11, x9
 2335c8c: 54ffff83     	b.lo	0x2335c7c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x378>
 2335c90: 927ef508     	and	x8, x8, #0xfffffffffffffffc
 2335c94: 91002108     	add	x8, x8, #0x8
 2335c98: cb0802d7     	sub	x23, x22, x8
 2335c9c: f10006ff     	cmp	x23, #0x1
 2335ca0: 540000ca     	b.ge	0x2335cb8 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x3b4>
 2335ca4: 14000025     	b	0x2335d38 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x434>
 2335ca8: 52800088     	mov	w8, #0x4                ; =4
 2335cac: cb0802d7     	sub	x23, x22, x8
 2335cb0: f10006ff     	cmp	x23, #0x1
 2335cb4: 5400042b     	b.lt	0x2335d38 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x434>
 2335cb8: 8b080aa1     	add	x1, x21, x8, lsl #2
 2335cbc: f10012ff     	cmp	x23, #0x4
 2335cc0: 54000081     	b.ne	0x2335cd0 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x3cc>
 2335cc4: 3dc00020     	ldr	q0, [x1]
 2335cc8: 4e21d400     	fadd.4s	v0, v0, v1
 2335ccc: 1400001a     	b	0x2335d34 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x430>
 2335cd0: a9057fff     	stp	xzr, xzr, [sp, #0x50]
 2335cd4: f10012ff     	cmp	x23, #0x4
 2335cd8: 52800088     	mov	w8, #0x4                ; =4
 2335cdc: 9a8832e8     	csel	x8, x23, x8, lo
 2335ce0: d37ef502     	lsl	x2, x8, #2
 2335ce4: 910143e0     	add	x0, sp, #0x50
 2335ce8: 3d8013e1     	str	q1, [sp, #0x40]
 2335cec: 94b2ae8c     	bl	0x4fe171c <_zunmqr_+0x4fe171c>
 2335cf0: ad4203e2     	ldp	q2, q0, [sp, #0x40]
 2335cf4: 4e22d400     	fadd.4s	v0, v0, v2
 2335cf8: f1000eff     	cmp	x23, #0x3
 2335cfc: 54000160     	b.eq	0x2335d28 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x424>
 2335d00: f1000aff     	cmp	x23, #0x2
 2335d04: 540000c0     	b.eq	0x2335d1c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x418>
 2335d08: f10006ff     	cmp	x23, #0x1
 2335d0c: 54000141     	b.ne	0x2335d34 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x430>
 2335d10: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335d14: 3dc16101     	ldr	q1, [x8, #0x580]
 2335d18: 14000006     	b	0x2335d30 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x42c>
 2335d1c: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335d20: 3dc16501     	ldr	q1, [x8, #0x590]
 2335d24: 14000003     	b	0x2335d30 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x42c>
 2335d28: d0086308     	adrp	x8, 0x12f97000 <std::__1::pair<at::OpMathType<c10::BFloat16>::type, at::OpMathType<c10::BFloat16>::type> at::native::DEFAULT::RowwiseMomentsImpl<c10::BFloat16, 64ll>(c10::BFloat16 const*, long long, long long)::c_vecs+0xe0>
 2335d2c: 3dc16901     	ldr	q1, [x8, #0x5a0]
 2335d30: 6ee11c40     	bif.16b	v0, v2, v1
 2335d34: 4ea01c01     	mov.16b	v1, v0
 2335d38: 6e014020     	ext.16b	v0, v1, v1, #0x8
 2335d3c: 4e21d400     	fadd.4s	v0, v0, v1
 2335d40: 4e0c0401     	dup.4s	v1, v0[1]
 2335d44: 4e21d400     	fadd.4s	v0, v0, v1
 2335d48: eb1603e8     	negs	x8, x22
 2335d4c: 92400508     	and	x8, x8, #0x3
 2335d50: 924006c9     	and	x9, x22, #0x3
 2335d54: da884528     	csneg	x8, x9, x8, mi
 2335d58: cb0802c9     	sub	x9, x22, x8
 2335d5c: 1e201901     	fdiv	s1, s8, s0
 2335d60: f100053f     	cmp	x9, #0x1
 2335d64: 540001ab     	b.lt	0x2335d98 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x494>
 2335d68: d2800008     	mov	x8, #0x0                ; =0
 2335d6c: 9b18734a     	madd	x10, x26, x24, x28
 2335d70: 3dc00140     	ldr	q0, [x10]
 2335d74: 4f819000     	fmul.4s	v0, v0, v1[0]
 2335d78: 3c810540     	str	q0, [x10], #0x10
 2335d7c: 91001108     	add	x8, x8, #0x4
 2335d80: eb09011f     	cmp	x8, x9
 2335d84: 54ffff6b     	b.lt	0x2335d70 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x46c>
 2335d88: cb0802c9     	sub	x9, x22, x8
 2335d8c: f100053f     	cmp	x9, #0x1
 2335d90: 540000ca     	b.ge	0x2335da8 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x4a4>
 2335d94: 17ffff02     	b	0x233599c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x98>
 2335d98: d2800008     	mov	x8, #0x0                ; =0
 2335d9c: cb0802c9     	sub	x9, x22, x8
 2335da0: f100053f     	cmp	x9, #0x1
 2335da4: 54ffdfcb     	b.lt	0x233599c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x98>
 2335da8: 8b080ab5     	add	x21, x21, x8, lsl #2
 2335dac: f100113f     	cmp	x9, #0x4
 2335db0: 54ffdd41     	b.ne	0x2335958 <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x54>
 2335db4: 3dc002a0     	ldr	q0, [x21]
 2335db8: 4f819000     	fmul.4s	v0, v0, v1[0]
 2335dbc: 3d8002a0     	str	q0, [x21]
 2335dc0: 17fffef7     	b	0x233599c <std::__1::enable_if<std::is_same_v<float, at::OpMathType<float>::type>, void>::type at::native::(anonymous namespace)::_vec_softmax_lastdim<float>(float const*, float*, long long, long long)::'lambda'(long long, long long)::operator()(long long, long long) const+0x98>
 2335dc4: a94c7bfd     	ldp	x29, x30, [sp, #0xc0]
 2335dc8: a94b4ff4     	ldp	x20, x19, [sp, #0xb0]
 2335dcc: a94a57f6     	ldp	x22, x21, [sp, #0xa0]
 2335dd0: a9495ff8     	ldp	x24, x23, [sp, #0x90]
 2335dd4: a94867fa     	ldp	x26, x25, [sp, #0x80]
 2335dd8: a9476ffc     	ldp	x28, x27, [sp, #0x70]
 2335ddc: 6d4623e9     	ldp	d9, d8, [sp, #0x60]
 2335de0: 910343ff     	add	sp, sp, #0xd0
 2335de4: d65f03c0     	ret
