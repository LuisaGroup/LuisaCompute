
/tmp/luisa-torch-simd.IxI6kn/torch-small/inductor/5a/c5a76bfhvklxdy7efdg5zaqsw77fycygxch4lij4apfiqp6xcfhi.main.so:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000000000874 <_kernel.omp_outlined>:
     874: d10343ff     	sub	sp, sp, #0xd0
     878: 6d052beb     	stp	d11, d10, [sp, #0x50]
     87c: 6d0623e9     	stp	d9, d8, [sp, #0x60]
     880: a9076ffc     	stp	x28, x27, [sp, #0x70]
     884: a90867fa     	stp	x26, x25, [sp, #0x80]
     888: a9095ff8     	stp	x24, x23, [sp, #0x90]
     88c: a90a57f6     	stp	x22, x21, [sp, #0xa0]
     890: a90b4ff4     	stp	x20, x19, [sp, #0xb0]
     894: a90c7bfd     	stp	x29, x30, [sp, #0xc0]
     898: 910303fd     	add	x29, sp, #0xc0
     89c: aa0503f3     	mov	x19, x5
     8a0: aa0403f4     	mov	x20, x4
     8a4: aa0303f5     	mov	x21, x3
     8a8: aa0203f6     	mov	x22, x2
     8ac: aa0003f7     	mov	x23, x0
     8b0: 9400041f     	bl	0x192c <dyld_stub_binder+0x192c>
     8b4: 528007f8     	mov	w24, #0x3f              ; =63
     8b8: a9047ff8     	stp	x24, xzr, [sp, #0x40]
     8bc: 52800028     	mov	w8, #0x1                ; =1
     8c0: f9001fe8     	str	x8, [sp, #0x38]
     8c4: b90037ff     	str	wzr, [sp, #0x34]
     8c8: b94002f7     	ldr	w23, [x23]
     8cc: f90003e8     	str	x8, [sp]
     8d0: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     8d4: 91026000     	add	x0, x0, #0x98
     8d8: 9100d3e3     	add	x3, sp, #0x34
     8dc: 910123e4     	add	x4, sp, #0x48
     8e0: 910103e5     	add	x5, sp, #0x40
     8e4: 9100e3e6     	add	x6, sp, #0x38
     8e8: aa1703e1     	mov	x1, x23
     8ec: 52800442     	mov	w2, #0x22               ; =34
     8f0: 52800027     	mov	w7, #0x1                ; =1
     8f4: 940003f9     	bl	0x18d8 <dyld_stub_binder+0x18d8>
     8f8: a94467e8     	ldp	x8, x25, [sp, #0x40]
     8fc: f100fd1f     	cmp	x8, #0x3f
     900: 9a98b118     	csel	x24, x8, x24, lt
     904: f90023f8     	str	x24, [sp, #0x40]
     908: eb18033f     	cmp	x25, x24
     90c: 5400026d     	b.le	0x958 <_kernel.omp_outlined+0xe4>
     910: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     914: 9102c000     	add	x0, x0, #0xb0
     918: aa1703e1     	mov	x1, x23
     91c: 940003ec     	bl	0x18cc <dyld_stub_binder+0x18cc>
     920: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     924: 91032000     	add	x0, x0, #0xc8
     928: aa1703e1     	mov	x1, x23
     92c: 940003e5     	bl	0x18c0 <dyld_stub_binder+0x18c0>
     930: a94c7bfd     	ldp	x29, x30, [sp, #0xc0]
     934: a94b4ff4     	ldp	x20, x19, [sp, #0xb0]
     938: a94a57f6     	ldp	x22, x21, [sp, #0xa0]
     93c: a9495ff8     	ldp	x24, x23, [sp, #0x90]
     940: a94867fa     	ldp	x26, x25, [sp, #0x80]
     944: a9476ffc     	ldp	x28, x27, [sp, #0x70]
     948: 6d4623e9     	ldp	d9, d8, [sp, #0x60]
     94c: 6d452beb     	ldp	d11, d10, [sp, #0x50]
     950: 910343ff     	add	sp, sp, #0xd0
     954: d65f03c0     	ret
     958: d376d73a     	lsl	x26, x25, #10
     95c: 52a77008     	mov	w8, #0x3b800000         ; =998244352
     960: 1e270108     	fmov	s8, w8
     964: 5298b588     	mov	w8, #0xc5ac             ; =50604
     968: 72a6e4e8     	movk	w8, #0x3727, lsl #16
     96c: 1e270109     	fmov	s9, w8
     970: 1e2e100a     	fmov	s10, #1.00000000
     974: 14000005     	b	0x988 <_kernel.omp_outlined+0x114>
     978: 9110035a     	add	x26, x26, #0x400
     97c: eb18033f     	cmp	x25, x24
     980: 91000739     	add	x25, x25, #0x1
     984: 54fffc60     	b.eq	0x910 <_kernel.omp_outlined+0x9c>
     988: f94002c8     	ldr	x8, [x22]
     98c: 8b1a0108     	add	x8, x8, x26
     990: 6f00e400     	movi.2d	v0, #0000000000000000
     994: 92800069     	mov	x9, #-0x4               ; =-4
     998: 3cc10501     	ldr	q1, [x8], #0x10
     99c: 6e21dc21     	fmul.4s	v1, v1, v1
     9a0: 4e21d400     	fadd.4s	v0, v0, v1
     9a4: 91001129     	add	x9, x9, #0x4
     9a8: f103f13f     	cmp	x9, #0xfc
     9ac: 54ffff63     	b.lo	0x998 <_kernel.omp_outlined+0x124>
     9b0: d280001b     	mov	x27, #0x0               ; =0
     9b4: 4e180401     	dup.2d	v1, v0[1]
     9b8: 4e21d400     	fadd.4s	v0, v0, v1
     9bc: 7e30d800     	faddp.2s	s0, v0
     9c0: f94002a8     	ldr	x8, [x21]
     9c4: bc397900     	str	s0, [x8, x25, lsl #2]
     9c8: 9280007c     	mov	x28, #-0x4              ; =-4
     9cc: f94002c8     	ldr	x8, [x22]
     9d0: 8b1a0108     	add	x8, x8, x26
     9d4: 3cfb6902     	ldr	q2, [x8, x27]
     9d8: f94002a8     	ldr	x8, [x21]
     9dc: bc797900     	ldr	s0, [x8, x25, lsl #2]
     9e0: f9400288     	ldr	x8, [x20]
     9e4: 3cfb6903     	ldr	q3, [x8, x27]
     9e8: 1e280800     	fmul	s0, s0, s8
     9ec: 1e292801     	fadd	s1, s0, s9
     9f0: 1e21c020     	fsqrt	s0, s1
     9f4: 1e202000     	fcmp	s0, s0
     9f8: 54000186     	b.vs	0xa28 <_kernel.omp_outlined+0x1b4>
     9fc: 1e201940     	fdiv	s0, s10, s0
     a00: 4f809040     	fmul.4s	v0, v2, v0[0]
     a04: 6e20dc60     	fmul.4s	v0, v3, v0
     a08: f9400268     	ldr	x8, [x19]
     a0c: 8b1a0108     	add	x8, x8, x26
     a10: 3cbb6900     	str	q0, [x8, x27]
     a14: 9100139c     	add	x28, x28, #0x4
     a18: 9100437b     	add	x27, x27, #0x10
     a1c: f103f39f     	cmp	x28, #0xfc
     a20: 54fffd63     	b.lo	0x9cc <_kernel.omp_outlined+0x158>
     a24: 17ffffd5     	b	0x978 <_kernel.omp_outlined+0x104>
     a28: 4ea11c20     	mov.16b	v0, v1
     a2c: ad008be3     	stp	q3, q2, [sp, #0x10]
     a30: 940003c2     	bl	0x1938 <dyld_stub_binder+0x1938>
     a34: ad408be3     	ldp	q3, q2, [sp, #0x10]
     a38: 17fffff1     	b	0x9fc <_kernel.omp_outlined+0x188>
