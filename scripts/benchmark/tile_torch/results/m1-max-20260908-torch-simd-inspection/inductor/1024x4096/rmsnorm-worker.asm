
/tmp/luisa-torch-simd.IxI6kn/torch-run/inductor/ur/curje5trypvpayz2t3ffthspzqqgj34vio7n6hny2mxs4v7y6km2.main.so:	file format mach-o arm64

Disassembly of section __TEXT,__text:

000000000000086c <_kernel.omp_outlined>:
     86c: d10343ff     	sub	sp, sp, #0xd0
     870: 6d052beb     	stp	d11, d10, [sp, #0x50]
     874: 6d0623e9     	stp	d9, d8, [sp, #0x60]
     878: a9076ffc     	stp	x28, x27, [sp, #0x70]
     87c: a90867fa     	stp	x26, x25, [sp, #0x80]
     880: a9095ff8     	stp	x24, x23, [sp, #0x90]
     884: a90a57f6     	stp	x22, x21, [sp, #0xa0]
     888: a90b4ff4     	stp	x20, x19, [sp, #0xb0]
     88c: a90c7bfd     	stp	x29, x30, [sp, #0xc0]
     890: 910303fd     	add	x29, sp, #0xc0
     894: aa0503f3     	mov	x19, x5
     898: aa0403f4     	mov	x20, x4
     89c: aa0303f5     	mov	x21, x3
     8a0: aa0203f6     	mov	x22, x2
     8a4: aa0003f7     	mov	x23, x0
     8a8: 9400041f     	bl	0x1924 <dyld_stub_binder+0x1924>
     8ac: 52807ff8     	mov	w24, #0x3ff             ; =1023
     8b0: a9047ff8     	stp	x24, xzr, [sp, #0x40]
     8b4: 52800028     	mov	w8, #0x1                ; =1
     8b8: f9001fe8     	str	x8, [sp, #0x38]
     8bc: b90037ff     	str	wzr, [sp, #0x34]
     8c0: b94002f7     	ldr	w23, [x23]
     8c4: f90003e8     	str	x8, [sp]
     8c8: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     8cc: 91026000     	add	x0, x0, #0x98
     8d0: 9100d3e3     	add	x3, sp, #0x34
     8d4: 910123e4     	add	x4, sp, #0x48
     8d8: 910103e5     	add	x5, sp, #0x40
     8dc: 9100e3e6     	add	x6, sp, #0x38
     8e0: aa1703e1     	mov	x1, x23
     8e4: 52800442     	mov	w2, #0x22               ; =34
     8e8: 52800027     	mov	w7, #0x1                ; =1
     8ec: 940003f9     	bl	0x18d0 <dyld_stub_binder+0x18d0>
     8f0: a94467e8     	ldp	x8, x25, [sp, #0x40]
     8f4: f10ffd1f     	cmp	x8, #0x3ff
     8f8: 9a98b118     	csel	x24, x8, x24, lt
     8fc: f90023f8     	str	x24, [sp, #0x40]
     900: eb18033f     	cmp	x25, x24
     904: 5400026d     	b.le	0x950 <_kernel.omp_outlined+0xe4>
     908: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     90c: 9102c000     	add	x0, x0, #0xb0
     910: aa1703e1     	mov	x1, x23
     914: 940003ec     	bl	0x18c4 <dyld_stub_binder+0x18c4>
     918: 90000020     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
     91c: 91032000     	add	x0, x0, #0xc8
     920: aa1703e1     	mov	x1, x23
     924: 940003e5     	bl	0x18b8 <dyld_stub_binder+0x18b8>
     928: a94c7bfd     	ldp	x29, x30, [sp, #0xc0]
     92c: a94b4ff4     	ldp	x20, x19, [sp, #0xb0]
     930: a94a57f6     	ldp	x22, x21, [sp, #0xa0]
     934: a9495ff8     	ldp	x24, x23, [sp, #0x90]
     938: a94867fa     	ldp	x26, x25, [sp, #0x80]
     93c: a9476ffc     	ldp	x28, x27, [sp, #0x70]
     940: 6d4623e9     	ldp	d9, d8, [sp, #0x60]
     944: 6d452beb     	ldp	d11, d10, [sp, #0x50]
     948: 910343ff     	add	sp, sp, #0xd0
     94c: d65f03c0     	ret
     950: d372c73a     	lsl	x26, x25, #14
     954: 52a73008     	mov	w8, #0x39800000         ; =964689920
     958: 1e270108     	fmov	s8, w8
     95c: 5298b588     	mov	w8, #0xc5ac             ; =50604
     960: 72a6e4e8     	movk	w8, #0x3727, lsl #16
     964: 1e270109     	fmov	s9, w8
     968: 1e2e100a     	fmov	s10, #1.00000000
     96c: 14000005     	b	0x980 <_kernel.omp_outlined+0x114>
     970: 9140135a     	add	x26, x26, #0x4, lsl #12 ; =0x4000
     974: eb18033f     	cmp	x25, x24
     978: 91000739     	add	x25, x25, #0x1
     97c: 54fffc60     	b.eq	0x908 <_kernel.omp_outlined+0x9c>
     980: f94002c8     	ldr	x8, [x22]
     984: 8b1a0108     	add	x8, x8, x26
     988: 6f00e400     	movi.2d	v0, #0000000000000000
     98c: 92800069     	mov	x9, #-0x4               ; =-4
     990: 3cc10501     	ldr	q1, [x8], #0x10
     994: 6e21dc21     	fmul.4s	v1, v1, v1
     998: 4e21d400     	fadd.4s	v0, v0, v1
     99c: 91001129     	add	x9, x9, #0x4
     9a0: f13ff13f     	cmp	x9, #0xffc
     9a4: 54ffff63     	b.lo	0x990 <_kernel.omp_outlined+0x124>
     9a8: d280001b     	mov	x27, #0x0               ; =0
     9ac: 4e180401     	dup.2d	v1, v0[1]
     9b0: 4e21d400     	fadd.4s	v0, v0, v1
     9b4: 7e30d800     	faddp.2s	s0, v0
     9b8: f94002a8     	ldr	x8, [x21]
     9bc: bc397900     	str	s0, [x8, x25, lsl #2]
     9c0: 9280007c     	mov	x28, #-0x4              ; =-4
     9c4: f94002c8     	ldr	x8, [x22]
     9c8: 8b1a0108     	add	x8, x8, x26
     9cc: 3cfb6902     	ldr	q2, [x8, x27]
     9d0: f94002a8     	ldr	x8, [x21]
     9d4: bc797900     	ldr	s0, [x8, x25, lsl #2]
     9d8: f9400288     	ldr	x8, [x20]
     9dc: 3cfb6903     	ldr	q3, [x8, x27]
     9e0: 1e280800     	fmul	s0, s0, s8
     9e4: 1e292801     	fadd	s1, s0, s9
     9e8: 1e21c020     	fsqrt	s0, s1
     9ec: 1e202000     	fcmp	s0, s0
     9f0: 54000186     	b.vs	0xa20 <_kernel.omp_outlined+0x1b4>
     9f4: 1e201940     	fdiv	s0, s10, s0
     9f8: 4f809040     	fmul.4s	v0, v2, v0[0]
     9fc: 6e20dc60     	fmul.4s	v0, v3, v0
     a00: f9400268     	ldr	x8, [x19]
     a04: 8b1a0108     	add	x8, x8, x26
     a08: 3cbb6900     	str	q0, [x8, x27]
     a0c: 9100139c     	add	x28, x28, #0x4
     a10: 9100437b     	add	x27, x27, #0x10
     a14: f13ff39f     	cmp	x28, #0xffc
     a18: 54fffd63     	b.lo	0x9c4 <_kernel.omp_outlined+0x158>
     a1c: 17ffffd5     	b	0x970 <_kernel.omp_outlined+0x104>
     a20: 4ea11c20     	mov.16b	v0, v1
     a24: ad008be3     	stp	q3, q2, [sp, #0x10]
     a28: 940003c2     	bl	0x1930 <dyld_stub_binder+0x1930>
     a2c: ad408be3     	ldp	q3, q2, [sp, #0x10]
     a30: 17fffff1     	b	0x9f4 <_kernel.omp_outlined+0x188>
