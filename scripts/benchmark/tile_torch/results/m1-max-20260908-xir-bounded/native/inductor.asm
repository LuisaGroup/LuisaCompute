
/private/tmp/luisa-xir-bounded.kADecq/native/inductor/k4/ck4nfag4k45q2yoscptekuktu2yh27whx467wsxbf23ntdpvi64m.main.so:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000000000006a0 <_kernel>:
     6a0: d102c3ff     	sub	sp, sp, #0xb0
     6a4: 6d042beb     	stp	d11, d10, [sp, #0x40]
     6a8: 6d0523e9     	stp	d9, d8, [sp, #0x50]
     6ac: a90667fa     	stp	x26, x25, [sp, #0x60]
     6b0: a9075ff8     	stp	x24, x23, [sp, #0x70]
     6b4: a90857f6     	stp	x22, x21, [sp, #0x80]
     6b8: a9094ff4     	stp	x20, x19, [sp, #0x90]
     6bc: a90a7bfd     	stp	x29, x30, [sp, #0xa0]
     6c0: 910283fd     	add	x29, sp, #0xa0
     6c4: d2800013     	mov	x19, #0x0               ; =0
     6c8: b90023ff     	str	wzr, [sp, #0x20]
     6cc: 90000034     	adrp	x20, 0x4000 <dyld_stub_binder+0x4000>
     6d0: f9404a94     	ldr	x20, [x20, #0x90]
     6d4: 910083e8     	add	x8, sp, #0x20
     6d8: f9000288     	str	x8, [x20]
     6dc: 52a77008     	mov	w8, #0x3b800000         ; =998244352
     6e0: 1e270108     	fmov	s8, w8
     6e4: 5298b588     	mov	w8, #0xc5ac             ; =50604
     6e8: 72a6e4e8     	movk	w8, #0x3727, lsl #16
     6ec: 1e270109     	fmov	s9, w8
     6f0: 1e2e100a     	fmov	s10, #1.00000000
     6f4: 14000006     	b	0x70c <_kernel+0x6c>
     6f8: 91000673     	add	x19, x19, #0x1
     6fc: 91100000     	add	x0, x0, #0x400
     700: 91100063     	add	x3, x3, #0x400
     704: f101027f     	cmp	x19, #0x40
     708: 540005c0     	b.eq	0x7c0 <_kernel+0x120>
     70c: 6f00e400     	movi.2d	v0, #0000000000000000
     710: 92800068     	mov	x8, #-0x4               ; =-4
     714: aa0003e9     	mov	x9, x0
     718: 3cc10521     	ldr	q1, [x9], #0x10
     71c: 6e21dc21     	fmul.4s	v1, v1, v1
     720: 4e21d400     	fadd.4s	v0, v0, v1
     724: 91001108     	add	x8, x8, #0x4
     728: f103f11f     	cmp	x8, #0xfc
     72c: 54ffff63     	b.lo	0x718 <_kernel+0x78>
     730: d2800015     	mov	x21, #0x0               ; =0
     734: 4e180401     	dup.2d	v1, v0[1]
     738: 4e21d400     	fadd.4s	v0, v0, v1
     73c: 7e30d800     	faddp.2s	s0, v0
     740: bc337840     	str	s0, [x2, x19, lsl #2]
     744: 92800076     	mov	x22, #-0x4              ; =-4
     748: 3cf56802     	ldr	q2, [x0, x21]
     74c: bc737840     	ldr	s0, [x2, x19, lsl #2]
     750: 3cf56823     	ldr	q3, [x1, x21]
     754: 1e280800     	fmul	s0, s0, s8
     758: 1e292801     	fadd	s1, s0, s9
     75c: 1e21c020     	fsqrt	s0, s1
     760: 1e202000     	fcmp	s0, s0
     764: 54000146     	b.vs	0x78c <_kernel+0xec>
     768: 1e201940     	fdiv	s0, s10, s0
     76c: 4f809040     	fmul.4s	v0, v2, v0[0]
     770: 6e20dc60     	fmul.4s	v0, v3, v0
     774: 3cb56860     	str	q0, [x3, x21]
     778: 910012d6     	add	x22, x22, #0x4
     77c: 910042b5     	add	x21, x21, #0x10
     780: f103f2df     	cmp	x22, #0xfc
     784: 54fffe23     	b.lo	0x748 <_kernel+0xa8>
     788: 17ffffdc     	b	0x6f8 <_kernel+0x58>
     78c: 4ea11c20     	mov.16b	v0, v1
     790: aa0303f7     	mov	x23, x3
     794: aa0203f9     	mov	x25, x2
     798: aa0103f8     	mov	x24, x1
     79c: aa0003fa     	mov	x26, x0
     7a0: ad000be3     	stp	q3, q2, [sp]
     7a4: 940003ef     	bl	0x1760 <dyld_stub_binder+0x1760>
     7a8: ad400be3     	ldp	q3, q2, [sp]
     7ac: aa1a03e0     	mov	x0, x26
     7b0: aa1803e1     	mov	x1, x24
     7b4: aa1903e2     	mov	x2, x25
     7b8: aa1703e3     	mov	x3, x23
     7bc: 17ffffeb     	b	0x768 <_kernel+0xc8>
     7c0: f900029f     	str	xzr, [x20]
     7c4: 910083e8     	add	x8, sp, #0x20
     7c8: b8bfc108     	ldapr	w8, [x8]
     7cc: 35000148     	cbnz	w8, 0x7f4 <_kernel+0x154>
     7d0: a94a7bfd     	ldp	x29, x30, [sp, #0xa0]
     7d4: a9494ff4     	ldp	x20, x19, [sp, #0x90]
     7d8: a94857f6     	ldp	x22, x21, [sp, #0x80]
     7dc: a9475ff8     	ldp	x24, x23, [sp, #0x70]
     7e0: a94667fa     	ldp	x26, x25, [sp, #0x60]
     7e4: 6d4523e9     	ldp	d9, d8, [sp, #0x50]
     7e8: 6d442beb     	ldp	d11, d10, [sp, #0x40]
     7ec: 9102c3ff     	add	sp, sp, #0xb0
     7f0: d65f03c0     	ret
     7f4: 52800200     	mov	w0, #0x10               ; =16
     7f8: 940003bf     	bl	0x16f4 <dyld_stub_binder+0x16f4>
     7fc: aa0003f3     	mov	x19, x0
     800: 528067a8     	mov	w8, #0x33d              ; =829
     804: b90027e8     	str	w8, [sp, #0x24]
     808: b0000000     	adrp	x0, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     80c: 9128c000     	add	x0, x0, #0xa30
     810: b0000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     814: 912af021     	add	x1, x1, #0xabc
     818: b0000003     	adrp	x3, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     81c: 912b9c63     	add	x3, x3, #0xae7
     820: b0000004     	adrp	x4, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     824: 912d9484     	add	x4, x4, #0xb65
     828: b0000002     	adrp	x2, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     82c: 912b9042     	add	x2, x2, #0xae4
     830: 9100a3e8     	add	x8, sp, #0x28
     834: 910093e5     	add	x5, sp, #0x24
     838: b0000007     	adrp	x7, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     83c: 912d9ce7     	add	x7, x7, #0xb67
     840: aa0203e6     	mov	x6, x2
     844: 9400001d     	bl	0x8b8 <auto c10::detail::torchCheckMsgImpl<char [40], char [3], char [126], char [2], int, char [3], char [18]>(char const*, char const (&) [40], char const (&) [3], char const (&) [126], char const (&) [2], int const&, char const (&) [3], char const (&) [18])>
     848: 52800035     	mov	w21, #0x1               ; =1
     84c: 9100a3e1     	add	x1, sp, #0x28
     850: aa1303e0     	mov	x0, x19
     854: 9400037b     	bl	0x1640 <dyld_stub_binder+0x1640>
     858: 52800015     	mov	w21, #0x0               ; =0
     85c: 90000021     	adrp	x1, 0x4000 <dyld_stub_binder+0x4000>
     860: f9400c21     	ldr	x1, [x1, #0x18]
     864: 90000022     	adrp	x2, 0x4000 <dyld_stub_binder+0x4000>
     868: f9400442     	ldr	x2, [x2, #0x8]
     86c: aa1303e0     	mov	x0, x19
     870: 940003ad     	bl	0x1724 <dyld_stub_binder+0x1724>
     874: d4200020     	brk	#0x1
     878: aa0003f4     	mov	x20, x0
     87c: 39c0ffe8     	ldrsb	w8, [sp, #0x3f]
     880: 36f800e8     	tbz	w8, #0x1f, 0x89c <_kernel+0x1fc>
     884: f94017e0     	ldr	x0, [sp, #0x28]
     888: f9401fe8     	ldr	x8, [sp, #0x38]
     88c: 9240f901     	and	x1, x8, #0x7fffffffffffffff
     890: 94000393     	bl	0x16dc <dyld_stub_binder+0x16dc>
     894: 370000b5     	tbnz	w21, #0x0, 0x8a8 <_kernel+0x208>
     898: 14000006     	b	0x8b0 <_kernel+0x210>
     89c: 35000075     	cbnz	w21, 0x8a8 <_kernel+0x208>
     8a0: 14000004     	b	0x8b0 <_kernel+0x210>
     8a4: aa0003f4     	mov	x20, x0
     8a8: aa1303e0     	mov	x0, x19
     8ac: 9400039b     	bl	0x1718 <dyld_stub_binder+0x1718>
     8b0: aa1403e0     	mov	x0, x20
     8b4: 94000354     	bl	0x1604 <dyld_stub_binder+0x1604>

00000000000008b8 <auto c10::detail::torchCheckMsgImpl<char [40], char [3], char [126], char [2], int, char [3], char [18]>(char const*, char const (&) [40], char const (&) [3], char const (&) [126], char const (&) [2], int const&, char const (&) [3], char const (&) [18])>:
     8b8: d10103ff     	sub	sp, sp, #0x40
     8bc: a9037bfd     	stp	x29, x30, [sp, #0x30]
     8c0: 9100c3fd     	add	x29, sp, #0x30
     8c4: aa0503e9     	mov	x9, x5
     8c8: a93f07a2     	stp	x2, x1, [x29, #-0x10]
     8cc: a9010fe4     	stp	x4, x3, [sp, #0x10]
     8d0: a9001be7     	stp	x7, x6, [sp]
     8d4: d10023a0     	sub	x0, x29, #0x8
     8d8: d10043a1     	sub	x1, x29, #0x10
     8dc: 910063e2     	add	x2, sp, #0x18
     8e0: 910043e3     	add	x3, sp, #0x10
     8e4: 910023e5     	add	x5, sp, #0x8
     8e8: 910003e6     	mov	x6, sp
     8ec: aa0903e4     	mov	x4, x9
     8f0: 94000004     	bl	0x900 <c10::detail::_str_wrapper<char const*, char const*, char const*, char const*, int const&, char const*, char const*>::call(char const* const&, char const* const&, char const* const&, char const* const&, int const&, char const* const&, char const* const&)>
     8f4: a9437bfd     	ldp	x29, x30, [sp, #0x30]
     8f8: 910103ff     	add	sp, sp, #0x40
     8fc: d65f03c0     	ret

0000000000000900 <c10::detail::_str_wrapper<char const*, char const*, char const*, char const*, int const&, char const*, char const*>::call(char const* const&, char const* const&, char const* const&, char const* const&, int const&, char const* const&, char const* const&)>:
     900: d105c3ff     	sub	sp, sp, #0x170
     904: a9116ffc     	stp	x28, x27, [sp, #0x110]
     908: a91267fa     	stp	x26, x25, [sp, #0x120]
     90c: a9135ff8     	stp	x24, x23, [sp, #0x130]
     910: a91457f6     	stp	x22, x21, [sp, #0x140]
     914: a9154ff4     	stp	x20, x19, [sp, #0x150]
     918: a9167bfd     	stp	x29, x30, [sp, #0x160]
     91c: 910583fd     	add	x29, sp, #0x160
     920: aa0603f4     	mov	x20, x6
     924: aa0503f5     	mov	x21, x5
     928: aa0403f6     	mov	x22, x4
     92c: aa0303f7     	mov	x23, x3
     930: aa0203f8     	mov	x24, x2
     934: aa0103f9     	mov	x25, x1
     938: aa0003fa     	mov	x26, x0
     93c: aa0803f3     	mov	x19, x8
     940: 910023e0     	add	x0, sp, #0x8
     944: 9400005c     	bl	0xab4 <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::basic_ostringstream[abi:nqe220108]()>
     948: f940035a     	ldr	x26, [x26]
     94c: aa1a03e0     	mov	x0, x26
     950: 94000387     	bl	0x176c <dyld_stub_binder+0x176c>
     954: aa0003e2     	mov	x2, x0
     958: 910023e0     	add	x0, sp, #0x8
     95c: aa1a03e1     	mov	x1, x26
     960: 9400013d     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     964: f9400339     	ldr	x25, [x25]
     968: aa1903e0     	mov	x0, x25
     96c: 94000380     	bl	0x176c <dyld_stub_binder+0x176c>
     970: aa0003e2     	mov	x2, x0
     974: 910023e0     	add	x0, sp, #0x8
     978: aa1903e1     	mov	x1, x25
     97c: 94000136     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     980: f9400318     	ldr	x24, [x24]
     984: aa1803e0     	mov	x0, x24
     988: 94000379     	bl	0x176c <dyld_stub_binder+0x176c>
     98c: aa0003e2     	mov	x2, x0
     990: 910023e0     	add	x0, sp, #0x8
     994: aa1803e1     	mov	x1, x24
     998: 9400012f     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     99c: f94002f7     	ldr	x23, [x23]
     9a0: aa1703e0     	mov	x0, x23
     9a4: 94000372     	bl	0x176c <dyld_stub_binder+0x176c>
     9a8: aa0003e2     	mov	x2, x0
     9ac: 910023e0     	add	x0, sp, #0x8
     9b0: aa1703e1     	mov	x1, x23
     9b4: 94000128     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     9b8: b94002c1     	ldr	w1, [x22]
     9bc: 910023e0     	add	x0, sp, #0x8
     9c0: 9400032f     	bl	0x167c <dyld_stub_binder+0x167c>
     9c4: f94002b5     	ldr	x21, [x21]
     9c8: aa1503e0     	mov	x0, x21
     9cc: 94000368     	bl	0x176c <dyld_stub_binder+0x176c>
     9d0: aa0003e2     	mov	x2, x0
     9d4: 910023e0     	add	x0, sp, #0x8
     9d8: aa1503e1     	mov	x1, x21
     9dc: 9400011e     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     9e0: f9400294     	ldr	x20, [x20]
     9e4: aa1403e0     	mov	x0, x20
     9e8: 94000361     	bl	0x176c <dyld_stub_binder+0x176c>
     9ec: aa0003e2     	mov	x2, x0
     9f0: 910023f5     	add	x21, sp, #0x8
     9f4: 910023e0     	add	x0, sp, #0x8
     9f8: aa1403e1     	mov	x1, x20
     9fc: 94000116     	bl	0xe54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>
     a00: 910022a0     	add	x0, x21, #0x8
     a04: aa1303e8     	mov	x8, x19
     a08: 940001dc     	bl	0x1178 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&>
     a0c: 90000033     	adrp	x19, 0x4000 <dyld_stub_binder+0x4000>
     a10: f9401673     	ldr	x19, [x19, #0x28]
     a14: f9400268     	ldr	x8, [x19]
     a18: f90007e8     	str	x8, [sp, #0x8]
     a1c: f9400e69     	ldr	x9, [x19, #0x18]
     a20: f85e8108     	ldur	x8, [x8, #-0x18]
     a24: 910023f4     	add	x20, sp, #0x8
     a28: f8286a89     	str	x9, [x20, x8]
     a2c: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     a30: f9401d08     	ldr	x8, [x8, #0x38]
     a34: 91004108     	add	x8, x8, #0x10
     a38: f9000be8     	str	x8, [sp, #0x10]
     a3c: 39c19fe8     	ldrsb	w8, [sp, #0x67]
     a40: 36f800a8     	tbz	w8, #0x1f, 0xa54 <c10::detail::_str_wrapper<char const*, char const*, char const*, char const*, int const&, char const*, char const*>::call(char const* const&, char const* const&, char const* const&, char const* const&, int const&, char const* const&, char const* const&)+0x154>
     a44: f9402be0     	ldr	x0, [sp, #0x50]
     a48: f94033e8     	ldr	x8, [sp, #0x60]
     a4c: 9240f901     	and	x1, x8, #0x7fffffffffffffff
     a50: 94000323     	bl	0x16dc <dyld_stub_binder+0x16dc>
     a54: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     a58: f9401908     	ldr	x8, [x8, #0x30]
     a5c: 91004108     	add	x8, x8, #0x10
     a60: f9000be8     	str	x8, [sp, #0x10]
     a64: 91004280     	add	x0, x20, #0x10
     a68: 9400030b     	bl	0x1694 <dyld_stub_binder+0x1694>
     a6c: 910023e0     	add	x0, sp, #0x8
     a70: 91002261     	add	x1, x19, #0x8
     a74: 940002ff     	bl	0x1670 <dyld_stub_binder+0x1670>
     a78: 9101c280     	add	x0, x20, #0x70
     a7c: 94000312     	bl	0x16c4 <dyld_stub_binder+0x16c4>
     a80: a9567bfd     	ldp	x29, x30, [sp, #0x160]
     a84: a9554ff4     	ldp	x20, x19, [sp, #0x150]
     a88: a95457f6     	ldp	x22, x21, [sp, #0x140]
     a8c: a9535ff8     	ldp	x24, x23, [sp, #0x130]
     a90: a95267fa     	ldp	x26, x25, [sp, #0x120]
     a94: a9516ffc     	ldp	x28, x27, [sp, #0x110]
     a98: 9105c3ff     	add	sp, sp, #0x170
     a9c: d65f03c0     	ret
     aa0: aa0003f3     	mov	x19, x0
     aa4: 910023e0     	add	x0, sp, #0x8
     aa8: 94000051     	bl	0xbec <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::~basic_ostringstream()>
     aac: aa1303e0     	mov	x0, x19
     ab0: 940002d5     	bl	0x1604 <dyld_stub_binder+0x1604>

0000000000000ab4 <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::basic_ostringstream[abi:nqe220108]()>:
     ab4: a9bc5ff8     	stp	x24, x23, [sp, #-0x40]!
     ab8: a90157f6     	stp	x22, x21, [sp, #0x10]
     abc: a9024ff4     	stp	x20, x19, [sp, #0x20]
     ac0: a9037bfd     	stp	x29, x30, [sp, #0x30]
     ac4: 9100c3fd     	add	x29, sp, #0x30
     ac8: aa0003f4     	mov	x20, x0
     acc: 90000038     	adrp	x24, 0x4000 <dyld_stub_binder+0x4000>
     ad0: f9402318     	ldr	x24, [x24, #0x40]
     ad4: 91010317     	add	x23, x24, #0x40
     ad8: aa0003f3     	mov	x19, x0
     adc: f8070e77     	str	x23, [x19, #0x70]!
     ae0: f900501f     	str	xzr, [x0, #0xa0]
     ae4: 90000036     	adrp	x22, 0x4000 <dyld_stub_binder+0x4000>
     ae8: f94016d6     	ldr	x22, [x22, #0x28]
     aec: a940a6c8     	ldp	x8, x9, [x22, #0x8]
     af0: f9000008     	str	x8, [x0]
     af4: f85e8108     	ldur	x8, [x8, #-0x18]
     af8: f8286809     	str	x9, [x0, x8]
     afc: f9400008     	ldr	x8, [x0]
     b00: f85e8108     	ldur	x8, [x8, #-0x18]
     b04: 8b080015     	add	x21, x0, x8
     b08: 91002001     	add	x1, x0, #0x8
     b0c: aa1503e0     	mov	x0, x21
     b10: 940002e7     	bl	0x16ac <dyld_stub_binder+0x16ac>
     b14: f90046bf     	str	xzr, [x21, #0x88]
     b18: 12800008     	mov	w8, #-0x1               ; =-1
     b1c: b90092a8     	str	w8, [x21, #0x90]
     b20: 91006308     	add	x8, x24, #0x18
     b24: f9003a97     	str	x23, [x20, #0x70]
     b28: 90000037     	adrp	x23, 0x4000 <dyld_stub_binder+0x4000>
     b2c: f9401af7     	ldr	x23, [x23, #0x30]
     b30: 910042e9     	add	x9, x23, #0x10
     b34: a9002688     	stp	x8, x9, [x20]
     b38: 91004280     	add	x0, x20, #0x10
     b3c: 940002d3     	bl	0x1688 <dyld_stub_binder+0x1688>
     b40: 6f00e400     	movi.2d	v0, #0000000000000000
     b44: 3c838280     	stur	q0, [x20, #0x38]
     b48: 3c828280     	stur	q0, [x20, #0x28]
     b4c: 3c818280     	stur	q0, [x20, #0x18]
     b50: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     b54: f9401d08     	ldr	x8, [x8, #0x38]
     b58: 91004108     	add	x8, x8, #0x10
     b5c: f9000688     	str	x8, [x20, #0x8]
     b60: 3c848280     	stur	q0, [x20, #0x48]
     b64: 3c858280     	stur	q0, [x20, #0x58]
     b68: 52800208     	mov	w8, #0x10               ; =16
     b6c: b9006a88     	str	w8, [x20, #0x68]
     b70: 91002280     	add	x0, x20, #0x8
     b74: 94000042     	bl	0xc7c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()>
     b78: aa1403e0     	mov	x0, x20
     b7c: a9437bfd     	ldp	x29, x30, [sp, #0x30]
     b80: a9424ff4     	ldp	x20, x19, [sp, #0x20]
     b84: a94157f6     	ldp	x22, x21, [sp, #0x10]
     b88: a8c45ff8     	ldp	x24, x23, [sp], #0x40
     b8c: d65f03c0     	ret
     b90: aa0003f5     	mov	x21, x0
     b94: 39c17e88     	ldrsb	w8, [x20, #0x5f]
     b98: 36f800a8     	tbz	w8, #0x1f, 0xbac <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::basic_ostringstream[abi:nqe220108]()+0xf8>
     b9c: f9402680     	ldr	x0, [x20, #0x48]
     ba0: f9402e88     	ldr	x8, [x20, #0x58]
     ba4: 9240f901     	and	x1, x8, #0x7fffffffffffffff
     ba8: 940002cd     	bl	0x16dc <dyld_stub_binder+0x16dc>
     bac: 910042e8     	add	x8, x23, #0x10
     bb0: f9000688     	str	x8, [x20, #0x8]
     bb4: 91004280     	add	x0, x20, #0x10
     bb8: 940002b7     	bl	0x1694 <dyld_stub_binder+0x1694>
     bbc: 910022c1     	add	x1, x22, #0x8
     bc0: aa1403e0     	mov	x0, x20
     bc4: 940002ab     	bl	0x1670 <dyld_stub_binder+0x1670>
     bc8: aa1303e0     	mov	x0, x19
     bcc: 940002be     	bl	0x16c4 <dyld_stub_binder+0x16c4>
     bd0: aa1503e0     	mov	x0, x21
     bd4: 9400028c     	bl	0x1604 <dyld_stub_binder+0x1604>
     bd8: aa0003f5     	mov	x21, x0
     bdc: aa1303e0     	mov	x0, x19
     be0: 940002b9     	bl	0x16c4 <dyld_stub_binder+0x16c4>
     be4: aa1503e0     	mov	x0, x21
     be8: 94000287     	bl	0x1604 <dyld_stub_binder+0x1604>

0000000000000bec <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::~basic_ostringstream()>:
     bec: a9be4ff4     	stp	x20, x19, [sp, #-0x20]!
     bf0: a9017bfd     	stp	x29, x30, [sp, #0x10]
     bf4: 910043fd     	add	x29, sp, #0x10
     bf8: aa0003f3     	mov	x19, x0
     bfc: 90000034     	adrp	x20, 0x4000 <dyld_stub_binder+0x4000>
     c00: f9401694     	ldr	x20, [x20, #0x28]
     c04: f9400288     	ldr	x8, [x20]
     c08: f9000008     	str	x8, [x0]
     c0c: f9400e89     	ldr	x9, [x20, #0x18]
     c10: f85e8108     	ldur	x8, [x8, #-0x18]
     c14: f8286809     	str	x9, [x0, x8]
     c18: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     c1c: f9401d08     	ldr	x8, [x8, #0x38]
     c20: 91004108     	add	x8, x8, #0x10
     c24: f9000408     	str	x8, [x0, #0x8]
     c28: 39c17c08     	ldrsb	w8, [x0, #0x5f]
     c2c: 36f800a8     	tbz	w8, #0x1f, 0xc40 <std::__1::basic_ostringstream<char, std::__1::char_traits<char>, std::__1::allocator<char>>::~basic_ostringstream()+0x54>
     c30: f9402660     	ldr	x0, [x19, #0x48]
     c34: f9402e68     	ldr	x8, [x19, #0x58]
     c38: 9240f901     	and	x1, x8, #0x7fffffffffffffff
     c3c: 940002a8     	bl	0x16dc <dyld_stub_binder+0x16dc>
     c40: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     c44: f9401908     	ldr	x8, [x8, #0x30]
     c48: 91004108     	add	x8, x8, #0x10
     c4c: f9000668     	str	x8, [x19, #0x8]
     c50: 91004260     	add	x0, x19, #0x10
     c54: 94000290     	bl	0x1694 <dyld_stub_binder+0x1694>
     c58: 91002281     	add	x1, x20, #0x8
     c5c: aa1303e0     	mov	x0, x19
     c60: 94000284     	bl	0x1670 <dyld_stub_binder+0x1670>
     c64: 9101c260     	add	x0, x19, #0x70
     c68: 94000297     	bl	0x16c4 <dyld_stub_binder+0x16c4>
     c6c: aa1303e0     	mov	x0, x19
     c70: a9417bfd     	ldp	x29, x30, [sp, #0x10]
     c74: a8c24ff4     	ldp	x20, x19, [sp], #0x20
     c78: d65f03c0     	ret

0000000000000c7c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()>:
     c7c: a9bd57f6     	stp	x22, x21, [sp, #-0x30]!
     c80: a9014ff4     	stp	x20, x19, [sp, #0x10]
     c84: a9027bfd     	stp	x29, x30, [sp, #0x20]
     c88: 910083fd     	add	x29, sp, #0x20
     c8c: aa0003e8     	mov	x8, x0
     c90: f8440d09     	ldr	x9, [x8, #0x40]!
     c94: f9000d1f     	str	xzr, [x8, #0x18]
     c98: 39405d0b     	ldrb	w11, [x8, #0x17]
     c9c: 13001d6a     	sxtb	w10, w11
     ca0: 7100015f     	cmp	w10, #0x0
     ca4: 9a884133     	csel	x19, x9, x8, mi
     ca8: f940050c     	ldr	x12, [x8, #0x8]
     cac: 92401d6b     	and	x11, x11, #0xff
     cb0: 9a8b4194     	csel	x20, x12, x11, mi
     cb4: b940210b     	ldr	w11, [x8, #0x20]
     cb8: 361800ab     	tbz	w11, #0x3, 0xccc <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x50>
     cbc: 8b14026c     	add	x12, x19, x20
     cc0: f9002c0c     	str	x12, [x0, #0x58]
     cc4: a9014c13     	stp	x19, x19, [x0, #0x10]
     cc8: f900100c     	str	x12, [x0, #0x20]
     ccc: 3620070b     	tbz	w11, #0x4, 0xdac <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x130>
     cd0: 8b14026b     	add	x11, x19, x20
     cd4: f9002c0b     	str	x11, [x0, #0x58]
     cd8: f940280b     	ldr	x11, [x0, #0x50]
     cdc: 9240f96b     	and	x11, x11, #0x7fffffffffffffff
     ce0: d100056c     	sub	x12, x11, #0x1
     ce4: 7100015f     	cmp	w10, #0x0
     ce8: 528002cb     	mov	w11, #0x16              ; =22
     cec: 9a8b418b     	csel	x11, x12, x11, mi
     cf0: eb140161     	subs	x1, x11, x20
     cf4: 540000e9     	b.ls	0xd10 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x94>
     cf8: aa0003f5     	mov	x21, x0
     cfc: aa0803e0     	mov	x0, x8
     d00: 52800002     	mov	w2, #0x0                ; =0
     d04: 94000252     	bl	0x164c <dyld_stub_binder+0x164c>
     d08: aa1503e0     	mov	x0, x21
     d0c: 14000008     	b	0xd2c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0xb0>
     d10: 37f8008a     	tbnz	w10, #0x1f, 0xd20 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0xa4>
     d14: 528002c9     	mov	w9, #0x16               ; =22
     d18: 39015c09     	strb	w9, [x0, #0x57]
     d1c: 14000003     	b	0xd28 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0xac>
     d20: f900240c     	str	x12, [x0, #0x48]
     d24: aa0903e8     	mov	x8, x9
     d28: 382b691f     	strb	wzr, [x8, x11]
     d2c: 39415c08     	ldrb	w8, [x0, #0x57]
     d30: f9402409     	ldr	x9, [x0, #0x48]
     d34: 7219011f     	tst	w8, #0x80
     d38: 9a881128     	csel	x8, x9, x8, ne
     d3c: 8b080268     	add	x8, x19, x8
     d40: a902cc13     	stp	x19, x19, [x0, #0x28]
     d44: f9001c08     	str	x8, [x0, #0x38]
     d48: 39418008     	ldrb	w8, [x0, #0x60]
     d4c: 7200051f     	tst	w8, #0x3
     d50: 540002e0     	b.eq	0xdac <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x130>
     d54: d35ffe88     	lsr	x8, x20, #31
     d58: b4000248     	cbz	x8, 0xda0 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x124>
     d5c: b26183e8     	mov	x8, #-0x80000000        ; =-2147483648
     d60: 8b080288     	add	x8, x20, x8
     d64: d28000a9     	mov	x9, #0x5                ; =5
     d68: f2c00049     	movk	x9, #0x2, lsl #32
     d6c: 9bc97d09     	umulh	x9, x8, x9
     d70: cb090108     	sub	x8, x8, x9
     d74: 8b480528     	add	x8, x9, x8, lsr #1
     d78: d35efd08     	lsr	x8, x8, #30
     d7c: d3618109     	lsl	x9, x8, #31
     d80: 12b0000a     	mov	w10, #0x7fffffff        ; =2147483647
     d84: cb080128     	sub	x8, x9, x8
     d88: cb080289     	sub	x9, x20, x8
     d8c: 8b0a026a     	add	x10, x19, x10
     d90: 8b080153     	add	x19, x10, x8
     d94: b26187e8     	mov	x8, #-0x7fffffff        ; =-2147483647
     d98: 8b080134     	add	x20, x9, x8
     d9c: f9001813     	str	x19, [x0, #0x30]
     da0: b4000074     	cbz	x20, 0xdac <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()+0x130>
     da4: 8b140268     	add	x8, x19, x20
     da8: f9001808     	str	x8, [x0, #0x30]
     dac: a9427bfd     	ldp	x29, x30, [sp, #0x20]
     db0: a9414ff4     	ldp	x20, x19, [sp, #0x10]
     db4: a8c357f6     	ldp	x22, x21, [sp], #0x30
     db8: d65f03c0     	ret

0000000000000dbc <___clang_call_terminate>:
     dbc: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     dc0: 910003fd     	mov	x29, sp
     dc4: 9400024f     	bl	0x1700 <dyld_stub_binder+0x1700>
     dc8: 94000242     	bl	0x16d0 <dyld_stub_binder+0x16d0>

0000000000000dcc <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__throw_length_error[abi:nqe220108]()>:
     dcc: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     dd0: 910003fd     	mov	x29, sp
     dd4: b0000000     	adrp	x0, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
     dd8: 912de400     	add	x0, x0, #0xb79
     ddc: 94000001     	bl	0xde0 <std::__1::__throw_length_error[abi:nqe220108](char const*)>

0000000000000de0 <std::__1::__throw_length_error[abi:nqe220108](char const*)>:
     de0: a9be4ff4     	stp	x20, x19, [sp, #-0x20]!
     de4: a9017bfd     	stp	x29, x30, [sp, #0x10]
     de8: 910043fd     	add	x29, sp, #0x10
     dec: aa0003f4     	mov	x20, x0
     df0: 52800200     	mov	w0, #0x10               ; =16
     df4: 94000240     	bl	0x16f4 <dyld_stub_binder+0x16f4>
     df8: aa0003f3     	mov	x19, x0
     dfc: aa1403e1     	mov	x1, x20
     e00: 9400000c     	bl	0xe30 <std::length_error::length_error[abi:nqe220108](char const*)>
     e04: 90000021     	adrp	x1, 0x4000 <dyld_stub_binder+0x4000>
     e08: f9403c21     	ldr	x1, [x1, #0x78]
     e0c: 90000022     	adrp	x2, 0x4000 <dyld_stub_binder+0x4000>
     e10: f9400042     	ldr	x2, [x2]
     e14: aa1303e0     	mov	x0, x19
     e18: 94000243     	bl	0x1724 <dyld_stub_binder+0x1724>
     e1c: aa0003f4     	mov	x20, x0
     e20: aa1303e0     	mov	x0, x19
     e24: 9400023d     	bl	0x1718 <dyld_stub_binder+0x1718>
     e28: aa1403e0     	mov	x0, x20
     e2c: 940001f6     	bl	0x1604 <dyld_stub_binder+0x1604>

0000000000000e30 <std::length_error::length_error[abi:nqe220108](char const*)>:
     e30: a9bf7bfd     	stp	x29, x30, [sp, #-0x10]!
     e34: 910003fd     	mov	x29, sp
     e38: 940001fc     	bl	0x1628 <dyld_stub_binder+0x1628>
     e3c: 90000028     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
     e40: f9402508     	ldr	x8, [x8, #0x48]
     e44: 91004108     	add	x8, x8, #0x10
     e48: f9000008     	str	x8, [x0]
     e4c: a8c17bfd     	ldp	x29, x30, [sp], #0x10
     e50: d65f03c0     	ret

0000000000000e54 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)>:
     e54: d101c3ff     	sub	sp, sp, #0x70
     e58: a90267fa     	stp	x26, x25, [sp, #0x20]
     e5c: a9035ff8     	stp	x24, x23, [sp, #0x30]
     e60: a90457f6     	stp	x22, x21, [sp, #0x40]
     e64: a9054ff4     	stp	x20, x19, [sp, #0x50]
     e68: a9067bfd     	stp	x29, x30, [sp, #0x60]
     e6c: 910183fd     	add	x29, sp, #0x60
     e70: aa0203f5     	mov	x21, x2
     e74: aa0103f4     	mov	x20, x1
     e78: aa0003f3     	mov	x19, x0
     e7c: 910023e0     	add	x0, sp, #0x8
     e80: aa1303e1     	mov	x1, x19
     e84: 940001f5     	bl	0x1658 <dyld_stub_binder+0x1658>
     e88: 394023e8     	ldrb	w8, [sp, #0x8]
     e8c: 7100051f     	cmp	w8, #0x1
     e90: 54000561     	b.ne	0xf3c <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0xe8>
     e94: f9400268     	ldr	x8, [x19]
     e98: f85e8108     	ldur	x8, [x8, #-0x18]
     e9c: 8b080264     	add	x4, x19, x8
     ea0: f9401496     	ldr	x22, [x4, #0x28]
     ea4: b9400898     	ldr	w24, [x4, #0x8]
     ea8: b9409097     	ldr	w23, [x4, #0x90]
     eac: 310006ff     	cmn	w23, #0x1
     eb0: 54000241     	b.ne	0xef8 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0xa4>
     eb4: 910063e8     	add	x8, sp, #0x18
     eb8: aa0403f9     	mov	x25, x4
     ebc: aa0403e0     	mov	x0, x4
     ec0: 940001d7     	bl	0x161c <dyld_stub_binder+0x161c>
     ec4: 90000021     	adrp	x1, 0x4000 <dyld_stub_binder+0x4000>
     ec8: f9400821     	ldr	x1, [x1, #0x10]
     ecc: 910063e0     	add	x0, sp, #0x18
     ed0: 940001d0     	bl	0x1610 <dyld_stub_binder+0x1610>
     ed4: f9400008     	ldr	x8, [x0]
     ed8: f9401d08     	ldr	x8, [x8, #0x38]
     edc: 52800401     	mov	w1, #0x20               ; =32
     ee0: d63f0100     	blr	x8
     ee4: aa0003f7     	mov	x23, x0
     ee8: 910063e0     	add	x0, sp, #0x18
     eec: 940001ea     	bl	0x1694 <dyld_stub_binder+0x1694>
     ef0: aa1903e4     	mov	x4, x25
     ef4: b9009337     	str	w23, [x25, #0x90]
     ef8: 52801608     	mov	w8, #0xb0               ; =176
     efc: 0a080308     	and	w8, w24, w8
     f00: 8b150283     	add	x3, x20, x21
     f04: 7100811f     	cmp	w8, #0x20
     f08: 9a940062     	csel	x2, x3, x20, eq
     f0c: 13001ee5     	sxtb	w5, w23
     f10: aa1603e0     	mov	x0, x22
     f14: aa1403e1     	mov	x1, x20
     f18: 9400002a     	bl	0xfc0 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)>
     f1c: b5000100     	cbnz	x0, 0xf3c <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0xe8>
     f20: f9400268     	ldr	x8, [x19]
     f24: f85e8108     	ldur	x8, [x8, #-0x18]
     f28: 8b080260     	add	x0, x19, x8
     f2c: b9402008     	ldr	w8, [x0, #0x20]
     f30: 528000a9     	mov	w9, #0x5                ; =5
     f34: 2a090101     	orr	w1, w8, w9
     f38: 940001e0     	bl	0x16b8 <dyld_stub_binder+0x16b8>
     f3c: 910023e0     	add	x0, sp, #0x8
     f40: 940001c9     	bl	0x1664 <dyld_stub_binder+0x1664>
     f44: aa1303e0     	mov	x0, x19
     f48: a9467bfd     	ldp	x29, x30, [sp, #0x60]
     f4c: a9454ff4     	ldp	x20, x19, [sp, #0x50]
     f50: a94457f6     	ldp	x22, x21, [sp, #0x40]
     f54: a9435ff8     	ldp	x24, x23, [sp, #0x30]
     f58: a94267fa     	ldp	x26, x25, [sp, #0x20]
     f5c: 9101c3ff     	add	sp, sp, #0x70
     f60: d65f03c0     	ret
     f64: 14000005     	b	0xf78 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0x124>
     f68: aa0003f4     	mov	x20, x0
     f6c: 910063e0     	add	x0, sp, #0x18
     f70: 940001c9     	bl	0x1694 <dyld_stub_binder+0x1694>
     f74: 14000002     	b	0xf7c <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0x128>
     f78: aa0003f4     	mov	x20, x0
     f7c: 910023e0     	add	x0, sp, #0x8
     f80: 940001b9     	bl	0x1664 <dyld_stub_binder+0x1664>
     f84: 14000002     	b	0xf8c <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0x138>
     f88: aa0003f4     	mov	x20, x0
     f8c: aa1403e0     	mov	x0, x20
     f90: 940001dc     	bl	0x1700 <dyld_stub_binder+0x1700>
     f94: f9400268     	ldr	x8, [x19]
     f98: f85e8108     	ldur	x8, [x8, #-0x18]
     f9c: 8b080260     	add	x0, x19, x8
     fa0: 940001c0     	bl	0x16a0 <dyld_stub_binder+0x16a0>
     fa4: 940001da     	bl	0x170c <dyld_stub_binder+0x170c>
     fa8: 17ffffe7     	b	0xf44 <std::__1::basic_ostream<char, std::__1::char_traits<char>>& std::__1::__put_character_sequence[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::basic_ostream<char, std::__1::char_traits<char>>&, char const*, unsigned long)+0xf0>
     fac: aa0003f3     	mov	x19, x0
     fb0: 940001d7     	bl	0x170c <dyld_stub_binder+0x170c>
     fb4: aa1303e0     	mov	x0, x19
     fb8: 94000193     	bl	0x1604 <dyld_stub_binder+0x1604>
     fbc: 97ffff80     	bl	0xdbc <___clang_call_terminate>

0000000000000fc0 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)>:
     fc0: d101c3ff     	sub	sp, sp, #0x70
     fc4: a90267fa     	stp	x26, x25, [sp, #0x20]
     fc8: a9035ff8     	stp	x24, x23, [sp, #0x30]
     fcc: a90457f6     	stp	x22, x21, [sp, #0x40]
     fd0: a9054ff4     	stp	x20, x19, [sp, #0x50]
     fd4: a9067bfd     	stp	x29, x30, [sp, #0x60]
     fd8: 910183fd     	add	x29, sp, #0x60
     fdc: aa0003f3     	mov	x19, x0
     fe0: b4000a80     	cbz	x0, 0x1130 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x170>
     fe4: aa0503f8     	mov	x24, x5
     fe8: aa0403f4     	mov	x20, x4
     fec: aa0303f6     	mov	x22, x3
     ff0: aa0203f5     	mov	x21, x2
     ff4: aa0103f7     	mov	x23, x1
     ff8: f9400c99     	ldr	x25, [x4, #0x18]
     ffc: cb01005a     	sub	x26, x2, x1
    1000: f100075f     	cmp	x26, #0x1
    1004: 5400012b     	b.lt	0x1028 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x68>
    1008: f9400268     	ldr	x8, [x19]
    100c: f9403108     	ldr	x8, [x8, #0x60]
    1010: cb1702a2     	sub	x2, x21, x23
    1014: aa1303e0     	mov	x0, x19
    1018: aa1703e1     	mov	x1, x23
    101c: d63f0100     	blr	x8
    1020: eb1a001f     	cmp	x0, x26
    1024: 54000841     	b.ne	0x112c <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x16c>
    1028: cb1702c8     	sub	x8, x22, x23
    102c: eb08033f     	cmp	x25, x8
    1030: 5400064d     	b.le	0x10f8 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x138>
    1034: 92800109     	mov	x9, #-0x9               ; =-9
    1038: f2efffe9     	movk	x9, #0x7fff, lsl #48
    103c: cb080337     	sub	x23, x25, x8
    1040: eb0902ff     	cmp	x23, x9
    1044: 54000862     	b.hs	0x1150 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x190>
    1048: f1005eff     	cmp	x23, #0x17
    104c: 54000082     	b.hs	0x105c <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x9c>
    1050: 39007ff7     	strb	w23, [sp, #0x1f]
    1054: 910023f9     	add	x25, sp, #0x8
    1058: 1400000c     	b	0x1088 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0xc8>
    105c: 927deee8     	and	x8, x23, #0x7ffffffffffffff8
    1060: 91002108     	add	x8, x8, #0x8
    1064: 52800329     	mov	w9, #0x19               ; =25
    1068: f100611f     	cmp	x8, #0x18
    106c: 9a88013a     	csel	x26, x9, x8, eq
    1070: aa1a03e0     	mov	x0, x26
    1074: 9400019d     	bl	0x16e8 <dyld_stub_binder+0x16e8>
    1078: aa0003f9     	mov	x25, x0
    107c: b2410348     	orr	x8, x26, #0x8000000000000000
    1080: a900dfe0     	stp	x0, x23, [sp, #0x8]
    1084: f9000fe8     	str	x8, [sp, #0x18]
    1088: aa1903e0     	mov	x0, x25
    108c: aa1803e1     	mov	x1, x24
    1090: aa1703e2     	mov	x2, x23
    1094: 940001b0     	bl	0x1754 <dyld_stub_binder+0x1754>
    1098: 38376b3f     	strb	wzr, [x25, x23]
    109c: 39c07fe8     	ldrsb	w8, [sp, #0x1f]
    10a0: f94007e9     	ldr	x9, [sp, #0x8]
    10a4: 7100011f     	cmp	w8, #0x0
    10a8: 910023e8     	add	x8, sp, #0x8
    10ac: 9a884121     	csel	x1, x9, x8, mi
    10b0: f9400268     	ldr	x8, [x19]
    10b4: f9403108     	ldr	x8, [x8, #0x60]
    10b8: aa1303e0     	mov	x0, x19
    10bc: aa1703e2     	mov	x2, x23
    10c0: d63f0100     	blr	x8
    10c4: 39c07fe8     	ldrsb	w8, [sp, #0x1f]
    10c8: 37f80088     	tbnz	w8, #0x1f, 0x10d8 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x118>
    10cc: eb17001f     	cmp	x0, x23
    10d0: 540002e1     	b.ne	0x112c <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x16c>
    10d4: 14000009     	b	0x10f8 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x138>
    10d8: f94007e8     	ldr	x8, [sp, #0x8]
    10dc: f9400fe9     	ldr	x9, [sp, #0x18]
    10e0: 9240f921     	and	x1, x9, #0x7fffffffffffffff
    10e4: aa0003f8     	mov	x24, x0
    10e8: aa0803e0     	mov	x0, x8
    10ec: 9400017c     	bl	0x16dc <dyld_stub_binder+0x16dc>
    10f0: eb17031f     	cmp	x24, x23
    10f4: 540001c1     	b.ne	0x112c <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x16c>
    10f8: cb1502d6     	sub	x22, x22, x21
    10fc: f10006df     	cmp	x22, #0x1
    1100: 5400012b     	b.lt	0x1124 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x164>
    1104: f9400268     	ldr	x8, [x19]
    1108: f9403108     	ldr	x8, [x8, #0x60]
    110c: aa1303e0     	mov	x0, x19
    1110: aa1503e1     	mov	x1, x21
    1114: aa1603e2     	mov	x2, x22
    1118: d63f0100     	blr	x8
    111c: eb16001f     	cmp	x0, x22
    1120: 54000061     	b.ne	0x112c <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x16c>
    1124: f9000e9f     	str	xzr, [x20, #0x18]
    1128: 14000002     	b	0x1130 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x170>
    112c: d2800013     	mov	x19, #0x0               ; =0
    1130: aa1303e0     	mov	x0, x19
    1134: a9467bfd     	ldp	x29, x30, [sp, #0x60]
    1138: a9454ff4     	ldp	x20, x19, [sp, #0x50]
    113c: a94457f6     	ldp	x22, x21, [sp, #0x40]
    1140: a9435ff8     	ldp	x24, x23, [sp, #0x30]
    1144: a94267fa     	ldp	x26, x25, [sp, #0x20]
    1148: 9101c3ff     	add	sp, sp, #0x70
    114c: d65f03c0     	ret
    1150: 97ffff1f     	bl	0xdcc <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__throw_length_error[abi:nqe220108]()>
    1154: aa0003f3     	mov	x19, x0
    1158: 39c07fe8     	ldrsb	w8, [sp, #0x1f]
    115c: 36f800a8     	tbz	w8, #0x1f, 0x1170 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x1b0>
    1160: f94007e0     	ldr	x0, [sp, #0x8]
    1164: f9400fe8     	ldr	x8, [sp, #0x18]
    1168: 9240f901     	and	x1, x8, #0x7fffffffffffffff
    116c: 9400015c     	bl	0x16dc <dyld_stub_binder+0x16dc>
    1170: aa1303e0     	mov	x0, x19
    1174: 94000124     	bl	0x1604 <dyld_stub_binder+0x1604>

0000000000001178 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&>:
    1178: a9bd57f6     	stp	x22, x21, [sp, #-0x30]!
    117c: a9014ff4     	stp	x20, x19, [sp, #0x10]
    1180: a9027bfd     	stp	x29, x30, [sp, #0x20]
    1184: 910083fd     	add	x29, sp, #0x20
    1188: aa0003f4     	mov	x20, x0
    118c: aa0803f3     	mov	x19, x8
    1190: b9406008     	ldr	w8, [x0, #0x60]
    1194: 37200088     	tbnz	w8, #0x4, 0x11a4 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x2c>
    1198: 37180288     	tbnz	w8, #0x3, 0x11e8 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x70>
    119c: d2800008     	mov	x8, #0x0                ; =0
    11a0: 14000017     	b	0x11fc <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x84>
    11a4: f9402e88     	ldr	x8, [x20, #0x58]
    11a8: f9401a89     	ldr	x9, [x20, #0x30]
    11ac: eb09011f     	cmp	x8, x9
    11b0: 54000062     	b.hs	0x11bc <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x44>
    11b4: f9002e89     	str	x9, [x20, #0x58]
    11b8: aa0903e8     	mov	x8, x9
    11bc: 9100a289     	add	x9, x20, #0x28
    11c0: f9400129     	ldr	x9, [x9]
    11c4: eb090108     	subs	x8, x8, x9
    11c8: 540001a0     	b.eq	0x11fc <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x84>
    11cc: aa1403ea     	mov	x10, x20
    11d0: f8440d4b     	ldr	x11, [x10, #0x40]!
    11d4: 39c05d4c     	ldrsb	w12, [x10, #0x17]
    11d8: 7100019f     	cmp	w12, #0x0
    11dc: 9a8a416a     	csel	x10, x11, x10, mi
    11e0: cb0a0135     	sub	x21, x9, x10
    11e4: 14000007     	b	0x1200 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x88>
    11e8: 91004289     	add	x9, x20, #0x10
    11ec: f9401288     	ldr	x8, [x20, #0x20]
    11f0: f9400129     	ldr	x9, [x9]
    11f4: eb090108     	subs	x8, x8, x9
    11f8: 54fffea1     	b.ne	0x11cc <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x54>
    11fc: d2800015     	mov	x21, #0x0               ; =0
    1200: f9402a89     	ldr	x9, [x20, #0x50]
    1204: f9000a69     	str	x9, [x19, #0x10]
    1208: 3dc01280     	ldr	q0, [x20, #0x40]
    120c: 3d800260     	str	q0, [x19]
    1210: a904fe9f     	stp	xzr, xzr, [x20, #0x48]
    1214: f900229f     	str	xzr, [x20, #0x40]
    1218: d378fd2a     	lsr	x10, x9, #56
    121c: 13001d49     	sxtb	w9, w10
    1220: f940066b     	ldr	x11, [x19, #0x8]
    1224: 7100013f     	cmp	w9, #0x0
    1228: 9a8a416a     	csel	x10, x11, x10, mi
    122c: 8b0802a8     	add	x8, x21, x8
    1230: eb0a0101     	subs	x1, x8, x10
    1234: 54000169     	b.ls	0x1260 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0xe8>
    1238: aa1303e0     	mov	x0, x19
    123c: 52800002     	mov	w2, #0x0                ; =0
    1240: 94000103     	bl	0x164c <dyld_stub_binder+0x164c>
    1244: b10006bf     	cmn	x21, #0x1
    1248: 54000180     	b.eq	0x1278 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x100>
    124c: aa1303e0     	mov	x0, x19
    1250: d2800001     	mov	x1, #0x0                ; =0
    1254: aa1503e2     	mov	x2, x21
    1258: 94000025     	bl	0x12ec <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)>
    125c: 14000015     	b	0x12b0 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x138>
    1260: 37f80169     	tbnz	w9, #0x1f, 0x128c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x114>
    1264: 12001909     	and	w9, w8, #0x7f
    1268: 39005e69     	strb	w9, [x19, #0x17]
    126c: 38286a7f     	strb	wzr, [x19, x8]
    1270: b10006bf     	cmn	x21, #0x1
    1274: 54fffec1     	b.ne	0x124c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0xd4>
    1278: 39c05e68     	ldrsb	w8, [x19, #0x17]
    127c: 37f80148     	tbnz	w8, #0x1f, 0x12a4 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x12c>
    1280: 39005e7f     	strb	wzr, [x19, #0x17]
    1284: 3900027f     	strb	wzr, [x19]
    1288: 1400000a     	b	0x12b0 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x138>
    128c: f9400269     	ldr	x9, [x19]
    1290: f9000668     	str	x8, [x19, #0x8]
    1294: 3828693f     	strb	wzr, [x9, x8]
    1298: b10006bf     	cmn	x21, #0x1
    129c: 54fffd81     	b.ne	0x124c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0xd4>
    12a0: 17fffff6     	b	0x1278 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x100>
    12a4: f9400268     	ldr	x8, [x19]
    12a8: f900067f     	str	xzr, [x19, #0x8]
    12ac: 3900011f     	strb	wzr, [x8]
    12b0: aa1403e0     	mov	x0, x20
    12b4: 97fffe72     	bl	0xc7c <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs[abi:nqe220108]()>
    12b8: a9427bfd     	ldp	x29, x30, [sp, #0x20]
    12bc: a9414ff4     	ldp	x20, x19, [sp, #0x10]
    12c0: a8c357f6     	ldp	x22, x21, [sp], #0x30
    12c4: d65f03c0     	ret
    12c8: aa0003f4     	mov	x20, x0
    12cc: 39c05e68     	ldrsb	w8, [x19, #0x17]
    12d0: 36f800a8     	tbz	w8, #0x1f, 0x12e4 <std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::str[abi:nqe220108]() &&+0x16c>
    12d4: f9400260     	ldr	x0, [x19]
    12d8: f9400a68     	ldr	x8, [x19, #0x10]
    12dc: 9240f901     	and	x1, x8, #0x7fffffffffffffff
    12e0: 940000ff     	bl	0x16dc <dyld_stub_binder+0x16dc>
    12e4: aa1403e0     	mov	x0, x20
    12e8: 940000c7     	bl	0x1604 <dyld_stub_binder+0x1604>

00000000000012ec <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)>:
    12ec: b4000422     	cbz	x2, 0x1370 <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)+0x84>
    12f0: a9bd57f6     	stp	x22, x21, [sp, #-0x30]!
    12f4: a9014ff4     	stp	x20, x19, [sp, #0x10]
    12f8: a9027bfd     	stp	x29, x30, [sp, #0x20]
    12fc: 910083fd     	add	x29, sp, #0x20
    1300: 39405c09     	ldrb	w9, [x0, #0x17]
    1304: 13001d28     	sxtb	w8, w9
    1308: 7100011f     	cmp	w8, #0x0
    130c: a940280b     	ldp	x11, x10, [x0]
    1310: 9a894154     	csel	x20, x10, x9, mi
    1314: 9a804173     	csel	x19, x11, x0, mi
    1318: cb010289     	sub	x9, x20, x1
    131c: eb02013f     	cmp	x9, x2
    1320: 9a823135     	csel	x21, x9, x2, lo
    1324: 54000129     	b.ls	0x1348 <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)+0x5c>
    1328: cb150122     	sub	x2, x9, x21
    132c: 8b010268     	add	x8, x19, x1
    1330: 8b150101     	add	x1, x8, x21
    1334: aa0003f6     	mov	x22, x0
    1338: aa0803e0     	mov	x0, x8
    133c: 94000103     	bl	0x1748 <dyld_stub_binder+0x1748>
    1340: aa1603e0     	mov	x0, x22
    1344: 39405ec8     	ldrb	w8, [x22, #0x17]
    1348: cb150289     	sub	x9, x20, x21
    134c: 37380088     	tbnz	w8, #0x7, 0x135c <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)+0x70>
    1350: 12001928     	and	w8, w9, #0x7f
    1354: 39005c08     	strb	w8, [x0, #0x17]
    1358: 14000002     	b	0x1360 <std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__erase_external_with_move(unsigned long, unsigned long)+0x74>
    135c: f9000409     	str	x9, [x0, #0x8]
    1360: 38296a7f     	strb	wzr, [x19, x9]
    1364: a9427bfd     	ldp	x29, x30, [sp, #0x20]
    1368: a9414ff4     	ldp	x20, x19, [sp, #0x10]
    136c: a8c357f6     	ldp	x22, x21, [sp], #0x30
    1370: d65f03c0     	ret

0000000000001374 <_PyInit_kernel>:
    1374: d100c3ff     	sub	sp, sp, #0x30
    1378: a9014ff4     	stp	x20, x19, [sp, #0x10]
    137c: a9027bfd     	stp	x29, x30, [sp, #0x20]
    1380: 910083fd     	add	x29, sp, #0x20
    1384: 90000000     	adrp	x0, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    1388: 912e1800     	add	x0, x0, #0xb86
    138c: 940000ec     	bl	0x173c <dyld_stub_binder+0x173c>
    1390: b4000300     	cbz	x0, 0x13f0 <_PyInit_kernel+0x7c>
    1394: aa0003f3     	mov	x19, x0
    1398: f90007ff     	str	xzr, [sp, #0x8]
    139c: 940000e5     	bl	0x1730 <dyld_stub_binder+0x1730>
    13a0: b900001f     	str	wzr, [x0]
    13a4: 910023e1     	add	x1, sp, #0x8
    13a8: aa1303e0     	mov	x0, x19
    13ac: 52800142     	mov	w2, #0xa                ; =10
    13b0: 940000f2     	bl	0x1778 <dyld_stub_binder+0x1778>
    13b4: aa0003f4     	mov	x20, x0
    13b8: 940000de     	bl	0x1730 <dyld_stub_binder+0x1730>
    13bc: b9400008     	ldr	w8, [x0]
    13c0: 340002e8     	cbz	w8, 0x141c <_PyInit_kernel+0xa8>
    13c4: f0000008     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
    13c8: f9403108     	ldr	x8, [x8, #0x60]
    13cc: f9400100     	ldr	x0, [x8]
    13d0: 90000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    13d4: 912f8821     	add	x1, x1, #0xbe2
    13d8: 94000085     	bl	0x15ec <dyld_stub_binder+0x15ec>
    13dc: d2800000     	mov	x0, #0x0                ; =0
    13e0: a9427bfd     	ldp	x29, x30, [sp, #0x20]
    13e4: a9414ff4     	ldp	x20, x19, [sp, #0x10]
    13e8: 9100c3ff     	add	sp, sp, #0x30
    13ec: d65f03c0     	ret
    13f0: f0000008     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
    13f4: f9403108     	ldr	x8, [x8, #0x60]
    13f8: f9400100     	ldr	x0, [x8]
    13fc: 90000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    1400: 912eb821     	add	x1, x1, #0xbae
    1404: 9400007a     	bl	0x15ec <dyld_stub_binder+0x15ec>
    1408: d2800000     	mov	x0, #0x0                ; =0
    140c: a9427bfd     	ldp	x29, x30, [sp, #0x20]
    1410: a9414ff4     	ldp	x20, x19, [sp, #0x10]
    1414: 9100c3ff     	add	sp, sp, #0x30
    1418: d65f03c0     	ret
    141c: f94007e8     	ldr	x8, [sp, #0x8]
    1420: eb13011f     	cmp	x8, x19
    1424: 54fffd00     	b.eq	0x13c4 <_PyInit_kernel+0x50>
    1428: b4fffcf4     	cbz	x20, 0x13c4 <_PyInit_kernel+0x50>
    142c: f0000028     	adrp	x8, 0x8000 <dyld_stub_binder+0x8000>
    1430: f900dd14     	str	x20, [x8, #0x1b8]
    1434: f0000020     	adrp	x0, 0x8000 <dyld_stub_binder+0x8000>
    1438: 91042000     	add	x0, x0, #0x108
    143c: 52807ea1     	mov	w1, #0x3f5              ; =1013
    1440: 9400006e     	bl	0x15f8 <dyld_stub_binder+0x15f8>
    1444: a9427bfd     	ldp	x29, x30, [sp, #0x20]
    1448: a9414ff4     	ldp	x20, x19, [sp, #0x10]
    144c: 9100c3ff     	add	sp, sp, #0x30
    1450: d65f03c0     	ret

0000000000001454 <kernel_py(_object*, _object*)>:
    1454: a9bc5ff8     	stp	x24, x23, [sp, #-0x40]!
    1458: a90157f6     	stp	x22, x21, [sp, #0x10]
    145c: a9024ff4     	stp	x20, x19, [sp, #0x20]
    1460: a9037bfd     	stp	x29, x30, [sp, #0x30]
    1464: 9100c3fd     	add	x29, sp, #0x30
    1468: f9400428     	ldr	x8, [x1, #0x8]
    146c: f0000009     	adrp	x9, 0x4000 <dyld_stub_binder+0x4000>
    1470: f9403529     	ldr	x9, [x9, #0x68]
    1474: eb09011f     	cmp	x8, x9
    1478: 54000421     	b.ne	0x14fc <kernel_py(_object*, _object*)+0xa8>
    147c: aa0103f3     	mov	x19, x1
    1480: f9400828     	ldr	x8, [x1, #0x10]
    1484: f100111f     	cmp	x8, #0x4
    1488: 54000481     	b.ne	0x1518 <kernel_py(_object*, _object*)+0xc4>
    148c: f0000037     	adrp	x23, 0x8000 <dyld_stub_binder+0x8000>
    1490: f940dee8     	ldr	x8, [x23, #0x1b8]
    1494: f9400e60     	ldr	x0, [x19, #0x18]
    1498: d63f0100     	blr	x8
    149c: aa0003f4     	mov	x20, x0
    14a0: f940dee8     	ldr	x8, [x23, #0x1b8]
    14a4: f9401260     	ldr	x0, [x19, #0x20]
    14a8: d63f0100     	blr	x8
    14ac: aa0003f5     	mov	x21, x0
    14b0: f940dee8     	ldr	x8, [x23, #0x1b8]
    14b4: f9401660     	ldr	x0, [x19, #0x28]
    14b8: d63f0100     	blr	x8
    14bc: aa0003f6     	mov	x22, x0
    14c0: f940dee8     	ldr	x8, [x23, #0x1b8]
    14c4: f9401a60     	ldr	x0, [x19, #0x30]
    14c8: d63f0100     	blr	x8
    14cc: aa0003e3     	mov	x3, x0
    14d0: aa1403e0     	mov	x0, x20
    14d4: aa1503e1     	mov	x1, x21
    14d8: aa1603e2     	mov	x2, x22
    14dc: 97fffc71     	bl	0x6a0 <_kernel>
    14e0: f0000000     	adrp	x0, 0x4000 <dyld_stub_binder+0x4000>
    14e4: f9403800     	ldr	x0, [x0, #0x70]
    14e8: a9437bfd     	ldp	x29, x30, [sp, #0x30]
    14ec: a9424ff4     	ldp	x20, x19, [sp, #0x20]
    14f0: a94157f6     	ldp	x22, x21, [sp, #0x10]
    14f4: a8c45ff8     	ldp	x24, x23, [sp], #0x40
    14f8: d65f03c0     	ret
    14fc: 52800200     	mov	w0, #0x10               ; =16
    1500: 9400007d     	bl	0x16f4 <dyld_stub_binder+0x16f4>
    1504: aa0003f3     	mov	x19, x0
    1508: 90000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    150c: 91308821     	add	x1, x1, #0xc22
    1510: 94000049     	bl	0x1634 <dyld_stub_binder+0x1634>
    1514: 14000007     	b	0x1530 <kernel_py(_object*, _object*)+0xdc>
    1518: 52800200     	mov	w0, #0x10               ; =16
    151c: 94000076     	bl	0x16f4 <dyld_stub_binder+0x16f4>
    1520: aa0003f3     	mov	x19, x0
    1524: 90000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    1528: 9130d821     	add	x1, x1, #0xc36
    152c: 94000042     	bl	0x1634 <dyld_stub_binder+0x1634>
    1530: f0000001     	adrp	x1, 0x4000 <dyld_stub_binder+0x4000>
    1534: f9400c21     	ldr	x1, [x1, #0x18]
    1538: f0000002     	adrp	x2, 0x4000 <dyld_stub_binder+0x4000>
    153c: f9400442     	ldr	x2, [x2, #0x8]
    1540: aa1303e0     	mov	x0, x19
    1544: 94000078     	bl	0x1724 <dyld_stub_binder+0x1724>
    1548: d4200020     	brk	#0x1
    154c: 14000001     	b	0x1550 <kernel_py(_object*, _object*)+0xfc>
    1550: aa0103f4     	mov	x20, x1
    1554: aa0003f5     	mov	x21, x0
    1558: aa1303e0     	mov	x0, x19
    155c: 9400006f     	bl	0x1718 <dyld_stub_binder+0x1718>
    1560: aa1503e0     	mov	x0, x21
    1564: 14000002     	b	0x156c <kernel_py(_object*, _object*)+0x118>
    1568: aa0103f4     	mov	x20, x1
    156c: 94000065     	bl	0x1700 <dyld_stub_binder+0x1700>
    1570: f0000008     	adrp	x8, 0x4000 <dyld_stub_binder+0x4000>
    1574: f9403108     	ldr	x8, [x8, #0x60]
    1578: f9400113     	ldr	x19, [x8]
    157c: 71000a9f     	cmp	w20, #0x2
    1580: 54000101     	b.ne	0x15a0 <kernel_py(_object*, _object*)+0x14c>
    1584: f9400008     	ldr	x8, [x0]
    1588: f9400908     	ldr	x8, [x8, #0x10]
    158c: d63f0100     	blr	x8
    1590: aa0003e1     	mov	x1, x0
    1594: aa1303e0     	mov	x0, x19
    1598: 94000015     	bl	0x15ec <dyld_stub_binder+0x15ec>
    159c: 14000005     	b	0x15b0 <kernel_py(_object*, _object*)+0x15c>
    15a0: 90000001     	adrp	x1, 0x1000 <std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>> std::__1::__pad_and_output[abi:nqe220108]<char, std::__1::char_traits<char>>(std::__1::ostreambuf_iterator<char, std::__1::char_traits<char>>, char const*, char const*, char const*, std::__1::ios_base&, char)+0x40>
    15a4: 91311821     	add	x1, x1, #0xc46
    15a8: aa1303e0     	mov	x0, x19
    15ac: 94000010     	bl	0x15ec <dyld_stub_binder+0x15ec>
    15b0: 94000057     	bl	0x170c <dyld_stub_binder+0x170c>
    15b4: d2800000     	mov	x0, #0x0                ; =0
    15b8: a9437bfd     	ldp	x29, x30, [sp, #0x30]
    15bc: a9424ff4     	ldp	x20, x19, [sp, #0x20]
    15c0: a94157f6     	ldp	x22, x21, [sp, #0x10]
    15c4: a8c45ff8     	ldp	x24, x23, [sp], #0x40
    15c8: d65f03c0     	ret
    15cc: aa0003f3     	mov	x19, x0
    15d0: 9400004f     	bl	0x170c <dyld_stub_binder+0x170c>
    15d4: 14000003     	b	0x15e0 <kernel_py(_object*, _object*)+0x18c>
    15d8: aa0003f3     	mov	x19, x0
    15dc: 9400004c     	bl	0x170c <dyld_stub_binder+0x170c>
    15e0: aa1303e0     	mov	x0, x19
    15e4: 94000008     	bl	0x1604 <dyld_stub_binder+0x1604>
    15e8: 97fffdf5     	bl	0xdbc <___clang_call_terminate>

Disassembly of section __TEXT,__stubs:

00000000000015ec <__stubs>:
    15ec: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    15f0: f9400210     	ldr	x16, [x16]
    15f4: d61f0200     	br	x16
    15f8: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    15fc: f9400610     	ldr	x16, [x16, #0x8]
    1600: d61f0200     	br	x16
    1604: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1608: f9400a10     	ldr	x16, [x16, #0x10]
    160c: d61f0200     	br	x16
    1610: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1614: f9400e10     	ldr	x16, [x16, #0x18]
    1618: d61f0200     	br	x16
    161c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1620: f9401210     	ldr	x16, [x16, #0x20]
    1624: d61f0200     	br	x16
    1628: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    162c: f9401610     	ldr	x16, [x16, #0x28]
    1630: d61f0200     	br	x16
    1634: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1638: f9401a10     	ldr	x16, [x16, #0x30]
    163c: d61f0200     	br	x16
    1640: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1644: f9401e10     	ldr	x16, [x16, #0x38]
    1648: d61f0200     	br	x16
    164c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1650: f9402210     	ldr	x16, [x16, #0x40]
    1654: d61f0200     	br	x16
    1658: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    165c: f9402610     	ldr	x16, [x16, #0x48]
    1660: d61f0200     	br	x16
    1664: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1668: f9402a10     	ldr	x16, [x16, #0x50]
    166c: d61f0200     	br	x16
    1670: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1674: f9402e10     	ldr	x16, [x16, #0x58]
    1678: d61f0200     	br	x16
    167c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1680: f9403210     	ldr	x16, [x16, #0x60]
    1684: d61f0200     	br	x16
    1688: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    168c: f9403610     	ldr	x16, [x16, #0x68]
    1690: d61f0200     	br	x16
    1694: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1698: f9403a10     	ldr	x16, [x16, #0x70]
    169c: d61f0200     	br	x16
    16a0: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16a4: f9403e10     	ldr	x16, [x16, #0x78]
    16a8: d61f0200     	br	x16
    16ac: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16b0: f9404210     	ldr	x16, [x16, #0x80]
    16b4: d61f0200     	br	x16
    16b8: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16bc: f9404610     	ldr	x16, [x16, #0x88]
    16c0: d61f0200     	br	x16
    16c4: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16c8: f9404a10     	ldr	x16, [x16, #0x90]
    16cc: d61f0200     	br	x16
    16d0: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16d4: f9404e10     	ldr	x16, [x16, #0x98]
    16d8: d61f0200     	br	x16
    16dc: f0000010     	adrp	x16, 0x4000 <dyld_stub_binder+0x4000>
    16e0: f9404210     	ldr	x16, [x16, #0x80]
    16e4: d61f0200     	br	x16
    16e8: f0000010     	adrp	x16, 0x4000 <dyld_stub_binder+0x4000>
    16ec: f9404610     	ldr	x16, [x16, #0x88]
    16f0: d61f0200     	br	x16
    16f4: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    16f8: f9405210     	ldr	x16, [x16, #0xa0]
    16fc: d61f0200     	br	x16
    1700: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1704: f9405610     	ldr	x16, [x16, #0xa8]
    1708: d61f0200     	br	x16
    170c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1710: f9405a10     	ldr	x16, [x16, #0xb0]
    1714: d61f0200     	br	x16
    1718: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    171c: f9405e10     	ldr	x16, [x16, #0xb8]
    1720: d61f0200     	br	x16
    1724: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1728: f9406210     	ldr	x16, [x16, #0xc0]
    172c: d61f0200     	br	x16
    1730: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1734: f9406610     	ldr	x16, [x16, #0xc8]
    1738: d61f0200     	br	x16
    173c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1740: f9406a10     	ldr	x16, [x16, #0xd0]
    1744: d61f0200     	br	x16
    1748: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    174c: f9406e10     	ldr	x16, [x16, #0xd8]
    1750: d61f0200     	br	x16
    1754: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1758: f9407210     	ldr	x16, [x16, #0xe0]
    175c: d61f0200     	br	x16
    1760: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1764: f9407610     	ldr	x16, [x16, #0xe8]
    1768: d61f0200     	br	x16
    176c: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    1770: f9407a10     	ldr	x16, [x16, #0xf0]
    1774: d61f0200     	br	x16
    1778: f0000030     	adrp	x16, 0x8000 <dyld_stub_binder+0x8000>
    177c: f9407e10     	ldr	x16, [x16, #0xf8]
    1780: d61f0200     	br	x16

Disassembly of section __TEXT,__stub_helper:

0000000000001784 <__stub_helper>:
    1784: f0000031     	adrp	x17, 0x8000 <dyld_stub_binder+0x8000>
    1788: 9106c231     	add	x17, x17, #0x1b0
    178c: a9bf47f0     	stp	x16, x17, [sp, #-0x10]!
    1790: f0000010     	adrp	x16, 0x4000 <dyld_stub_binder+0x4000>
    1794: f9402e10     	ldr	x16, [x16, #0x58]
    1798: d61f0200     	br	x16
    179c: 18000050     	ldr	w16, 0x17a4 <__stub_helper+0x20>
    17a0: 17fffff9     	b	0x1784 <__stub_helper>
    17a4: 00000000     	udf	#0x0
    17a8: 18000050     	ldr	w16, 0x17b0 <__stub_helper+0x2c>
    17ac: 17fffff6     	b	0x1784 <__stub_helper>
    17b0: 00000017     	udf	#0x17
    17b4: 18000050     	ldr	w16, 0x17bc <__stub_helper+0x38>
    17b8: 17fffff3     	b	0x1784 <__stub_helper>
    17bc: 0000002f     	udf	#0x2f
    17c0: 18000050     	ldr	w16, 0x17c8 <__stub_helper+0x44>
    17c4: 17fffff0     	b	0x1784 <__stub_helper>
    17c8: 00000045     	udf	#0x45
    17cc: 18000050     	ldr	w16, 0x17d4 <__stub_helper+0x50>
    17d0: 17ffffed     	b	0x1784 <__stub_helper>
    17d4: 00000072     	udf	#0x72
    17d8: 18000050     	ldr	w16, 0x17e0 <__stub_helper+0x5c>
    17dc: 17ffffea     	b	0x1784 <__stub_helper>
    17e0: 00000096     	udf	#0x96
    17e4: 18000050     	ldr	w16, 0x17ec <__stub_helper+0x68>
    17e8: 17ffffe7     	b	0x1784 <__stub_helper>
    17ec: 000000b6     	udf	#0xb6
    17f0: 18000050     	ldr	w16, 0x17f8 <__stub_helper+0x74>
    17f4: 17ffffe4     	b	0x1784 <__stub_helper>
    17f8: 000000d8     	udf	#0xd8
    17fc: 18000050     	ldr	w16, 0x1804 <__stub_helper+0x80>
    1800: 17ffffe1     	b	0x1784 <__stub_helper>
    1804: 00000139     	udf	#0x139
    1808: 18000050     	ldr	w16, 0x1810 <__stub_helper+0x8c>
    180c: 17ffffde     	b	0x1784 <__stub_helper>
    1810: 0000018a     	udf	#0x18a
    1814: 18000050     	ldr	w16, 0x181c <__stub_helper+0x98>
    1818: 17ffffdb     	b	0x1784 <__stub_helper>
    181c: 000001cf     	udf	#0x1cf
    1820: 18000050     	ldr	w16, 0x1828 <__stub_helper+0xa4>
    1824: 17ffffd8     	b	0x1784 <__stub_helper>
    1828: 00000211     	udf	#0x211
    182c: 18000050     	ldr	w16, 0x1834 <__stub_helper+0xb0>
    1830: 17ffffd5     	b	0x1784 <__stub_helper>
    1834: 0000024c     	udf	#0x24c
    1838: 18000050     	ldr	w16, 0x1840 <__stub_helper+0xbc>
    183c: 17ffffd2     	b	0x1784 <__stub_helper>
    1840: 00000287     	udf	#0x287
    1844: 18000050     	ldr	w16, 0x184c <__stub_helper+0xc8>
    1848: 17ffffcf     	b	0x1784 <__stub_helper>
    184c: 000002a3     	udf	#0x2a3
    1850: 18000050     	ldr	w16, 0x1858 <__stub_helper+0xd4>
    1854: 17ffffcc     	b	0x1784 <__stub_helper>
    1858: 000002bf     	udf	#0x2bf
    185c: 18000050     	ldr	w16, 0x1864 <__stub_helper+0xe0>
    1860: 17ffffc9     	b	0x1784 <__stub_helper>
    1864: 000002fe     	udf	#0x2fe
    1868: 18000050     	ldr	w16, 0x1870 <__stub_helper+0xec>
    186c: 17ffffc6     	b	0x1784 <__stub_helper>
    1870: 00000321     	udf	#0x321
    1874: 18000050     	ldr	w16, 0x187c <__stub_helper+0xf8>
    1878: 17ffffc3     	b	0x1784 <__stub_helper>
    187c: 00000344     	udf	#0x344
    1880: 18000050     	ldr	w16, 0x1888 <__stub_helper+0x104>
    1884: 17ffffc0     	b	0x1784 <__stub_helper>
    1888: 0000037b     	udf	#0x37b
    188c: 18000050     	ldr	w16, 0x1894 <__stub_helper+0x110>
    1890: 17ffffbd     	b	0x1784 <__stub_helper>
    1894: 00000393     	udf	#0x393
    1898: 18000050     	ldr	w16, 0x18a0 <__stub_helper+0x11c>
    189c: 17ffffba     	b	0x1784 <__stub_helper>
    18a0: 000003b4     	udf	#0x3b4
    18a4: 18000050     	ldr	w16, 0x18ac <__stub_helper+0x128>
    18a8: 17ffffb7     	b	0x1784 <__stub_helper>
    18ac: 000003ce     	udf	#0x3ce
    18b0: 18000050     	ldr	w16, 0x18b8 <__stub_helper+0x134>
    18b4: 17ffffb4     	b	0x1784 <__stub_helper>
    18b8: 000003e6     	udf	#0x3e6
    18bc: 18000050     	ldr	w16, 0x18c4 <__stub_helper+0x140>
    18c0: 17ffffb1     	b	0x1784 <__stub_helper>
    18c4: 00000403     	udf	#0x403
    18c8: 18000050     	ldr	w16, 0x18d0 <__stub_helper+0x14c>
    18cc: 17ffffae     	b	0x1784 <__stub_helper>
    18d0: 00000417     	udf	#0x417
    18d4: 18000050     	ldr	w16, 0x18dc <__stub_helper+0x158>
    18d8: 17ffffab     	b	0x1784 <__stub_helper>
    18dc: 00000427     	udf	#0x427
    18e0: 18000050     	ldr	w16, 0x18e8 <__stub_helper+0x164>
    18e4: 17ffffa8     	b	0x1784 <__stub_helper>
    18e8: 00000436     	udf	#0x436
    18ec: 18000050     	ldr	w16, 0x18f4 <__stub_helper+0x170>
    18f0: 17ffffa5     	b	0x1784 <__stub_helper>
    18f4: 00000446     	udf	#0x446
    18f8: 18000050     	ldr	w16, 0x1900 <__stub_helper+0x17c>
    18fc: 17ffffa2     	b	0x1784 <__stub_helper>
    1900: 00000455     	udf	#0x455
    1904: 18000050     	ldr	w16, 0x190c <__stub_helper+0x188>
    1908: 17ffff9f     	b	0x1784 <__stub_helper>
    190c: 00000463     	udf	#0x463
    1910: 18000050     	ldr	w16, 0x1918 <__stub_helper+0x194>
    1914: 17ffff9c     	b	0x1784 <__stub_helper>
    1918: 00000472     	udf	#0x472
