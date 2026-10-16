# Physical frame indices in selective-copy regression

The pre-existing `selective_frame_copy_does_not_touch_inactive_fields`
regression supplied payload-local indices to storage APIs that require physical
frame indices, including the seven reserved invocation/token fields.

Its float3 write at payload index 3 actually targeted physical field 3
(`dispatch_size_x`). For the four-slot AoS test, the address was
`3 * 64 + 12 = 204`, violating float3's 16-byte alignment. HIP happened to
execute this invalid test; fallback's explicit alignment check rejected it.
The complete frame-layout test source otherwise differs from the published
baseline only by stream-submission spelling.

The correction consistently adds `CoroFrameDesc::reserved_field_count` to
the selected field list and all field accesses. Expected data is unchanged;
neither frame layout nor backend alignment rules are weakened.

Evidence: `/var/tmp/psycles-coro-resume-8fNIXm`.

- `sdk-fallback-test_coro_soa_layout.log`: original rejection at byte 204.
- `layout-green-fallback.log`: corrected complete executable passes.
- `clean-hip-test_coro_soa_layout.log` and
  `clean-fallback-test_coro_soa_layout.log`: isolated candidate source and
  headers, all 13 tests and 126 assertions pass on each backend.
