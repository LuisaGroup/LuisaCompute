"""Host-only tests for opt-in isolation and actual-pointer alignment receipts."""
import copy
import itertools
import unittest

from cuda_matrix import alignment_receipts, route_environment


class AlignmentTests(unittest.TestCase):
    def test_inherited_flags_are_removed_for_every_control_and_route(self):
        inherited = {"PATH": "existing", "LUISA_CUDA_TILE_IR": "1", "LUISA_CUDA_TILE_IR_ALIGNED16": "1"}
        before = inherited.copy()
        for route in ("native", "tirx", "simd", "torch"):
            for enabled in (False, True):
                result = route_environment(inherited, route, enabled)
                self.assertEqual(result["PATH"], "existing")
                self.assertEqual(result.get("LUISA_CUDA_TILE_IR"), "1" if route == "native" else None)
                self.assertEqual(result.get("LUISA_CUDA_TILE_IR_ALIGNED16"), "1" if route == "native" and enabled else None)
        self.assertEqual(inherited, before)

    @staticmethod
    def packet(mask=11, residues=None, requested=True):
        residues = [0, 0, 0, 0] if residues is None else residues
        aligned = bool(mask) and all(value == 0 for slot, value in enumerate(residues) if mask & (1 << slot))
        realization = "CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin; no cache"
        if requested:
            realization += f"; aligned16-requested; aligned16-buffer-mask={mask}; "
            realization += "host-selected-dual-entry-aligned16-v1" if mask else "aligned16-ineligible"
        return dict(realization=realization, native_alignment=[dict(stage=0, requested=requested,
                    eligible_buffer_mask=mask, final_argument_mod16=residues,
                    expected_selected_entry="luisa_tile_aligned16" if aligned else "luisa_tile_main")])

    def test_actual_offsets_and_unused_argument_decide_selection(self):
        # 64/65 narrow elements -> byte residues 0/2. Include all combinations,
        # including the nonaligned unused slot, plus each possible one-root mask.
        for mask in (1, 2, 8, 11):
            for residues in itertools.product((0, 2), repeat=4):
                packet = self.packet(mask, list(residues))
                checked = alignment_receipts(packet, True)
                self.assertEqual(len(checked), 1)
                if mask == 11 and residues == (0, 0, 2, 0):
                    self.assertEqual(checked[0]["expected_selected_entry"], "luisa_tile_aligned16")

    def test_enabled_but_ineligible_is_explicitly_original(self):
        packet = self.packet(mask=0)
        checked = alignment_receipts(packet, True)
        self.assertEqual(checked[0]["expected_selected_entry"], "luisa_tile_main")
        packet["native_alignment"][0]["expected_selected_entry"] = "luisa_tile_aligned16"
        with self.assertRaisesRegex(ValueError, "selected entry"):
            alignment_receipts(packet, True)

    def test_default_and_legacy_control_are_not_specialized(self):
        self.assertEqual(alignment_receipts(dict(realization="old native route"), False), [])
        self.assertEqual(len(alignment_receipts(self.packet(mask=0, requested=False), False)), 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            alignment_receipts(dict(realization="old native route"), True)
        with self.assertRaisesRegex(ValueError, "request/stage"):
            alignment_receipts(self.packet(), False)

    def test_invalid_masks_residues_and_metadata_fail_closed(self):
        original = self.packet()
        changes = [dict(eligible_buffer_mask=16), dict(eligible_buffer_mask=-1), dict(eligible_buffer_mask=True),
                   dict(final_argument_mod16=[0, 0, 16, 0]), dict(final_argument_mod16=[0, 0, True, 0]),
                   dict(stage=1), dict(requested=1), dict(expected_selected_entry="luisa_tile_main")]
        for change in changes:
            packet = copy.deepcopy(original)
            packet["native_alignment"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                alignment_receipts(packet, True)
        packet = copy.deepcopy(original)
        packet["realization"] = packet["realization"].replace("mask=11;", "mask=3;")
        with self.assertRaisesRegex(ValueError, "eligibility"):
            alignment_receipts(packet, True)

    def test_multistage_receipts_keep_each_realization_and_binding_mask(self):
        first = self.packet()
        second = self.packet(mask=0)
        packet = dict(pipeline_stages=[dict(realization=first["realization"]), dict(realization=second["realization"])],
                      native_alignment=[first["native_alignment"][0], second["native_alignment"][0]])
        packet["native_alignment"][1]["stage"] = 1
        self.assertEqual(len(alignment_receipts(packet, True)), 2)
        packet["native_alignment"].pop()
        with self.assertRaisesRegex(ValueError, "per-stage"):
            alignment_receipts(packet, True)


if __name__ == "__main__":
    unittest.main()
