#!/usr/bin/env python3
"""Finite examples/counterexamples for the proposed execution calculus.

This is a dependency-free reference model, not a verifier for generated code
or a mechanized proof. Run: python3 scripts/test_tile_execution_calculus.py
The rectangular task product follows Hidet (ASPLOS 2023), Section 5.1:
https://arxiv.org/abs/2210.09603
"""

from dataclasses import dataclass
import itertools
import math
import unittest


def domain(shape):
    return tuple(itertools.product(*(range(n) for n in shape)))


@dataclass(frozen=True)
class TaskMap:
    shape: tuple
    workers: tuple  # One ordered tuple of task coordinates per worker.

    def product(self, inner):
        if len(self.shape) != len(inner.shape):
            raise ValueError("task ranks must agree")
        return TaskMap(
            tuple(a * b for a, b in zip(self.shape, inner.shape)),
            tuple(
                tuple(
                    tuple(a * n + b for a, n, b in zip(x, inner.shape, y))
                    for x in outer_tasks for y in inner_tasks
                )
                for outer_tasks in self.workers for inner_tasks in inner.workers
            ),
        )

    @staticmethod
    def spatial(shape):
        return TaskMap(shape, tuple((point,) for point in domain(shape)))

    @staticmethod
    def repeat(shape):
        return TaskMap(shape, (domain(shape),))


def factor_prefix(points, remap, new_cut, old_cut):
    """Return a bijective prefix witness, or None when no such witness exists."""
    witness = {}
    for point in points:
        key, value = point[:new_cut], remap(point)[:old_cut]
        if key in witness and witness[key] != value:
            return None
        witness[key] = value
    if len(set(witness.values())) != len(witness):
        return None
    return witness


def obeys(order, edges):
    position = {event: i for i, event in enumerate(order)}
    return all(position[a] < position[b] for a, b in edges)


def schedules(events, edges):
    return {order for order in itertools.permutations(events) if obeys(order, edges)}


def versions_fit(ii, live_duration, slots, count=12):
    intervals = [(k % slots, k * ii, k * ii + live_duration) for k in range(count)]
    return all(
        slot_a != slot_b or end_a <= start_b or end_b <= start_a
        for (slot_a, start_a, end_a), (slot_b, start_b, end_b)
        in itertools.combinations(intervals, 2)
    )


class ExecutionCalculusTests(unittest.TestCase):
    def test_rectangular_product_associativity_and_coverage(self):
        maps = [make(shape) for shape in ((1, 1), (1, 2), (2, 1), (2, 2))
                for make in (TaskMap.spatial, TaskMap.repeat)]
        for a, b, c in itertools.product(maps, repeat=3):
            left = a.product(b).product(c)
            self.assertEqual(left, a.product(b.product(c)))
            points = tuple(itertools.chain.from_iterable(left.workers))
            self.assertEqual(len(points), math.prod(left.shape))
            self.assertEqual(set(points), set(domain(left.shape)))

    def test_product_is_not_commutative(self):
        a, b = TaskMap.spatial((2,)), TaskMap.repeat((2,))
        self.assertEqual(a.product(b).workers, (((0,), (1,)), ((2,), (3,))))
        self.assertEqual(b.product(a).workers, (((0,), (2,)), ((1,), (3,))))
        self.assertNotEqual(a.product(b), b.product(a))
        with self.assertRaises(ValueError):
            a.product(TaskMap.repeat((1, 2)))

    def test_split_tail_exact_coverage(self):
        for n in range(18):
            for factor in range(1, 9):
                points = [q * factor + r for q in range((n + factor - 1) // factor)
                          for r in range(factor) if q * factor + r < n]
                self.assertEqual(points, list(range(n)))

    def test_flat_bijection_does_not_preserve_observed_prefix(self):
        points = domain((3, 2))
        transpose = lambda p: (p[1], p[0])
        self.assertEqual({transpose(p) for p in points}, set(domain((2, 3))))
        self.assertIsNone(factor_prefix(points, transpose, 1, 1))
        # Inserting a unit factor retains the outer resource owner.
        extended = domain((2, 1, 3))
        project = lambda p: (p[0], p[2])
        self.assertIsNotNone(factor_prefix(extended, project, 1, 1))
        self.assertIsNotNone(factor_prefix(extended, project, 2, 1))

    def test_order_strength_does_not_infer_independence(self):
        events = (0, 1, 2)
        weak = schedules(events, {(0, 2)})
        strong = schedules(events, {(0, 1), (1, 2)})
        self.assertLess(strong, weak)
        self.assertLess(weak, schedules(events, set()))
        # An ordered recurrence is valid but has an inter-instance RAW edge.
        reads, writes = ({"state"}, {"state"}), ({"state"}, {"state"})
        self.assertTrue(writes[0] & reads[1])

    def test_parallel_fusion_cross_edges(self):
        events = tuple((phase, i) for phase in ("A", "B") for i in range(3))
        fused = {(("A", i), ("B", i)) for i in range(3)}
        candidates = schedules(events, fused)
        self.assertTrue(candidates)
        self.assertTrue(all(obeys(order, fused) for order in candidates))
        neighbor = {(("A", (i + 1) % 3), ("B", i)) for i in range(3)}
        self.assertTrue(any(not obeys(order, neighbor) for order in candidates))
        # Simulate the neighbor consumer: pairwise fusion can read old values.
        def execute(order):
            temporary, output = [-1] * 3, [None] * 3
            for phase, i in order:
                if phase == "A":
                    temporary[i] = i + 10
                else:
                    output[i] = temporary[(i + 1) % 3]
            return tuple(output)
        reference = execute(events)
        self.assertTrue(any(execute(order) != reference for order in candidates))

    def test_serial_fusion_inequality(self):
        fused = tuple((phase, i) for i in range(4) for phase in ("A", "B"))
        for i, j in itertools.product(range(4), repeat=2):
            self.assertEqual(obeys(fused, {(("A", i), ("B", j))}), i <= j)

    def test_reduction_fibers_order_and_scan(self):
        # Concatenation is associative and noncommutative.
        rows = ("abcdef", "uvwxyz")
        for row in rows:
            for cut in range(len(row) + 1):
                self.assertEqual("".join((row[:cut], row[cut:])), row)
            self.assertNotEqual(row[::2] + row[1::2], row)
            prefixes = tuple(row[:i + 1] for i in range(len(row)))
            self.assertEqual(prefixes[-1], row)
            self.assertNotEqual(prefixes, (row,))
        # Commutative integer addition permits worker-striped partials.
        for n in range(17):
            values = list(range(n))
            self.assertEqual(sum(map(sum, (values[::2], values[1::2]))), sum(values))

    def test_pipeline_version_lifetimes_and_capacity(self):
        for ii, live, slots in itertools.product(range(1, 5), range(1, 10), range(1, 5)):
            self.assertEqual(versions_fit(ii, live, slots), slots * ii >= live)
        shared_capacity, other_live, one_version = 32, 12, 16
        self.assertLessEqual(other_live + one_version, shared_capacity)
        self.assertGreater(other_live + 2 * one_version, shared_capacity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
