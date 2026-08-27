#!/usr/bin/env python3
# =============================================================================
# torch2_export.py — torch.export -> portable JSON artifact for LuisaCompute
# =============================================================================
# Exports a torch.nn.Module's normalized ATen graph (torch.export.export() +
# run_decompositions(), i.e. the "Core ATen" IR) into a small, self-describing
# JSON artifact consumed by the C++ side of the pipeline
# (examples/tensor/torch2_import.cpp, invoked as
# `example_tensor_stub <backend> --rnn-pt2 rnn_exported.pt2.json`).
#
# The JSON carries:
#   - the graph structure (nodes / args / out shapes / dtypes),
#   - parameters and example/test inputs as base64(float32 LE),
#   - PyTorch reference outputs (and optional int64 labels),
# so the C++ importer is a single self-contained file with no torch dependency.
#
# A canonical torch.export.save()/.pt2 archive is additionally written and
# round-trip-verified for documentation; the C++ side reads only the JSON
# (parsing pickle + flatbuffers of the .pt2 archive in C++ is not tractable
# here — see the plan).
#
# Artifact schema (version 1):
# {
#   "schema": "luisa.torch2.export", "version": 1,
#   "producer": "torch.export", "ir": "core-aten", "model": "SequenceRNN",
#   "inputs":  [ {"name": ..., "shape": [...], "dtype": "float32",
#                 "kind": "user_input"}, ... ],
#   "params":  [ {"name": ..., "shape": [...], "dtype": "float32",
#                 "data_base64": "..."}, ... ],
#   "nodes":   [ {"name": ..., "op": "call_function", "target": "aten.mm",
#                 "args": [ {"type":"node","name":...} | scalar | [ ... ] ],
#                 "kwargs": [], "out_shape": [...], "out_dtype": "float32"}, ... ],
#   "outputs": [ {"name": ..., "shape": [...], "dtype": "float32"}, ... ],
#   "labels":  { "data_base64": "...", "shape": [...], "dtype": "int64" },
#   "reference": { "inputs": {"<name>": "b64..."},
#                  "outputs": {"<name>": "b64..."} }
# }
# =============================================================================
import argparse
import base64
import json
from collections import Counter

import torch
import torch.nn as nn


def tensor_to_base64(t: torch.Tensor, dtype=None) -> str:
    """Serialize a tensor as base64(float32 LE); int64 labels via .long()."""
    t = t.detach().cpu().contiguous()
    if dtype is not None:
        t = t.to(dtype)
    return base64.b64encode(t.numpy().tobytes()).decode("ascii")


def _dtype_str(dtype) -> str:
    return str(dtype).replace("torch.", "")


# torch.* metadata args (dtype/device/layout/memory_format) are not needed by
# the C++ executor, so they are dropped during serialization.  `None` and plain
# scalars are kept so positional semantics stay stable for the ops we execute.
_IGNORED_ARG_TYPES = (torch.dtype, torch.device, torch.layout, torch.memory_format)


def _serialize_arg(arg):
    """Recursively convert an fx node arg into the JSON form."""
    from torch.fx import Node as FXNode  # local import keeps the header light

    if isinstance(arg, FXNode):
        return {"type": "node", "name": arg.name}
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(a) for a in arg]
    if isinstance(arg, _IGNORED_ARG_TYPES):
        raise TypeError(f"torch metadata arg should have been dropped: {arg!r}")
    if isinstance(arg, bool) or arg is None or isinstance(arg, (int, float, str)):
        return arg
    raise TypeError(f"torch2 export: unsupported arg type {type(arg)!r} (value {arg!r})")


def op_name(target) -> str:
    """Normalize torch.ops.aten.mm.default / aten.mm.default / aten.mm -> 'aten.mm'."""
    s = str(target)
    if s.startswith("torch.ops.aten."):
        # torch.ops.aten.<op>.<overload> -> aten.<op>
        return f"aten.{s.split('.')[2]}"
    if s.startswith("aten."):
        # aten.<op>[.<overload>] -> aten.<op>
        return f"aten.{s.split('.')[1]}"
    raise ValueError(f"torch2 export: non-aten target {s!r}")


def _shape_dtype(val):
    if val is None:
        return [], "float32"
    return list(val.shape), _dtype_str(val.dtype)


def export_module_to_json(module, example_args, out_path, *,
                          model_name="Module", labels=None,
                          ref_args=None, ref_kwargs=None):
    """Export `module` to the portable JSON artifact and return the ExportedProgram.

    Steps:
      1. ep = torch.export.export(module, args=example_args) (strict default)
      2. ep = ep.run_decompositions(default_decompositions()) -> Core ATen IR
      3. placeholders -> inputs/params (bound via graph_signature + state_dict)
      4. nodes (skip placeholders; record output node args as outputs)
      5. reference outputs from ep.module()(*ref_args) under no_grad
    """
    ep = torch.export.export(module, args=tuple(example_args))
    if hasattr(torch.export, "default_decompositions"):
        decomp_table = torch.export.default_decompositions()
    else:  # torch < 2.6 fallback
        decomp_table = torch._decomp.decompositions
    ep = ep.run_decompositions(decomp_table=decomp_table)

    graph = ep.graph_module.graph
    signature = ep.graph_signature
    state_dict = ep.state_dict

    placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
    input_specs = list(signature.input_specs)
    if len(input_specs) != len(placeholder_nodes):
        raise RuntimeError(
            f"torch2 export: {len(placeholder_nodes)} placeholders but "
            f"{len(input_specs)} input specs")

    inputs, params = [], []
    param_pos = 0  # positional fallback when signature targets are stale
    for node, spec in zip(placeholder_nodes, input_specs):
        kind = str(getattr(spec, "kind", ""))
        shape, dtype = _shape_dtype(node.meta.get("val"))
        if "parameter" in kind.lower() or "constant" in kind.lower():
            target = getattr(spec, "target", None)
            if target is None or target not in state_dict:
                keys = [k for k in state_dict.keys()][param_pos:param_pos + 1]
                target = keys[0] if keys else node.name
            tensor = state_dict[target]
            params.append({
                "name": node.name,
                "shape": list(tensor.shape),
                "dtype": "float32",
                "data_base64": tensor_to_base64(tensor),
            })
            param_pos += 1
        else:
            inputs.append({
                "name": node.name,
                "shape": shape,
                "dtype": dtype,
                "kind": "user_input",
            })

    # --- graph body -----------------------------------------------------------
    nodes, outputs = [], []
    for node in graph.nodes:
        if node.op == "output":
            for out_arg in node.args:
                flat = out_arg if isinstance(out_arg, (list, tuple)) else (out_arg,)
                for a in flat:
                    if not isinstance(a, torch.fx.Node):
                        continue
                    shape, dtype = _shape_dtype(a.meta.get("val"))
                    outputs.append({"name": a.name, "shape": shape, "dtype": dtype})
            continue
        if node.op != "call_function":
            continue  # placeholders handled above; get_attr/prim ops are skipped
        target = op_name(node.target)
        args = []
        for a in node.args:
            if isinstance(a, _IGNORED_ARG_TYPES):
                continue  # drop dtype/device/layout/memory_format metadata
            args.append(_serialize_arg(a))
        shape, dtype = _shape_dtype(node.meta.get("val"))
        nodes.append({
            "name": node.name,
            "op": node.op,
            "target": target,
            "args": args,
            "kwargs": [],
            "out_shape": shape,
            "out_dtype": dtype,
        })

    # --- reference outputs ----------------------------------------------------
    with torch.no_grad():
        ref_args = tuple(ref_args if ref_args is not None else example_args)
        ref_kwargs = ref_kwargs or {}
        ref_out = ep.module()(*ref_args, **ref_kwargs)
    ref_outputs = {}
    if isinstance(ref_out, torch.Tensor):
        ref_outputs[outputs[0]["name"]] = tensor_to_base64(ref_out) \
            if outputs else tensor_to_base64(ref_out)
    else:
        flat = ref_out if isinstance(ref_out, (list, tuple)) else (ref_out,)
        for name, t in zip([o["name"] for o in outputs], flat):
            ref_outputs[name] = tensor_to_base64(t)

    reference = {"inputs": {}, "outputs": ref_outputs}
    for name, t in zip([i["name"] for i in inputs], ref_args):
        if isinstance(t, torch.Tensor):
            reference["inputs"][name] = tensor_to_base64(t)

    doc = {
        "schema": "luisa.torch2.export",
        "version": 1,
        "producer": "torch.export",
        "ir": "core-aten",
        "model": model_name,
        "inputs": inputs,
        "params": params,
        "nodes": nodes,
        "outputs": outputs,
    }
    if labels is not None:
        labels = labels.detach().cpu()
        doc["labels"] = {
            "data_base64": tensor_to_base64(labels, dtype=torch.int64),
            "shape": list(labels.shape),
            "dtype": "int64",
        }
    if reference:
        doc["reference"] = reference

    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    histogram = Counter(n["target"] for n in nodes)
    size = __import__("os").path.getsize(out_path)
    print(f"[torch2] exported {model_name}: {len(nodes)} nodes -> {out_path} "
          f"({size} bytes)")
    print(f"[torch2] op histogram: " +
          ", ".join(f"{t} x{c}" for t, c in sorted(histogram.items())))
    return ep


def save_canonical_pt2(ep, path) -> bool:
    """Write the canonical torch.export.save() archive and round-trip-verify it.

    The C++ importer does NOT parse this archive (pickle + flatbuffers); the
    archive is produced/validated here purely as documentation of the canonical
    PT2 flow.
    """
    torch.export.save(ep, path)
    loaded = torch.export.load(path)
    ok = loaded is not None
    print(f"[torch2] canonical .pt2 archive written + round-trip load: "
          f"{'OK' if ok else 'FAILED'} -> {path}")
    return ok


# -----------------------------------------------------------------------------
# Standalone self-test: x + 10 (also serves as a minimal torch2 demo).
# -----------------------------------------------------------------------------
class _AddTen(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([10.0]))

    def forward(self, x):
        return x + self.bias


def _self_test() -> None:
    import os
    import tempfile

    m = _AddTen()
    x = torch.randn(4)
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "self_test.pt2.json")
        pt2_path = os.path.join(tmp, "self_test.pt2")
        ep = export_module_to_json(m, (x,), json_path, model_name="AddTen")
        save_canonical_pt2(ep, pt2_path)
        with open(json_path) as f:
            doc = json.load(f)
        assert doc["schema"] == "luisa.torch2.export" and doc["version"] == 1
        assert any(n["target"] == "aten.add" for n in doc["nodes"]), \
            f"expected aten.add in self-test graph, got {doc['nodes']}"
        assert doc["params"], "expected the self-test parameter in the artifact"
        with torch.no_grad():
            ref = m(x)
            got = ep.module()(x)
        assert torch.allclose(ref, got, atol=1e-6), \
            f"self-test reference mismatch: {ref} vs {got}"
        print("[torch2] self-test OK: export + JSON + canonical .pt2 round trip")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="torch2 export utility self-test")
    ap.add_argument("--out", default="self_test.pt2.json")
    args = ap.parse_args()
    _self_test()
