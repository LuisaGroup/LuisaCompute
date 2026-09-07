#!/usr/bin/env python3
"""Generate and execute the compact admission/data-quality companion."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

here = Path(__file__).resolve().parent
notebook = nbf.v4.new_notebook()
notebook.metadata.kernelspec = dict(name="python3", display_name="Python 3", language="python")
notebook.cells = [
    nbf.v4.new_markdown_cell("# MPP state-budget evidence review\n\n## tl;dr\n\nThe fixed search expands from six to ten valid candidates per shape, with 216 complete outputs checked. Search minima are not accepted performance wins; all eight fresh selections were slower than their selected trial. This companion audits saved evidence, and does not launch GPU work."),
    nbf.v4.new_markdown_cell("## Context & Methods\n\nTechnical companion to the existing Sphinx report. Four FP32 GEMM shapes, twelve candidate configurations per old/new compiler, five timing samples and preallocated outputs. GPU measurements are no-counter command-buffer intervals, not isolated kernel timestamps.\n\n### Key Assumptions\n\nSaved native/Torch/MPS numerical receipts are checked against the exact recorded output size. This audit recomputes raw timing denominators and selections but cannot reconstruct an idle desktop or prove the source of variable load. See `protocol.md`, `notes.md`, and the recorded binary/source hashes."),
    nbf.v4.new_markdown_cell("## Data\n\n### 1. Load the saved search and independent auditor\n\nRun from this notebook's directory. Only the registered experiment subtree is read."),
    nbf.v4.new_code_cell("import importlib.util\nimport json\nfrom pathlib import Path\n\nroot = Path.cwd()\nspec = importlib.util.spec_from_file_location('state_budget_audit', root / 'audit.py')\nauditor = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(auditor)\nreport = json.loads((root / 'search/results.json').read_text())\nchecked = auditor.audit_search(report, root / 'search')\nprint({key: value for key, value in checked.items() if key != 'summary'})"),
    nbf.v4.new_markdown_cell("## Results\n\n### 2. Admission and fresh-measurement instability\n\nExact lookup by shape and compiler variant. The last number is fresh GPU time divided by the same selected configuration's search time, not a compiler speedup."),
    nbf.v4.new_code_cell("for row in checked['summary']:\n    print('x'.join(map(str, row['shape'])), row['variant'],\n          f\"valid={row['valid_candidates']}/12\",\n          f\"fresh/search={row['selection_to_fresh_gpu_ratio']:.3f}\")\nassert checked['unchanged_common_candidates'] == 24\nassert checked['complete_outputs'] == 216\nassert all(row['selection_to_fresh_gpu_ratio'] > 1 for row in checked['summary'])"),
    nbf.v4.new_markdown_cell("### 3. Current numerical gates and old-library negative controls\n\nA test exits successfully only when nonzero assertions run; expected old-library failures are separately labeled."),
    nbf.v4.new_code_cell("receipt = json.loads((root / 'final-correctness/results.json').read_text())\nassert receipt['passed'] and receipt['metadata']['artifacts_unchanged']\nassert all(row['exit_code'] == 0 for row in receipt['metadata']['builds'])\npositive = [row for row in receipt['results'] if not row['expected_failure']]\nnegative = [row for row in receipt['results'] if row['expected_failure']]\nassert all(row['passed'] and row['passed_assertions'] > 0 for row in positive)\nassert len(negative) == 2 and all(row['passed'] and row['exit_code'] != 0 for row in negative)\nprint('Passing numerical test invocations:', len(positive))\nprint('Expected old-library negative controls:', len(negative))\nprint('Passing assertions:', sum(row['passed_assertions'] for row in positive))"),
    nbf.v4.new_markdown_cell("## Takeaways\n\nAdmission and numerical correctness are supported by saved, source-backed evidence. The 24 common candidate sources are unchanged, so their timing differences are not compiler transformations. Timing/model selection still requires a quiet-window, held-out and counterbalanced replay; no default schedule or cost coefficient is promoted. The PyTorch/MPS-equivalent performance objective remains open."),
]
nbf.validate(notebook)
NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(here)}}).execute()
nbf.validate(notebook)
nbf.write(notebook, here / "review.ipynb")
print("Validated and executed review.ipynb", flush=True)
