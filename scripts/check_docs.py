"""Check generated Sphinx HTML links/assets and compatibility redirect anchors.

Run after a fresh HTML build, with Sphinx available in the Python environment.
This checks local targets only; it does not replace rendered-page or API QA.
"""

import argparse
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(root):
    pages = {p: Page(p) for p in root.rglob("*.html")
             if "_downloads" not in p.relative_to(root).parts
             and "_static" not in p.relative_to(root).parts}
    if root / "index.html" not in pages:
        raise ValueError(f"No generated Sphinx index.html in {root}")
    errors = []
    checked = 0
    for source, page in pages.items():
        for link in page.links:
            uri = urlsplit(link)
            if uri.scheme or uri.netloc:
                continue
            # Root-relative links refer to the generated website, not the
            # machine filesystem. Current docs otherwise use relative paths.
            path = unquote(uri.path)
            target = ((root / path.lstrip("/")) if path.startswith("/") else
                      (source.parent / path) if path else source).resolve()
            if target.is_dir():
                target /= "index.html"
            checked += 1
            if not target.is_relative_to(root):
                errors.append(f"{source.relative_to(root)} -> {link}: escapes generated website")
            elif not target.exists():
                errors.append(f"{source.relative_to(root)} -> {link}: missing file")
            elif uri.fragment and target in pages and unquote(uri.fragment) not in pages[target].ids:
                errors.append(f"{source.relative_to(root)} -> {link}: missing anchor")

    redirect_file = Path(__file__).resolve().parents[1] / "docs/_ext/legacy_urls.py"
    spec = importlib.util.spec_from_file_location("legacy_urls", redirect_file)
    redirects = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(redirects)
    fragments = 0
    for old, destination in redirects.REDIRECTS.items():
        if root / (old + ".html") not in pages:
            errors.append(f"{old}: missing legacy redirect page")
        if root / (destination + ".html") not in pages:
            errors.append(f"{old} -> {destination}: missing redirect target")
        for anchor, target in redirects.FRAGMENTS.get(old, {}).items():
            anchor = redirects.RENAMED_FRAGMENTS.get(old, {}).get(anchor, anchor)
            fragments += 1
            page = pages.get(root / (target + ".html"))
            if page is None or anchor not in page.ids:
                errors.append(f"{old} -> {target}#{anchor}: missing redirect anchor")
        # Renamed anchors can also stay on the main redirect destination.
        for old_anchor, anchor in redirects.RENAMED_FRAGMENTS.get(old, {}).items():
            if old_anchor in redirects.FRAGMENTS.get(old, {}):
                continue
            fragments += 1
            page = pages.get(root / (destination + ".html"))
            if page is None or anchor not in page.ids:
                errors.append(f"{old}#{old_anchor} -> {destination}#{anchor}: missing renamed anchor")
    for source, mapping in redirects.SPLIT_FRAGMENTS.items():
        if root / (source + ".html") not in pages:
            errors.append(f"{source}: missing split-page overview")
        for anchor, target in mapping.items():
            fragments += 1
            page = pages.get(root / (target + ".html"))
            if page is None or anchor not in page.ids:
                errors.append(f"{source}#{anchor} -> {target}: missing moved anchor")
    print(f"{len(pages)} HTML pages; {checked} local links/assets; {fragments} compatibility anchors")
    print("\n".join(errors) if errors else "PASS: all checked local targets and anchors exist")
    return bool(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path, help="generated Sphinx HTML directory")
    args = parser.parse_args()
    raise SystemExit(check(args.output_directory.resolve()))
