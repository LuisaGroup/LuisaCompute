// Inspect the existing Sphinx surface; no parallel report application.
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
    const build = path.resolve(process.argv[2]);
    const root = path.resolve(__dirname, "../../../../..");
    const command = ["run", "--offline", "--no-project", "--python", "3.13", "--with", "sphinx", "--with", "sphinx-rtd-theme",
        "--with", "myst-parser", "--with", "breathe", "sphinx-build", "-b", "html", "-E", "-a", "-W", "--keep-going", "docs", build];
    const built = spawnSync("uv", command, {cwd: root, encoding: "utf8"});
    const log = built.stdout + built.stderr;
    fs.writeFileSync(path.join(__dirname, "sphinx.log"), log);
    const warnings = log.split("\n").filter(line => /WARNING:|ERROR:/.test(line));
    const known = warnings.length === 10 && warnings.every(line => /api_reference\.rst/.test(line) && /index\.xml/.test(line));
    if (built.status !== 0 && !(built.status === 1 && known)) throw new Error("Unexpected Sphinx failure: " + log);
    const links = spawnSync("uv", ["run", "--offline", "--no-project", "--python", "3.13", "--with", "sphinx", "python",
        "scripts/check_docs.py", build], {cwd: root, encoding: "utf8"});
    fs.writeFileSync(path.join(__dirname, "docs-links.log"), links.stdout + links.stderr);
    if (links.status !== 0) throw new Error("Link check failed: " + links.stdout + links.stderr);
    const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "search/audit.json"), "utf8"));
    if (audit.complete_outputs !== 216 || audit.unchanged_common_candidates !== 24) throw new Error("Stale audit narrative");
    const browser = await chromium.launch({headless: true, executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
    const records = [];
    try {
        for (const viewport of [{width: 1440, height: 1050}, {width: 390, height: 844}]) {
            for (const [relative, anchor, kind] of [
                ["source/internals/tile/matrix.html", "fragment-budgets-belong-to-the-emitted-realization", "proof"],
                ["source/performance/tile/results.html", "mpp-state-budget-and-candidate-admission", "results"],
                ["source/performance/tile/index.html", "results-by-route", "index"],
            ]) {
                const page = await browser.newPage({viewport});
                const url = pathToFileURL(path.join(build, relative)).href + "#" + anchor;
                await page.goto(url, {waitUntil: "load"});
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator("#" + anchor);
                await section.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 24));
                const text = await section.innerText();
                if (kind === "results" && !["6/12", "10/12", "216", "24", "4,992", "42", "1.23–1.88"].every(token => text.includes(token))) throw new Error("Missing numerical scope");
                if (kind === "results" && !text.includes("no accepted speedup")) throw new Error("Missing performance caveat");
                if (kind === "proof") {
                    const rows = await section.locator("tbody tr").evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll("td")].map(cell => cell.textContent.trim())));
                    const expected = [["SIMD-group reference", "A fragments, B fragments, accumulator", "2 * (rm*rn + rm + rn)"],
                        ["MPP memory operands", "Output cooperative tensor", "2 * rm*rn"]];
                    if (JSON.stringify(rows) !== JSON.stringify(expected)) throw new Error("Wrong state formula table");
                }
                const screenshot = path.join(path.dirname(build), `${kind}-${viewport.width}.png`);
                await page.screenshot({path: screenshot});
                const dimensions = await page.evaluate(() => ({viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth}));
                if (dimensions.page > dimensions.viewport + 1) throw new Error("Page overflow");
                records.push({url, viewport, dimensions, screenshot});
                await page.close();
            }
        }
    } finally { await browser.close(); }
    const receipt = {surface: "existing Sphinx documentation", build, sphinx_command: ["uv", ...command], sphinx_exit: built.status,
        warnings, new_tile_warnings: 0, local_links: links.stdout.trim(), records, full_api_docs_certified: false, manual_image_inspection_required: true};
    fs.writeFileSync(path.join(__dirname, "docs-qa.json"), JSON.stringify(receipt, null, 2) + "\n");
    process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
})().catch(error => {console.error(error); process.exitCode = 1;});
