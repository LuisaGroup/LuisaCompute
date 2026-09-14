// Build and inspect the user-selected Sphinx surface, including exact values.
const fs = require("node:fs");
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
    const build = path.resolve(process.argv[2]);
    const root = path.resolve(__dirname, "../../../../..");
    const command = ["run", "--offline", "--no-project", "--python", "3.13",
        "--with", "sphinx", "--with", "sphinx-rtd-theme", "--with", "myst-parser", "--with", "breathe",
        "sphinx-build", "-b", "html", "-E", "-a", "-W", "--keep-going", "docs", build];
    const built = spawnSync("uv", command, {cwd: root, encoding: "utf8"});
    const log = built.stdout + built.stderr;
    fs.writeFileSync(path.join(__dirname, "sphinx.log"), log);
    const warnings = log.split("\n").filter(line => /WARNING:|ERROR:/.test(line));
    const knownOnly = warnings.length === 10 && warnings.every(line => /api_reference\.rst/.test(line) && /index\.xml/.test(line));
    if (built.status !== 0 && !(built.status === 1 && knownOnly)) throw new Error("Unexpected Sphinx failure: " + log);
    const links = spawnSync("uv", ["run", "--offline", "--no-project", "--python", "3.13", "--with", "sphinx",
        "python", "scripts/check_docs.py", build], {cwd: root, encoding: "utf8"});
    fs.writeFileSync(path.join(__dirname, "docs-links.log"), links.stdout + links.stderr);
    if (links.status !== 0) throw new Error("Local link check failed: " + links.stdout + links.stderr);
    const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "audit.json"), "utf8"));
    const expected = audit.summary.map(row => {
        const v = row.metrics.gpu_throughput;
        return [row.aligned_control ? `${row.shape[0]}³ — unchanged control` : row.shape.join("×"),
            v.reference_us.toFixed(2), v.candidate_us.toFixed(2),
            `${v.new_old.median.toFixed(3)} [${v.new_old.minimum.toFixed(3)}–${v.new_old.maximum.toFixed(3)}]`,
            v.new_torch.median.toFixed(3), v.new_mps.median.toFixed(3)];
    });
    const browser = await chromium.launch({headless: true,
        executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
    const records = [];
    try {
        for (const viewport of [{width: 1440, height: 1050}, {width: 390, height: 844}]) {
            for (const [relative, anchor, kind] of [
                ["source/performance/tile/results.html", "bounded-output-removes-shared-c-not-the-whole-library-gap", "results"],
                ["source/internals/tile/matrix.html", "output-bounds-are-independent-of-input-padding", "proof"],
                ["source/performance/tile/index.html", "current-conclusion", "index"],
            ]) {
                const page = await browser.newPage({viewport});
                const url = pathToFileURL(path.join(build, relative)).href + "#" + anchor;
                await page.goto(url, {waitUntil: "load"});
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator("#" + anchor);
                await section.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 24));
                const screenshot = path.join(path.dirname(build), `${kind}-${viewport.width}.png`);
                await page.screenshot({path: screenshot});
                const dimensions = await page.evaluate(() => ({viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth}));
                let tableMatchesAudit = null;
                const tableScreenshots = [];
                if (kind === "results") {
                    const table = section.locator("table").first();
                    const actual = await table.locator("tbody tr").evaluateAll(rows =>
                        rows.map(row => Array.from(row.querySelectorAll("td"), cell => cell.textContent.trim())));
                    tableMatchesAudit = JSON.stringify(actual) === JSON.stringify(expected);
                    if (!tableMatchesAudit) throw new Error("Rendered values differ from independent audit: " + JSON.stringify(actual));
                    await table.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 60));
                    for (const edge of ["left", "right"]) {
                        await table.evaluate((el, edge) => {
                            const wrapper = el.closest(".wy-table-responsive");
                            if (wrapper) wrapper.scrollLeft = edge === "right" ? wrapper.scrollWidth : 0;
                        }, edge);
                        const file = path.join(path.dirname(build), `table-${viewport.width}-${edge}.png`);
                        await page.screenshot({path: file});
                        tableScreenshots.push(file);
                    }
                }
                records.push({url, viewport, dimensions, tableMatchesAudit, screenshot, tableScreenshots});
                await page.close();
            }
        }
    } finally {
        await browser.close();
    }
    if (records.some(row => row.dimensions.page > row.dimensions.viewport + 1)) throw new Error("Page horizontal overflow");
    const receipt = {surface: "existing Sphinx docs; no new reader pages", build,
        sphinx_command: ["uv", ...command], sphinx_exit: built.status, warnings,
        new_tile_warnings: 0, local_links: links.stdout.trim(), records,
        full_api_docs_certified: false, manual_image_inspection_required: true};
    fs.writeFileSync(path.join(__dirname, "docs-qa.json"), JSON.stringify(receipt, null, 2) + "\n");
    process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
})().catch(error => {console.error(error); process.exitCode = 1;});
