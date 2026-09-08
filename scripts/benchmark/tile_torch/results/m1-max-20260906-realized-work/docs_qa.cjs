// Validate the existing Sphinx pages, not a parallel report application.
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
    if (links.status !== 0) throw new Error("Local link failure: " + links.stdout + links.stderr);
    const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "selection/audit.json"), "utf8"));
    if (audit.complete_outputs !== 768) throw new Error("Stale selection narrative");
    const paired = JSON.parse(fs.readFileSync(path.join(__dirname, "replay/audit.json"), "utf8"));
    if (paired.complete_outputs !== 288 || paired.paired_rounds !== 48) throw new Error("Incomplete replay evidence");
    const browser = await chromium.launch({headless: true, executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
    const records = [];
    try {
        for (const viewport of [{width: 1440, height: 1050}, {width: 390, height: 844}]) {
            for (const [relative, anchor] of [
                ["source/internals/tile/matrix.html", "realization-derived-work-before-candidate-pruning"],
                ["source/performance/tile/results.html", "realization-derived-work-and-model-selection"],
                ["source/performance/tile/index.html", "results-by-route"],
            ]) {
                const page = await browser.newPage({viewport});
                const url = pathToFileURL(path.join(build, relative)).href + "#" + anchor;
                await page.goto(url, {waitUntil: "load"});
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator("#" + anchor);
                await section.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 24));
                const text = await section.innerText();
                if (anchor.startsWith("realization-derived-work-and") && !["768", "288", "8192", "Torch"].every(token => text.includes(token))) throw new Error("Missing cohort/metric scope");
                if (anchor.startsWith("realization-derived-work-and")) {
                    const labels = ["512³ †", "4096³ †", "1025×1025×1024", "4096×4096×11008", "257×769×113 †", "2049×4097×1025", "4097×4097×4096 †", "8192³ †"];
                    const expected = paired.summary.map((row, i) => {
                        const gpu = row.metrics.gpu_throughput;
                        const ratio = gpu.new_old;
                        return [labels[i], `${ratio.median.toFixed(3)} (${ratio.minimum.toFixed(3)}–${ratio.maximum.toFixed(3)})`,
                            String(ratio.faster_rounds), row.metrics.e2e_throughput.new_old.median.toFixed(3),
                            gpu.new_torch.median.toFixed(3), gpu.new_mps.median.toFixed(3)];
                    });
                    const rows = await section.locator("table").last().locator("tbody tr").evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll("td")].map(cell => cell.textContent.trim())));
                    if (JSON.stringify(rows) !== JSON.stringify(expected)) throw new Error("Rendered metric table differs from raw-data audit");
                    if (!text.includes("remain slower than Torch")) throw new Error("Missing parity limitation");
                }
                const screenshot = path.join(path.dirname(build), `${anchor}-${viewport.width}.png`);
                await page.screenshot({path: screenshot});
                const dimensions = await page.evaluate(() => ({viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth}));
                if (dimensions.page > dimensions.viewport + 1) throw new Error("Page overflow");
                records.push({url, viewport, dimensions, screenshot});
                if (anchor.startsWith("realization-derived-work-and")) {
                    await section.locator("table").last().evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 70));
                    const tableScreenshot = path.join(path.dirname(build), `metrics-${viewport.width}.png`);
                    await page.screenshot({path: tableScreenshot});
                    records.push({url, viewport, screenshot: tableScreenshot, purpose: "complete metric table and caveats"});
                    if (viewport.width === 390) {
                        const scrolling = await section.locator("table").last().evaluate(table => {
                            let element = table.parentElement;
                            while (element && !(element.scrollWidth > element.clientWidth && ["auto", "scroll"].includes(getComputedStyle(element).overflowX))) element = element.parentElement;
                            if (!element) throw new Error("Wide mobile table has no horizontal scroll container");
                            element.scrollLeft = element.scrollWidth;
                            return {offset: element.scrollLeft, width: element.clientWidth, content: element.scrollWidth};
                        });
                        if (scrolling.offset <= 0) throw new Error("Mobile table cannot reach its rightmost columns");
                        const rightScreenshot = path.join(path.dirname(build), "metrics-right-390.png");
                        await page.screenshot({path: rightScreenshot});
                        records.push({url, viewport, screenshot: rightScreenshot, scrolling, purpose: "mobile rightmost columns reachable"});
                    }
                }
                await page.close();
            }
        }
    } finally { await browser.close(); }
    const receipt = {surface: "existing Sphinx documentation", build, sphinx_command: ["uv", ...command], sphinx_exit: built.status,
        warnings, new_tile_warnings: 0, local_links: links.stdout.trim(), records, full_api_docs_certified: false, manual_image_inspection_required: true};
    fs.writeFileSync(path.join(__dirname, "docs-qa.json"), JSON.stringify(receipt, null, 2) + "\n");
    process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
})().catch(error => {console.error(error); process.exitCode = 1;});
