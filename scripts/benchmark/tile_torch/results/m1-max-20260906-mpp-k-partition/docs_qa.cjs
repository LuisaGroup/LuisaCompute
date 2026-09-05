// Render the existing Sphinx surface; no parallel report/chart implementation.
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
    const build = path.resolve(process.argv[2]);
    const output = fs.mkdtempSync(path.join(os.tmpdir(), "tile-diagnostics-docs-"));
    const evidence = JSON.parse(fs.readFileSync(path.join(__dirname, "audit.json"), "utf8"));
    const shapes = [[1024,1024,1537], [4096,4096,4096], [4096,4096,11008]];
    const expected = shapes.flatMap(shape => ["forward", "reverse"].map((order, index) => [
        shape.every(x => x === 4096) ? "4096³" : shape.join("×"),
        index === 0 ? "A" : "B",
        ...[128,512,1024,4096].map(k =>
            evidence[order].measurements.find(row => row.phase === "trial" &&
                row.shape.join() === shape.join() && row.block[2] === k)
                .paths.native.gpu_batch_us.toFixed(3))
    ]));
    const browser = await chromium.launch({headless: true,
        executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
    const records = [];
    try {
        for (const viewport of [{width: 1440, height: 1050}, {width: 390, height: 844}]) {
            for (const [relative, anchor, kind] of [
                ["source/performance/tile/results.html", "k-partition-and-program-walks-diagnostics-not-new-defaults", "results"],
                ["source/internals/tile/matrix.html", "physical-program-traversal-remains-a-candidate", "proof"],
            ]) {
                const page = await browser.newPage({viewport});
                const url = pathToFileURL(path.join(build, relative)).href + "#" + anchor;
                await page.goto(url, {waitUntil: "load"});
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator("#" + anchor);
                await section.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 24));
                const screenshot = path.join(output, `${kind}-${viewport.width}.png`);
                await page.screenshot({path: screenshot});
                const dimensions = await page.evaluate(() => ({
                    viewport: document.documentElement.clientWidth,
                    page: document.documentElement.scrollWidth,
                }));
                const overflow = dimensions.page > dimensions.viewport + 1;
                const overflowElements = overflow ? await page.evaluate(() =>
                    Array.from(document.querySelectorAll("main *, .document *"))
                        .filter(el => !el.closest("table, .highlight, .wy-table-responsive"))
                        .map(el => {
                        const rect = el.getBoundingClientRect();
                        return {tag: el.tagName, id: el.id, className: el.className,
                                left: rect.left, right: rect.right, width: rect.width,
                                text: el.textContent.slice(0, 100)};
                    }).filter(el => el.right > document.documentElement.clientWidth + 1)
                        .sort((a, b) => b.right - a.right).slice(0, 12)) : [];
                let tableMatchesAudit = null;
                const tableScreenshots = [];
                if (kind === "results") {
                    const actual = await section.locator("table").first().locator("tbody tr").evaluateAll(
                        rows => rows.map(row => Array.from(row.querySelectorAll("td"), cell => cell.textContent.trim())));
                    tableMatchesAudit = JSON.stringify(actual) === JSON.stringify(expected);
                    if (!tableMatchesAudit) throw new Error("Rendered table differs from audit: " + JSON.stringify(actual));
                    if (viewport.width < 600) {
                        const table = section.locator("table").first();
                        await table.evaluate(el => window.scrollTo(0, window.scrollY + el.getBoundingClientRect().top - 80));
                        for (const side of ["left", "right"]) {
                            await table.evaluate((el, edge) => {
                                const wrapper = el.closest(".wy-table-responsive");
                                if (wrapper) wrapper.scrollLeft = edge === "right" ? wrapper.scrollWidth : 0;
                            }, side);
                            const file = path.join(output, `table-${viewport.width}-${side}.png`);
                            await page.screenshot({path: file});
                            tableScreenshots.push(file);
                        }
                    }
                }
                records.push({url, viewport, dimensions, overflow, overflowElements, tableMatchesAudit, screenshot, tableScreenshots});
                await page.close();
            }
        }
    } finally {
        await browser.close();
    }
    const receipt = {build, records, manual_image_inspection_required: true};
    fs.writeFileSync(path.join(__dirname, "docs-qa.json"), JSON.stringify(receipt, null, 2) + "\n");
    process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
    if (records.some(record => record.overflow)) process.exitCode = 1;
})().catch(error => {console.error(error); process.exitCode = 1;});
