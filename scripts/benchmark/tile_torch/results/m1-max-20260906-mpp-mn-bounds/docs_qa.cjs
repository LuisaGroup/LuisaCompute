// Verify the repository's existing Sphinx surface, not a parallel report.
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const {pathToFileURL} = require("node:url");
const {chromium} = require("playwright");

(async () => {
    const build = path.resolve(process.argv[2]);
    const output = fs.mkdtempSync(path.join(os.tmpdir(), "tile-mnk-docs-qa-"));
    const audit = JSON.parse(fs.readFileSync(path.join(__dirname, "audit.json"), "utf8"));
    const comparisons = audit["shuffle-replay"].comparisons;
    const expected = comparisons.filter(row => row.order === "forward").map(row => {
        const pair = [row, comparisons.find(other => other.order === "reverse" && other.shape.join() === row.shape.join())];
        const values = select => pair.map(item => select(item).toFixed(3)).join(" / ");
        return [row.shape.every(x => x === 1024) ? "1024³ control" : row.shape.join("×"),
                values(item => item.old.native.gpu_batch_us), values(item => item.new.native.gpu_batch_us),
                values(item => item.new_over_mps.gpu_batch_us), values(item => item.new_over_torch.gpu_batch_us)];
    });
    const browser = await chromium.launch({headless: true,
        executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"});
    const records = [];
    try {
        for (const viewport of [{width: 1440, height: 1050}, {width: 390, height: 844}]) {
            for (const [relative, anchor, kind] of [
                ["source/performance/tile/results.html", "bounded-m-n-inputs-remove-an-admission-barrier", "results"],
                ["source/internals/tile/matrix.html", "bounded-m-n-views-compose-with-subgroup-coordinates", "proof"],
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
                    viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth}));
                let tableMatchesAudit = null;
                const tableScreenshots = [];
                if (kind === "results") {
                    const table = section.locator("table").first();
                    const actual = await table.locator("tbody tr").evaluateAll(rows =>
                        rows.map(row => Array.from(row.querySelectorAll("td"), cell => cell.textContent.trim())));
                    tableMatchesAudit = JSON.stringify(actual) === JSON.stringify(expected);
                    if (!tableMatchesAudit) throw new Error("Rendered table differs from audit: " + JSON.stringify(actual));
                    if (viewport.width < 600) {
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
                records.push({url, viewport, dimensions, tableMatchesAudit, screenshot, tableScreenshots});
                await page.close();
            }
        }
    } finally {
        await browser.close();
    }
    const receipt = {build, records, manual_image_inspection_required: true};
    fs.writeFileSync(path.join(__dirname, "docs-qa.json"), JSON.stringify(receipt, null, 2) + "\n");
    process.stdout.write(JSON.stringify(receipt, null, 2) + "\n");
    if (records.some(row => row.dimensions.page > row.dimensions.viewport + 1)) process.exitCode = 1;
})().catch(error => {console.error(error); process.exitCode = 1;});
