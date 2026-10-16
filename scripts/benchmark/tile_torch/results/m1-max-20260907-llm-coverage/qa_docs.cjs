// QA the requested Sphinx surface, not a second report renderer.
const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    fs.mkdirSync(output, {recursive: true});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        const errors = [];
        page.on('pageerror', e => errors.push(e.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, expected] of [
                ['source/performance/tile/index.html', 'current-conclusion', []],
                ['source/performance/tile/index.html', 'validation-and-next-milestone', []],
                ['source/performance/tile/results.html', 'partitioned-outputs-remove-the-rope-mapping-fallback', [[5, 6]]],
                ['source/performance/tile/results.html', 'attention-and-direct-simd-still-need-richer-execution-mappings', [[3, 5], [7, 4]]],
                ['source/internals/tile/lowering.html', 'automatic-gpu-pointwise-graphs', []],
                ['source/performance/tile/validation.html', 'correctness-common-llm-operators-now-use-both-bridges', [[6, 3]]],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const counts = await section.locator('table').evaluateAll(tables => tables.map(t => [t.rows.length, t.rows[0].cells.length]));
                if (JSON.stringify(counts) !== JSON.stringify(expected)) throw new Error('missing table cells: ' + id);
                const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
                if (pageWidth > width + 1) throw new Error('page horizontal overflow: ' + id);
                const screenshot = path.join(output, id + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                receipts.push({file, id, width, screenshot, pageWidth, counts});
                if (width === 390) {
                    for (let index = 0; index < expected.length; index++) {
                        const table = section.locator('table').nth(index);
                        const right = await table.evaluate(t => {
                            const p = t.parentElement;
                            p.scrollLeft = p.scrollWidth;
                            return {offset: p.scrollLeft, lastRight: t.rows[0].cells[t.rows[0].cells.length - 1].getBoundingClientRect().right,
                                containerRight: p.getBoundingClientRect().right};
                        });
                        if (right.lastRight > right.containerRight + 1) throw new Error('inaccessible right column');
                        const screenshot = path.join(output, id + '-390-table-' + index + '-right.png');
                        await table.locator('..').screenshot({path: screenshot});
                        receipts.push({file, id, width, screenshot, right});
                    }
                }
            }
        }
        if (errors.length) throw new Error(errors.join('\n'));
        const result = {passed: true, browser: await browser.version(), receipts};
        fs.writeFileSync(path.join(__dirname, 'docs-qa.json'), JSON.stringify(result, null, 2) + '\n');
        console.log(JSON.stringify(result, null, 2));
    } finally {
        await browser.close();
    }
})().catch(e => { console.error(e); process.exitCode = 1; });
