// Inspect the existing Sphinx surface; do not publish a second report renderer.
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
                ['internals/tile/related-work', 'closest-precedents', [[9, 3]]],
                ['internals/tile/related-work', 'cypress-the-closest-execution-resource-separation', []],
                ['internals/tile/related-work', 'twill-the-nearest-joint-cost-solver-comparison', []],
                ['internals/tile/related-work', 'tawa-compiler-internal-channels-with-operational-semantics', []],
                ['internals/tile/related-work', 'tirx-a-compatible-lower-boundary-not-our-automatic-planner', []],
                ['internals/tile/related-work', 'event-tensor-a-direct-precedent-for-fine-grained-sibling-execution', []],
                ['internals/tile/related-work', 'more-rigorous-not-unnecessarily-rigid', [[8, 3]]],
                ['internals/tile/related-work', 'design-consequences-and-concrete-comparison-cases', [[9, 3]]],
                ['internals/tile/related-work', 'a-falsifiable-contribution-not-a-novelty-slogan', [[7, 2]]],
                ['internals/tile/calculus', 'scope-of-the-claims', [[5, 2]]],
                ['internals/tile/calculus', 'typed-mapping-witness', [[6, 2]]],
                ['internals/tile/calculus', 'multiple-scopes-and-fusion', []],
                ['internals/tile/planner', 'formal-finite-optimization-problem', []],
                ['internals/tile/matrix', 'automatic-composed-programs', []],
                ['performance/tile/results', 'automatic-cooperation-removes-the-attention-worker-fallback', [[3, 7]]],
                ['performance/tile/index', 'validation-and-next-milestone', []],
            ]) {
                const route = 'source/' + file + '.html';
                await page.goto(pathToFileURL(path.join(root, route)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const counts = await section.locator('table').evaluateAll(tables => tables.map(t => [t.rows.length, t.rows[0].cells.length]));
                if (JSON.stringify(counts) !== JSON.stringify(expected)) throw new Error('missing table cells: ' + id);
                const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
                if (pageWidth > width + 1) throw new Error('page horizontal overflow: ' + id);
                const badImages = await section.locator('img').evaluateAll(images => images.filter(i => !i.complete || i.naturalWidth === 0).map(i => i.src));
                if (badImages.length) throw new Error('missing images: ' + badImages.join(','));
                const screenshot = path.join(output, id + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                receipts.push({route, id, width, screenshot, pageWidth, counts});
                if (width === 390) {
                    for (let index = 0; index < expected.length; index++) {
                        const table = section.locator('table').nth(index);
                        const right = await table.evaluate(t => {
                            const p = t.parentElement;
                            p.scrollLeft = p.scrollWidth;
                            return {offset: p.scrollLeft, lastRight: t.rows[0].cells[t.rows[0].cells.length - 1].getBoundingClientRect().right,
                                containerRight: p.getBoundingClientRect().right};
                        });
                        if (right.lastRight > right.containerRight + 1) throw new Error('inaccessible right column: ' + id);
                        const screenshot = path.join(output, id + '-390-table-' + index + '-right.png');
                        await table.locator('..').screenshot({path: screenshot});
                        receipts.push({route, id, width, screenshot, right});
                    }
                }
            }
        }
        if (errors.length) throw new Error(errors.join('\n'));
        const result = {passed: true, browser: await browser.version(), receipts};
        fs.writeFileSync(path.join(__dirname, 'docs-qa.json'), JSON.stringify(result, null, 2) + '\n');
        console.log(JSON.stringify({passed: result.passed, browser: result.browser, views: receipts.length}));
    } finally {
        await browser.close();
    }
})().catch(e => { console.error(e); process.exitCode = 1; });
