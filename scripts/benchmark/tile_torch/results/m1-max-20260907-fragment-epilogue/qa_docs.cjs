// Review the existing Sphinx surface; this is not a second report renderer.
const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        const errors = [];
        page.on('pageerror', e => errors.push(e.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, sectionId, tables] of [
                ['source/performance/tile/index.html', 'current-conclusion', 0],
                ['source/performance/tile/results.html', 'closed-matrix-epilogues-general-legality-mixed-profitability', 1],
                ['source/internals/tile/matrix.html', 'scalar-epilogues-use-the-same-element-owner', 0],
                ['source/performance/tile/validation.html', 'closed-matrix-epilogues-positive-and-fail-closed-coverage', 0],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + sectionId);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + sectionId);
                const tableCount = await section.locator('table').count();
                if (tableCount !== tables) throw new Error('wrong table count: ' + sectionId);
                const metrics = await section.evaluate(element => ({
                    pageWidth: document.documentElement.scrollWidth,
                    viewport: innerWidth,
                    tables: Array.from(element.querySelectorAll('table')).map(t => ({rows: t.rows.length, columns: t.rows[0].cells.length,
                        width: t.getBoundingClientRect().width, scrollWidth: t.parentElement.scrollWidth, containerWidth: t.parentElement.clientWidth})),
                }));
                if (tables && (metrics.tables[0].rows !== 13 || metrics.tables[0].columns !== 7)) throw new Error('missing result cells');
                if (metrics.pageWidth > width + 1) throw new Error('page-level horizontal overflow');
                const screenshot = path.join(output, sectionId + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                receipts.push({file, sectionId, width, screenshot, ...metrics});
                if (width === 390 && tables) {
                    const scrolled = await section.locator('table').evaluate(t => {
                        const p = t.parentElement;
                        p.scrollLeft = p.scrollWidth;
                        return {offset: p.scrollLeft, lastColumnRight: t.rows[0].cells[t.rows[0].cells.length - 1].getBoundingClientRect().right,
                            containerRight: p.getBoundingClientRect().right};
                    });
                    if (scrolled.offset <= 0 || scrolled.lastColumnRight > scrolled.containerRight + 1) throw new Error('last column inaccessible');
                    const rightScreenshot = path.join(output, sectionId + '-390-right.png');
                    await section.locator('table').locator('..').screenshot({path: rightScreenshot});
                    receipts.push({file, sectionId, width, screenshot: rightScreenshot, scrolled});
                }
            }
        }
        if (errors.length) throw new Error(errors.join('\n'));
        console.log(JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2));
    } finally {
        await browser.close();
    }
})().catch(e => { console.error(e); process.exitCode = 1; });
