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
                ['source/performance/tile/results.html', 'whole-group-mpp-participation-is-not-uniformly-better', 1],
                ['source/performance/tile/results.html', 'generic-traversal-composes-with-k-but-is-not-a-universal-win', 1],
                ['source/internals/tile/planner.html', 'program-traversal-is-a-mapping-choice-not-a-memory-scope', 0],
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
                if (metrics.pageWidth > width + 1) throw new Error('page-level horizontal overflow');
                const screenshot = path.join(output, sectionId + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                receipts.push({file, sectionId, width, screenshot, ...metrics});
            }
        }
        if (errors.length) throw new Error(errors.join('\n'));
        console.log(JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2));
    } finally {
        await browser.close();
    }
})().catch(e => { console.error(e); process.exitCode = 1; });
