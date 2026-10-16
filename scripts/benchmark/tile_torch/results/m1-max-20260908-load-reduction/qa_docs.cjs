// Validate the existing Sphinx surface, not a second report renderer.
const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const path = require('node:path');
const fs = require('node:fs');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    fs.mkdirSync(output, {recursive: true});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', e => errors.push(e.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, rows, columns] of [
                ['source/performance/tile/results.html', 'simd-local-distribution-and-private-layout-are-separate-decisions', 5, 6],
                ['source/internals/tile/xir.html', 'first-consumer-fusion-preserves-the-snapshot-contract', 0, 0],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({
                    width: innerWidth, page: document.documentElement.scrollWidth,
                    text: s.innerText, tables: [...s.querySelectorAll('table')].map(t => [t.rows.length, t.rows[0].cells.length]),
                }));
                if (geometry.page > width + 1 || !geometry.text.trim()) throw Error('missing/overflowed content: ' + id);
                if (JSON.stringify(geometry.tables) !== JSON.stringify(rows ? [[rows, columns]] : [])) throw Error('missing table cells: ' + id);
                const screenshot = path.join(output, id + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                const scrolls = await section.evaluate(s => [...s.querySelectorAll('table, pre')].flatMap(item => {
                    let scroller = item;
                    while (scroller && scroller !== s &&
                           !(scroller.scrollWidth > scroller.clientWidth + 1 &&
                             /auto|scroll/.test(getComputedStyle(scroller).overflowX))) {
                        scroller = scroller.parentElement;
                    }
                    if (!scroller || scroller === s) return [];
                    scroller.scrollLeft = scroller.scrollWidth;
                    const maximum = scroller.scrollWidth - scroller.clientWidth;
                    if (Math.abs(scroller.scrollLeft - maximum) > 1) throw Error('unreachable horizontal content');
                    if (item.tagName === 'TABLE') {
                        const cells = item.rows[0].cells;
                        if (cells[cells.length - 1].getBoundingClientRect().right >
                            scroller.getBoundingClientRect().right + 2) throw Error('unreachable final table column');
                    }
                    return [{element: item.tagName, maximum, reached: scroller.scrollLeft}];
                }));
                const scrolledScreenshot = scrolls.length ? path.join(output, id + '-' + width + '-right.png') : null;
                if (scrolledScreenshot) await section.screenshot({path: scrolledScreenshot});
                receipts.push({file, id, width, screenshot, scrolledScreenshot, scrolls, tables: geometry.tables, pageWidth: geometry.page});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'), JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: 4 rendered sections, reachable table/code content, no page overflow or browser errors');
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
