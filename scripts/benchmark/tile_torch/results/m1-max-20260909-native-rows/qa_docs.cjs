// Exercise the actual Sphinx page at desktop/mobile widths, not a second renderer.
const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    fs.mkdirSync(output, {recursive: true});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, tables, required] of [
                ['results.html', 'native-row-entries-expose-both-broader-wins-and-remaining-gaps', [[7, 3]], '0.337–0.469'],
                ['index.html', 'results-by-route', [], 'three native operator families win; three still lose'],
            ]) {
                await page.goto(pathToFileURL(path.join(root, 'source/performance/tile', file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({
                    pageWidth: document.documentElement.scrollWidth, text: s.innerText,
                    tables: [...s.querySelectorAll('table')].map(t => [t.rows.length, t.rows[0].cells.length]),
                }));
                if (geometry.pageWidth > width + 1 || !geometry.text.includes(required)) throw Error('missing/overflowed content: ' + id);
                if (JSON.stringify(geometry.tables) !== JSON.stringify(tables)) throw Error('missing table cells: ' + id);
                const screenshot = path.join(output, id + '-' + width + '.png');
                await page.screenshot({path: screenshot});
                const fullScreenshot = path.join(output, id + '-' + width + '-full.png');
                await section.screenshot({path: fullScreenshot});
                const scrolls = await section.evaluate(s => [...s.querySelectorAll('table')].flatMap(item => {
                    let scroller = item;
                    while (scroller && scroller !== s &&
                           !(scroller.scrollWidth > scroller.clientWidth + 1 && /auto|scroll/.test(getComputedStyle(scroller).overflowX))) {
                        scroller = scroller.parentElement;
                    }
                    if (!scroller || scroller === s) return [];
                    scroller.scrollLeft = scroller.scrollWidth;
                    const maximum = scroller.scrollWidth - scroller.clientWidth;
                    if (Math.abs(scroller.scrollLeft - maximum) > 1) throw Error('unreachable table content');
                    const cells = item.rows[0].cells;
                    if (cells[cells.length - 1].getBoundingClientRect().right > scroller.getBoundingClientRect().right + 2) {
                        throw Error('unreachable last column');
                    }
                    return [{maximum, reached: scroller.scrollLeft}];
                }));
                const rightScreenshot = scrolls.length ? path.join(output, id + '-' + width + '-right.png') : null;
                if (rightScreenshot) await section.screenshot({path: rightScreenshot});
                receipts.push({file, id, width, tables: geometry.tables, pageWidth: geometry.pageWidth,
                               screenshot, fullScreenshot, rightScreenshot, scrolls});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'), JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: 4 rendered sections; current values and table cells present; final column reachable; no page overflow/browser errors');
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
