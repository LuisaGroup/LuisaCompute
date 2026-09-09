// node qa-docs.cjs <built-html-root> <new-output-directory> <playwright-module>
const {chromium} = require(process.argv[4] || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
    fs.mkdirSync(output); // Refuse to overwrite an earlier QA visit.
    const browser = await chromium.launch({headless: true,
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, required] of [
                ['source/performance/tile/migration.html', 'full-baseline-current-matrix',
                    ['Error', 'Current XIR/SIMD', '4096']],
                ['source/tile/values.html', 'scan-preserves-prefixes-not-a-new-execution-hierarchy',
                    ['not an implemented builtin yet', 'ordered tree', 'commutativity is not required']],
                ['source/tile/values.html', 'logical-exchange-versus-physical-shuffle',
                    ['cross-worker communication', 'reindex', 'gather']],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({
                    width: document.documentElement.scrollWidth, text: s.innerText,
                    tables: Array.from(s.querySelectorAll('table')).map(t => ({
                        columns: t.rows[0]?.cells.length, rows: t.rows.length,
                        scrollWidth: t.parentElement.scrollWidth, clientWidth: t.parentElement.clientWidth
                    }))
                }));
                if (geometry.width > width + 1 || required.some(text => !geometry.text.includes(text))) {
                    throw Error('missing/overflowed content: ' + id);
                }
                if (id === 'full-baseline-current-matrix' &&
                    (geometry.tables[0]?.rows !== 39 || geometry.tables[0]?.columns !== 5)) {
                    throw Error('incomplete matrix table');
                }
                const screenshot = id + '-' + width + '.png';
                await page.screenshot({path: path.join(output, screenshot)});
                // Wide RTD tables deliberately scroll inside their container;
                // verify the rightmost current-version data is reachable too.
                if (id === 'full-baseline-current-matrix') {
                    const visible = await section.locator('table').first().evaluate(t => {
                        const wrapper = t.parentElement;
                        wrapper.scrollLeft = wrapper.scrollWidth;
                        const cell = t.rows[0].cells[t.rows[0].cells.length - 1].getBoundingClientRect();
                        const bounds = wrapper.getBoundingClientRect();
                        return cell.right <= bounds.right + 1 && cell.left < bounds.right;
                    });
                    if (!visible) throw Error('rightmost matrix column is unreachable');
                    await page.screenshot({path: path.join(output, id + '-' + width + '-right.png')});
                }
                receipts.push({file, id, width, pageWidth: geometry.width,
                    tables: geometry.tables, required, screenshot});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'),
            JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: scan/exchange design and complete matrix render at desktop/mobile widths');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
