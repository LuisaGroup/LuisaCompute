// node qa-docs.cjs <built-html-root> <new-output-directory> <playwright-module>
const {chromium} = require(process.argv[4] || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
    fs.mkdirSync(output);
    const browser = await chromium.launch({headless: true,
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, required, rows, columns] of [
                ['source/performance/tile/migration.html', 'pure-entry-follow-up-generic-map-fusion',
                    ['same current compiler', 'byte-identical', 'v4', 'calibrated cost-model victory'], 9, 4],
                ['source/performance/tile/migration.html', 'full-baseline-current-matrix',
                    ['Error', 'Current XIR/SIMD', '4096'], 39, 5],
                ['source/tile/values.html', 'logical-exchange-versus-physical-shuffle',
                    ['cross-worker communication', 'reindex', 'gather'], 0, 0],
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
                if (rows && (geometry.tables[0]?.rows !== rows || geometry.tables[0]?.columns !== columns)) {
                    throw Error('incomplete table: ' + id);
                }
                const screenshot = id + '-' + width + '.png';
                await page.screenshot({path: path.join(output, screenshot)});
                if (rows) {
                    const table = section.locator('table').first();
                    await table.scrollIntoViewIfNeeded();
                    const visible = await table.evaluate(t => {
                        const wrapper = t.parentElement;
                        wrapper.scrollLeft = wrapper.scrollWidth;
                        const cell = t.rows[0].cells[t.rows[0].cells.length - 1].getBoundingClientRect();
                        const bounds = wrapper.getBoundingClientRect();
                        return cell.right <= bounds.right + 1 && cell.left < bounds.right;
                    });
                    if (!visible) throw Error('rightmost column is unreachable: ' + id);
                    await page.screenshot({path: path.join(output, id + '-' + width + '-table-right.png')});
                }
                receipts.push({file, id, width, pageWidth: geometry.width,
                    tables: geometry.tables, required, screenshot});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'),
            JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: map-fusion evidence, original matrix and exchange semantics render at desktop/mobile widths');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
