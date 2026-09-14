const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
    fs.mkdirSync(output, {recursive: true});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, required] of [
                ['source/performance/tile/validation.html', 'expression-producers-join-their-first-reduction-traversal',
                    ['35 Tile CTests', '79 XIR/SIMD CTests', 'No new performance claim is accepted', '96 visits']],
                ['source/internals/tile/xir.html', 'first-consumer-fusion-preserves-the-snapshot-contract',
                    ['enable_expression_reduction_fusion', 'default-disabled', 'No reciprocal rewrite']],
                ['source/performance/tile/index.html', 'results-by-route',
                    ['expression/reduction fusion checkpoint', 'not', 'speedup claim']],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({width: document.documentElement.scrollWidth, text: s.innerText}));
                if (geometry.width > width + 1 || required.some(text => !geometry.text.includes(text))) {
                    throw Error('missing/overflowed section: ' + id);
                }
                if (file.includes('internals')) {
                    await section.getByText('The independent', {exact: false}).scrollIntoViewIfNeeded();
                }
                const screenshot = path.join(output, id + '-' + width + '.png');
                await page.screenshot({path: screenshot});
                const fullScreenshot = path.join(output, id + '-' + width + '-full.png');
                await section.screenshot({path: fullScreenshot});
                receipts.push({file, id, width, pageWidth: geometry.width, required, screenshot, fullScreenshot});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'), JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: all three changed sections render at desktop/mobile widths without page overflow or browser errors');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
