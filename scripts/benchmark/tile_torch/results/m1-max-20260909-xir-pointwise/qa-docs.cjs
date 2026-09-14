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
                ['source/internals/tile/xir.html', 'guarded-pointwise-dag-fusion-keeps-an-alias-safe-fallback', 'not kernel or operator names'],
                ['source/performance/tile/index.html', 'results-by-route', 'not yet a measured performance'],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({width: document.documentElement.scrollWidth, text: s.innerText}));
                if (geometry.width > width + 1 || !geometry.text.includes(required)) throw Error('missing/overflowed content: ' + id);
                const screenshot = path.join(output, id + '-' + width + '.png');
                await page.screenshot({path: screenshot});
                const fullScreenshot = path.join(output, id + '-' + width + '-full.png');
                await section.screenshot({path: fullScreenshot});
                receipts.push({file, id, width, pageWidth: geometry.width, screenshot, fullScreenshot});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'), JSON.stringify({passed: true, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: all four actual Sphinx sections render at desktop/mobile widths, no page overflow or browser errors');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
