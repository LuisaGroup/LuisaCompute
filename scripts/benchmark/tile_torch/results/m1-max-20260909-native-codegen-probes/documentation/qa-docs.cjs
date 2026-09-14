const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]), output = path.resolve(process.argv[3]);
    if (fs.existsSync(path.join(output, 'receipt.json'))) throw Error('Use a new QA destination');
    fs.mkdirSync(output, {recursive: true});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    const receipts = [], errors = [];
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, required] of [
                ['source/performance/tile/validation.html', 'native-codegen-prototypes-remain-separate-from-production-promotion',
                    ['80 XIR/SIMD CTests and 35 Tile CTests', '384 retained output snapshots', 'not', 'diagnostic-only']],
                ['source/performance/tile/results.html', 'late-native-codegen-probes-separate-address-demand-from-inlining',
                    ['Both complete timing cohorts are diagnostic-only', '864 timed native visits', 'remain outside the working compiler']],
                ['source/performance/tile/index.html', 'results-by-route',
                    ['native codegen probes', '0.337–0.469', 'not promoted into the working compiler']],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                const geometry = await section.evaluate(s => ({width: document.documentElement.scrollWidth, text: s.innerText}));
                if (geometry.width > width + 1 || required.some(text => !geometry.text.includes(text))) {
                    throw Error('Missing content or page overflow: ' + id + ' / ' + width);
                }
                const screenshot = path.join(output, id + '-' + width + '.png');
                await page.screenshot({path: screenshot});
                const fullScreenshot = path.join(output, id + '-' + width + '-full.png');
                await section.screenshot({path: fullScreenshot});
                receipts.push({file, id, width, pageWidth: geometry.width, required, screenshot, fullScreenshot});
            }
        }
        if (errors.length) throw Error(errors.join('\n'));
        fs.writeFileSync(path.join(output, 'receipt.json'), JSON.stringify({passed: true, nodeExecutable: process.execPath, browser: await browser.version(), receipts}, null, 2) + '\n');
        console.log('PASS: three sections render at desktop/mobile widths without page overflow or browser errors');
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
