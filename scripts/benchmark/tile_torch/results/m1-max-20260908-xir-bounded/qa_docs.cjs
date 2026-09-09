// Verify the repository's own Sphinx pages, not a parallel report renderer.
const {chromium} = require('/Users/mike/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

(async () => {
    const root = path.resolve(process.argv[2]);
    const output = path.resolve(process.argv[3]);
    fs.mkdirSync(output, {recursive: false});
    const browser = await chromium.launch({headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        const errors = [], receipts = [];
        page.on('pageerror', error => errors.push(error.message));
        for (const width of [1280, 390]) {
            await page.setViewportSize({width, height: 900});
            for (const [file, id, expected] of [
                ['source/performance/tile/results.html', 'bounded-xir-traversal-improves-compilation-not-yet-torch-parity', 'Every one of'],
                ['source/internals/tile/xir.html', 'bounded-traversal-partial-reductions-and-private-resources', 'not a measured complete stack'],
            ]) {
                await page.goto(pathToFileURL(path.join(root, file)).href + '#' + id);
                await page.evaluate(() => document.fonts.ready);
                const section = page.locator('#' + id);
                if (!(await section.innerText()).includes(expected)) throw new Error('missing content: ' + id);
                const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
                if (pageWidth > width + 1) throw new Error('horizontal page overflow: ' + id);
                const scrollers = await section.locator('.wy-table-responsive, pre').evaluateAll(nodes => nodes.map(node => {
                    const css = getComputedStyle(node);
                    const before = node.scrollLeft;
                    node.scrollLeft = node.scrollWidth;
                    const reached = node.scrollLeft;
                    node.scrollLeft = before;
                    return {tag: node.tagName, width: node.clientWidth, scrollWidth: node.scrollWidth, overflowX: css.overflowX, reached};
                }));
                if (scrollers.some(item => item.scrollWidth > item.width + 1 &&
                    (!['auto', 'scroll'].includes(item.overflowX) || item.reached <= 0))) throw new Error('unreachable clipped content: ' + id);
                const screenshot = path.join(output, id + '-' + width + '.png');
                await section.screenshot({path: screenshot});
                receipts.push({file, id, width, pageWidth, screenshot, scrollers});
            }
        }
        if (errors.length) throw new Error(errors.join('\n'));
        const result = {passed: true, browser: await browser.version(), receipts};
        fs.writeFileSync(path.join(output, 'docs-qa.json'), JSON.stringify(result, null, 2) + '\n');
        console.log(JSON.stringify(result));
    } finally {
        await browser.close();
    }
})().catch(error => {console.error(error); process.exitCode = 1;});
