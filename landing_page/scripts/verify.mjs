import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
const browser = await chromium.launch({
  executablePath:
    "C:/Users/kesha/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe",
  headless: true,
});
const page = await browser.newPage({
  viewport: { width: 1440, height: 1000 },
  reducedMotion: "reduce",
});
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await mkdir("artifacts", { recursive: true });
await page.goto("http://localhost:5173");
await page.waitForTimeout(700);
await page.screenshot({ path: "artifacts/desktop.png", fullPage: true });
for (const width of [1440, 1024, 768, 390, 320]) {
  await page.setViewportSize({ width, height: 900 });
  await page.waitForTimeout(100);
  const overflow = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
    overflow: [...document.querySelectorAll("body *")]
      .filter((e) => e.getBoundingClientRect().right > innerWidth + 1)
      .map((e) => e.className)
      .slice(0, 10),
  }));
  console.log("layout", JSON.stringify(overflow));
  if (overflow.scroll > width) throw Error("Horizontal overflow at " + width);
  if (width === 390)
    await page.screenshot({ path: "artifacts/mobile.png", fullPage: true });
}
await page.setViewportSize({ width: 1440, height: 1000 });
await page.getByRole("button", { name: "Play narration", exact: true }).click();
await page.waitForTimeout(400);
const before = Number(await page.getByRole("slider").inputValue());
await page.getByRole("button", { name: "Ask ConversIQ" }).click();
await page.waitForTimeout(1400);
const during = Number(await page.getByRole("slider").inputValue());
if (during - before > 1) throw Error("Narration did not pause");
await page.waitForTimeout(3300);
if (
  !(await page
    .getByRole("button", { name: "Pause narration", exact: true })
    .isVisible())
)
  throw Error("Narration did not resume");
console.log("interruption passed");
await page
  .getByRole("button", { name: "Pause narration", exact: true })
  .click();
await page.getByRole("button", { name: "Pages 7–22" }).first().click();
await page.getByRole("button", { name: "Open chapter", exact: true }).click();
await page.locator(".chapter-list button").nth(5).click();
if ((await page.locator(".book-content h3").textContent()) !== "The Portkey")
  throw Error("Chapter selection failed");
await page
  .getByRole("textbox", { name: "Ask about the document" })
  .fill("What is a Portkey?");
await page.getByRole("button", { name: "Send question", exact: true }).click();
await page.waitForTimeout(3400);
if (
  !(await page
    .locator(".message.ai")
    .last()
    .innerText()
    .then((t) => t.includes("predetermined destination")))
)
  throw Error("Chat answer failed");
await page.getByRole("button", { name: "Change narration speed" }).click();
await page.locator(".tree-chapter > button").first().click();
if (
  (await page
    .locator(".tree-chapter > button")
    .first()
    .getAttribute("aria-expanded")) !== "true"
)
  throw Error("Tree expansion failed");
await page
  .getByRole("button", { name: "Why was it important?", exact: false })
  .click();
await page
  .getByRole("button", { name: "Simulate interrupting narration" })
  .click();
await page.waitForTimeout(6300);
if (
  !(await page
    .getByRole("button", { name: "Simulate interrupting narration" })
    .isEnabled())
)
  throw Error("Voice showcase stuck");
await page.setViewportSize({ width: 390, height: 844 });
await page.getByRole("button", { name: "Open navigation" }).click();
await page
  .locator(".nav-links")
  .getByRole("link", { name: "Demo", exact: true })
  .click();
await page
  .locator(".mobile-tabs")
  .getByRole("button", { name: "Chapters", exact: true })
  .click();
await page.locator(".chapter-list button").nth(2).click();
if (!(await page.locator(".document-reader").isVisible()))
  throw Error("Mobile reader failed");
await page
  .locator(".mobile-tabs")
  .getByRole("button", { name: "Conversation", exact: true })
  .click();
if (
  !(await page
    .getByRole("textbox", { name: "Ask about the document" })
    .isVisible())
)
  throw Error("Mobile chat failed");
const missing = await page.evaluate(() =>
  [...document.querySelectorAll('a[href^="#"]')]
    .map((a) => a.getAttribute("href"))
    .filter((h) => h !== "#" && !document.querySelector(h)),
);
if (missing.length) throw Error("Broken anchors " + missing);
console.log(
  "All interaction checks passed; browser errors:",
  JSON.stringify(errors),
);
if (errors.length) throw Error(errors.join("\n"));
await page.setViewportSize({ width: 1440, height: 1000 });
await page.locator('.source-chip').first().click();
await page.keyboard.press('Escape');
if(await page.getByRole('dialog').count())throw Error('Source dialog did not dismiss');
await page.locator('input[type=file]').setInputFiles({name:'sample.txt',mimeType:'text/plain',buffer:Buffer.from('Sample document')});
if(!await page.getByRole('dialog').innerText().then(t=>t.includes('no file is uploaded')))throw Error('Upload simulation failed');
await page.getByRole('button',{name:'Continue exploring'}).click();
await page.evaluate(()=>window.scrollTo(0,0));
await page.screenshot({path:'artifacts/desktop.png',fullPage:true});
await page.locator('#demo').screenshot({path:'artifacts/workspace.png'});
await page.setViewportSize({width:390,height:844});
await page.locator('.mobile-tabs').getByRole('button',{name:'Document',exact:true}).click();
await page.evaluate(()=>window.scrollTo(0,0));
await page.screenshot({path:'artifacts/mobile.png',fullPage:true});
console.log('Upload simulation and keyboard dismissal passed.');
await browser.close();
