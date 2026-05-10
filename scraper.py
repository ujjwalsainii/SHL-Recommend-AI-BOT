"""
SHL Catalog Scraper - Run this LOCALLY (not on cloud servers).
Uses Playwright to handle JS-rendered pages and bypass bot detection.

Setup:
  pip install playwright beautifulsoup4 requests
  playwright install chromium
  python scraper.py
"""

import asyncio, json, re, time
from pathlib import Path
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/products/product-catalog/"


# ── HTML parsers ──────────────────────────────────────────────────────────────

def parse_catalog_page(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    table = soup.find("table")
    if not table:
        for a in soup.find_all("a", href=re.compile(r"/product-catalog/view/")):
            name = a.get_text(strip=True)
            if name:
                href = a["href"]
                items.append({"name": name, "url": href if href.startswith("http") else BASE_URL + href})
        return items

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        a = cells[0].find("a", href=re.compile(r"/product-catalog/view/"))
        if not a:
            continue
        href = a["href"]
        url = href if href.startswith("http") else BASE_URL + href

        def has_check(cell):
            txt = cell.get_text(strip=True)
            return bool(cell.find("img")) or txt in ("✓","●","•","Yes","yes") or \
                   any("yes" in " ".join(s.get("class", [])).lower() for s in cell.find_all("span"))

        items.append({
            "name": a.get_text(strip=True),
            "url": url,
            "remote_testing": has_check(cells[1]) if len(cells) > 1 else False,
            "adaptive_irt":   has_check(cells[2]) if len(cells) > 2 else False,
            "test_type":      cells[3].get_text(strip=True) if len(cells) > 3 else "",
        })
    return items


def parse_detail_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    data = {"url": url}
    h1 = soup.find("h1")
    data["name"] = h1.get_text(strip=True) if h1 else ""

    sections = {}
    for h4 in soup.find_all("h4"):
        label = h4.get_text(strip=True).lower().rstrip(":")
        sib = h4.find_next_sibling()
        text = sib.get_text(strip=True) if sib else ""
        if not text:
            text = re.sub(re.escape(h4.get_text(strip=True)), "",
                          (h4.parent or h4).get_text(" ", strip=True)).strip()
        sections[label] = text

    data["description"]    = sections.get("description", "")
    data["job_levels"]     = [j.strip() for j in sections.get("job levels","").split(",") if j.strip()]
    data["languages"]      = [l.strip() for l in sections.get("languages","").split(",")  if l.strip()]
    m = re.search(r"(\d+)", sections.get("assessment length",""))
    data["duration_minutes"] = int(m.group(1)) if m else None

    full = soup.get_text()
    tm = re.search(r"Test Type[:\s]+([A-Z])", full)
    data["test_type"] = tm.group(1) if tm else data.get("test_type","")
    rm = re.search(r"Remote Testing.*?(Yes|No|✓|✗)", full, re.IGNORECASE|re.DOTALL)
    data["remote_testing"] = rm and rm.group(1).strip() in ("Yes","yes","✓")
    return data


# ── Playwright scraper ────────────────────────────────────────────────────────

async def scrape_playwright():
    from playwright.async_api import async_playwright
    items, seen = [], set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx  = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"))
        page = await ctx.new_page()

        start = 0
        while True:
            url = f"{CATALOG_BASE}?action_doFilteringForm=Search&f=1&start={start}&type=1"
            print(f"  Listing start={start} …", end=" ")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            new = [i for i in parse_catalog_page(await page.content()) if i["url"] not in seen]
            for i in new: seen.add(i["url"])
            items.extend(new)
            print(f"+{len(new)}  (total {len(items)})")
            if not new or not await page.query_selector("a:has-text('Next')"):
                break
            start += 12
            await asyncio.sleep(0.5)

        print(f"\nFetching {len(items)} detail pages …")
        for idx, item in enumerate(items):
            print(f"  [{idx+1}/{len(items)}] {item['name']}")
            try:
                await page.goto(item["url"], wait_until="domcontentloaded", timeout=20000)
                detail = parse_detail_page(await page.content(), item["url"])
                for k, v in item.items():
                    if not detail.get(k): detail[k] = v
                items[idx] = detail
            except Exception as e:
                print(f"    Error: {e}")
            await asyncio.sleep(0.3)

        await browser.close()
    return items


# ── Requests fallback ─────────────────────────────────────────────────────────

def scrape_requests():
    import requests
    hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    items, seen = [], set()
    start = 0
    while True:
        url = f"{CATALOG_BASE}?action_doFilteringForm=Search&f=1&start={start}&type=1"
        print(f"  Listing start={start} …", end=" ")
        try:
            r = requests.get(url, headers=hdrs, timeout=15); r.raise_for_status()
        except Exception as e:
            print(f"Failed: {e}"); break
        new = [i for i in parse_catalog_page(r.text) if i["url"] not in seen]
        for i in new: seen.add(i["url"])
        items.extend(new)
        print(f"+{len(new)}  (total {len(items)})")
        if not new: break
        start += 12; time.sleep(0.5)

    print(f"\nFetching {len(items)} detail pages …")
    for idx, item in enumerate(items):
        print(f"  [{idx+1}/{len(items)}] {item['name']}")
        try:
            r = requests.get(item["url"], headers=hdrs, timeout=15); r.raise_for_status()
            detail = parse_detail_page(r.text, item["url"])
            for k, v in item.items():
                if not detail.get(k): detail[k] = v
            items[idx] = detail
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.3)
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== SHL Catalog Scraper ===\n")
    Path("data").mkdir(exist_ok=True)

    try:
        import playwright  # noqa
        print("Playwright found — using headless Chromium\n")
        items = asyncio.run(scrape_playwright())
    except ImportError:
        print("Playwright not installed — falling back to requests\n")
        items = scrape_requests()

    out = Path("data/catalog.json")
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"\n✓  Saved {len(items)} assessments → {out}")

    by_type = {}
    for it in items:
        t = it.get("test_type") or "?"
        by_type[t] = by_type.get(t, 0) + 1
    labels = {"A":"Ability","B":"Biodata/SJT","C":"Competencies","D":"Dev/360",
              "E":"Exercises","K":"Knowledge","P":"Personality","S":"Simulations"}
    print("\nBreakdown:")
    for t, n in sorted(by_type.items()):
        print(f"  {t} – {labels.get(t,'?')}: {n}")

if __name__ == "__main__":
    main()
