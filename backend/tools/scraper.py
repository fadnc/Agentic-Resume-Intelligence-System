"""
backend/tools/scraper.py
Fetches and cleans job descriptions from LinkedIn, Naukri, or generic URLs.
Uses requests + BeautifulSoup; Playwright path for JS-heavy pages.
"""
import re
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _extract_linkedin(soup: BeautifulSoup) -> str:
    el = soup.find("div", {"class": re.compile(r"description|job-description", re.I)})
    return el.get_text(separator="\n").strip() if el else soup.get_text(separator="\n").strip()


def _extract_naukri(soup: BeautifulSoup) -> str:
    el = soup.find("div", {"class": re.compile(r"job-desc|description|jd-container", re.I)})
    return el.get_text(separator="\n").strip() if el else soup.get_text(separator="\n").strip()


def _extract_generic(soup: BeautifulSoup) -> str:
    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()
    for sel in ["job-description", "description", "jd", "job-detail", "posting"]:
        el = soup.find(attrs={"class": re.compile(sel, re.I)})
        if el:
            return el.get_text(separator="\n").strip()
    return soup.get_text(separator="\n").strip()


def _clean(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 2]
    seen, out = set(), []
    for l in lines:
        if l.lower() not in seen:
            seen.add(l.lower())
            out.append(l)
    return "\n".join(out)


def scrape_jd(url: str, use_playwright: bool = False) -> str:
    url = url.strip()
    if use_playwright:
        return _playwright_scrape(url)

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    if "linkedin.com" in url:
        return _clean(_extract_linkedin(soup))
    if "naukri.com" in url:
        return _clean(_extract_naukri(soup))
    return _clean(_extract_generic(soup))


def _playwright_scrape(url: str) -> str:
    import time
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(extra_http_headers=HEADERS)
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(2)
        try:
            page.click('[aria-label="Click to see more description"]', timeout=2000)
        except Exception:
            pass
        html = page.content()
        browser.close()
    soup = BeautifulSoup(html, "html.parser")
    return _clean(_extract_linkedin(soup) if "linkedin.com" in url else _extract_generic(soup))