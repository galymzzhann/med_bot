# src/scrap.py
"""
Structured scraper for diseases.medelement.com

Key improvements over the original:
  1. Parses HTML structure instead of grabbing raw .text
  2. Splits content into canonical sections (definition, symptoms, diagnostics, …)
  3. Strips junk (disclaimers, navigation, footers) at scrape time
  4. Saves each disease as a structured JSON file with full metadata
  5. Supports resuming — skips already-scraped files

Output format per file (data/scraped_json/<sanitized_title>.json):
{
    "title":    "Бронхиальная астма",
    "url":      "https://diseases.medelement.com/…",
    "scraped_at": "2025-…",
    "sections": {
        "definition":  "…clean text…",
        "symptoms":    "…",
        "diagnostics": "…",
        "treatment":   "…",
        "_full":       "…entire clean text as fallback…"
    }
}
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone

import yaml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    src  = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(src)
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("scrap")

cfg  = _load_config()
ROOT = _project_root()

BASE_URL      = cfg["scrape"]["base_url"]
OUTPUT_DIR    = os.path.join(ROOT, cfg["data"]["scraped_dir"])
WAIT_TIMEOUT  = cfg["scrape"]["wait_timeout"]
PAGE_PAUSE    = cfg["scrape"]["page_pause"]
SCROLL_PAUSE  = cfg["scrape"]["scroll_pause"]
SELECTORS     = cfg["scrape"]["content_selectors"]
SECTION_MAP   = cfg["scrape"]["section_map"]      # canonical → [patterns]
JUNK_PHRASES  = cfg["scrape"]["junk_phrases"]


# ── Driver ────────────────────────────────────────────────────────────────────

def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name)
    return name[:180]


def scroll_to_load_all(driver: webdriver.Chrome, link_selector: str,
                       expected_min: int = 0, max_stale_rounds: int = 8) -> None:
    """
    Scroll repeatedly until no new links appear for *max_stale_rounds*
    consecutive attempts.  This handles sites that lazy-load results in
    batches — the page height may briefly stop growing while the next
    batch is being fetched, so a single stale check is not enough.

    Also clicks any "load more" / "show more" button if one exists.
    """
    LOAD_MORE_SELECTORS = [
        "button.load-more", "a.load-more",
        "button.show-more", "a.show-more",
        ".pagination__next", ".next-page",
        "[data-action='load-more']",
    ]

    prev_count    = 0
    stale_rounds  = 0

    while stale_rounds < max_stale_rounds:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)

        # Try clicking a "load more" button if one exists
        for sel in LOAD_MORE_SELECTORS:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, sel)
                if btn.is_displayed():
                    btn.click()
                    logger.info(f"Clicked load-more button: {sel}")
                    time.sleep(PAGE_PAUSE)
                    break
            except Exception:
                pass

        # Count how many links are on the page now
        current_count = len(driver.find_elements(By.CSS_SELECTOR, link_selector))

        if current_count > prev_count:
            logger.info(f"  … loaded {current_count} links so far")
            prev_count   = current_count
            stale_rounds = 0          # reset — we're still growing
        else:
            stale_rounds += 1
            # Wait a bit longer on stale rounds to give the server time
            time.sleep(SCROLL_PAUSE * 1.5)

        # Early exit if we've clearly loaded everything
        if expected_min > 0 and current_count >= expected_min:
            logger.info(f"  … reached expected minimum ({expected_min})")
            break

    logger.info(f"Scrolling done — {prev_count} links visible on page")


def _is_junk_line(line: str) -> bool:
    """Check if a line is boilerplate / disclaimer / navigation noise."""
    low = line.lower().strip()
    if not low:
        return True
    if len(low) < 5:
        return True
    return any(phrase in low for phrase in JUNK_PHRASES)


def _clean_text(raw: str) -> str:
    """Remove junk lines and normalise whitespace."""
    lines = raw.splitlines()
    clean = [ln.strip() for ln in lines if not _is_junk_line(ln)]
    text  = "\n".join(clean)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _match_section(header_text: str) -> str | None:
    """
    Given a header string from the page, return the canonical section name
    if it matches any pattern in SECTION_MAP, else None.
    """
    h = header_text.lower().strip()
    for canonical, patterns in SECTION_MAP.items():
        for pat in patterns:
            if pat in h:
                return canonical
    return None


# ── Structured extraction ─────────────────────────────────────────────────────

def _extract_sections_from_elements(driver: webdriver.Chrome) -> dict[str, str]:
    """
    Walk through the content area and split text by header elements.
    Returns {"definition": "...", "symptoms": "...", ...}
    plus a special "_full" key with the entire cleaned text.
    """
    # Find the best content container
    container = None
    for sel in SELECTORS:
        elems = driver.find_elements(By.CSS_SELECTOR, sel)
        for el in elems:
            if len(el.text.strip()) > 200:
                container = el
                break
        if container:
            break

    if not container:
        container = driver.find_element(By.TAG_NAME, "body")

    full_text = _clean_text(container.text)

    # Try to find headers (h1–h4, strong, b) inside the container
    headers = container.find_elements(
        By.CSS_SELECTOR, "h1, h2, h3, h4, h5, .block-title, strong, b"
    )

    # Build ordered list of (position_in_full_text, canonical_section_name, header_text)
    anchors: list[tuple[int, str, str]] = []
    for hdr in headers:
        hdr_text = hdr.text.strip()
        if not hdr_text or len(hdr_text) > 200:
            continue
        canonical = _match_section(hdr_text)
        if canonical is None:
            continue
        # Find where this header appears in the full text
        pos = full_text.lower().find(hdr_text.lower())
        if pos >= 0:
            anchors.append((pos, canonical, hdr_text))

    # Deduplicate and sort by position
    anchors.sort(key=lambda x: x[0])
    seen_sections: set[str] = set()
    unique_anchors: list[tuple[int, str, str]] = []
    for pos, canon, txt in anchors:
        if canon not in seen_sections:
            seen_sections.add(canon)
            unique_anchors.append((pos, canon, txt))

    # Split full_text into sections
    sections: dict[str, str] = {}
    for i, (pos, canon, hdr_txt) in enumerate(unique_anchors):
        start = pos + len(hdr_txt)
        end   = unique_anchors[i + 1][0] if i + 1 < len(unique_anchors) else len(full_text)
        chunk = full_text[start:end].strip()
        if chunk:
            sections[canon] = _clean_text(chunk)

    # Always include the full cleaned text as a fallback
    sections["_full"] = full_text

    return sections


# ── Link collection ──────────────────────────────────────────────────────────

def collect_links(driver: webdriver.Chrome) -> list[dict]:
    LINK_SEL = "a.results-item__title-link"

    logger.info("Opening index page …")
    driver.get(BASE_URL)

    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, LINK_SEL)))
    time.sleep(PAGE_PAUSE)

    logger.info("Scrolling to load all items (this may take a few minutes) …")
    scroll_to_load_all(driver, link_selector=LINK_SEL, expected_min=1100)
    time.sleep(PAGE_PAUSE)

    elements = driver.find_elements(By.CSS_SELECTOR, LINK_SEL)
    logger.info(f"Found {len(elements)} links")

    links: list[dict] = []
    seen:  set[str]   = set()
    for el in elements:
        href  = el.get_attribute("href") or ""
        title = el.text.strip()
        if href and href not in seen and title:
            seen.add(href)
            links.append({"title": title, "url": href})

    return links


# ── Page scraping ─────────────────────────────────────────────────────────────

def scrape_page(driver: webdriver.Chrome, url: str) -> dict[str, str]:
    """Navigate to a disease page and return structured sections dict."""
    driver.get(url)
    time.sleep(PAGE_PAUSE)
    # Simple scroll for individual pages (not the index)
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE)
    time.sleep(1)
    return _extract_sections_from_elements(driver)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    driver = make_driver()

    try:
        links = collect_links(driver)
        if not links:
            logger.warning("No links found. Site structure may have changed.")
            return

        for idx, item in enumerate(links, start=1):
            title = item["title"]
            url   = item["url"]
            safe  = sanitize_filename(title)
            fpath = os.path.join(OUTPUT_DIR, f"{safe}.json")

            # Resume support — skip files that already have good content
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    full_len = len(existing.get("sections", {}).get("_full", ""))
                    if full_len >= 200:
                        logger.info(f"[{idx:03d}/{len(links)}] SKIP (exists, {full_len:,} chars): {safe}")
                        continue
                    else:
                        logger.info(f"[{idx:03d}/{len(links)}] RE-SCRAPE (only {full_len} chars): {safe}")
                        os.remove(fpath)
                except (json.JSONDecodeError, KeyError):
                    logger.info(f"[{idx:03d}/{len(links)}] RE-SCRAPE (corrupt file): {safe}")
                    os.remove(fpath)

            logger.info(f"[{idx:03d}/{len(links)}] {title}")

            try:
                sections = scrape_page(driver, url)
                full_len = len(sections.get("_full", ""))

                # If page loaded with almost no content, retry once
                if full_len < 200:
                    logger.warning(f"  → only {full_len} chars, retrying in 5s …")
                    time.sleep(5)
                    sections = scrape_page(driver, url)
                    full_len = len(sections.get("_full", ""))

                doc = {
                    "title":      title,
                    "url":        url,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "sections":   sections,
                }

                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)

                n_sections = len([k for k in sections if k != "_full"])
                full_len   = len(sections.get("_full", ""))
                logger.info(f"  → saved {n_sections} sections, {full_len:,} chars total")

            except Exception:
                logger.exception(f"Failed to scrape: {url}")

    finally:
        driver.quit()
        logger.info(f"Done. Files saved in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()