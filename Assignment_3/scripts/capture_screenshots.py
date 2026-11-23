"""
Drive the Project 5 Streamlit app in a real browser and save screenshots.

Usage (with the app already running on http://localhost:8501):
    python scripts/capture_screenshots.py

Uses Playwright with the locally installed Microsoft Edge (channel="msedge"),
so no extra browser download is needed. Use channel="chrome" if you prefer.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "project5_live_training_dashboard" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def wait_finished(page: Page, run: int, timeout_s: int = 240) -> None:
    # the app shows a toast "Run N finished: ..." when training ends
    page.get_by_text(f"Run {run} finished").first.wait_for(timeout=timeout_s * 1000)
    page.wait_for_timeout(2500)  # let the charts settle


def choose_preset(page: Page, name: str) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    page.wait_for_timeout(2000)  # let any rerun finish first
    sidebar.locator('[data-testid="stSelectbox"]').first.click()
    page.keyboard.type(name.split()[0])  # filter the list, then pick the match
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    page.wait_for_timeout(1500)


def train(page: Page) -> None:
    page.get_by_test_id("stBaseButton-primary").click()
    page.wait_for_timeout(1500)


def shot(page: Page, name: str, full: bool = True) -> None:
    path = OUT / name
    page.screenshot(path=str(path), full_page=full)
    print("saved", path)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})

        # 1. Overfit preset (the default) - captured mid-training, then finished.
        page.goto(f"{URL}/?autotrain=1")
        page.get_by_text("Training · epoch").first.wait_for(timeout=120_000)
        page.wait_for_timeout(3000)
        shot(page, "01_training_live.png", full=False)
        wait_finished(page, 1)
        shot(page, "02_overfit.png", full=False)

        # 2. Underfit preset
        choose_preset(page, "Underfit (too small, too slow)")
        train(page)
        wait_finished(page, 2)
        shot(page, "03_underfit.png", full=False)

        # 3. Fixed preset
        choose_preset(page, "Fixed (regularised + early stopping)")
        train(page)
        wait_finished(page, 3)
        shot(page, "04_fixed.png")  # full page shows the run history table

        # 4. Screen patients tab
        page.get_by_role("tab", name="Screen patients").click()
        page.wait_for_timeout(3000)
        shot(page, "05_screen_patients.png")

        # 5. About & limitations
        page.get_by_role("tab", name="About").click()
        page.wait_for_timeout(1500)
        shot(page, "06_about_limitations.png", full=False)

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
