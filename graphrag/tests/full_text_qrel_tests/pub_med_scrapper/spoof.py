import os
from pathlib import Path
import requests
from urllib.parse import urljoin
from typing import Optional, Dict
from playwright.sync_api import sync_playwright

INPUT_FILE = "pmid_doi_map.json"
# OUTPUT_DIR = "pdfs"
BASE_URL = "https://sci-hub.se"
# USER_DATA_DIR = "./user_data"

def interactive_bypass(user_data_dir: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False
        )
        page = browser.new_page()
        page.goto("https://sci-hub.se")
        print("[INFO] Solve any challenge in the browser window.")
        input("Press Enter after solving and confirming the page has loaded...\n")
        browser.close()

def get_pdf_url_with_playwright(doi: str, user_data_dir: Path) -> Optional[str]:
    """Fetch the Sci-Hub page using Playwright and extract the direct PDF URL."""
    target_url = f"{BASE_URL}/{doi}"

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=True
        )
        page = browser.new_page()
        try:
            print(f"[INFO] Opening page: {target_url}")
            page.goto(target_url, timeout=60000)
            page.wait_for_load_state("networkidle")

            embed = page.query_selector("embed#pdf")
            if embed:
                src = embed.get_attribute("src")
                if src:
                    return urljoin(BASE_URL, src.split("#")[0])

            iframe = page.query_selector("iframe")
            if iframe:
                src = iframe.get_attribute("src")
                if src:
                    return urljoin(BASE_URL, src.split("#")[0])

            button = page.query_selector("button[onclick*='.pdf']")
            if button:
                onclick = button.get_attribute("onclick")
                if "location.href=" in onclick:
                    url = onclick.split("location.href=")[1].strip("'\"")
                    return urljoin(BASE_URL, url.split("?")[0])

            print("[WARN] PDF URL not found in the page.")
            return None

        finally:
            print("[INFO] Closing browser.")
            browser.close()

def download_pdf(pmid: str, pdf_url: str, pdfs_dir: Path) -> bool:
    """Download the PDF from the given URL and save it to disk."""
    try:
        output_path = os.path.join(pdfs_dir, f"{pmid}.pdf")
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[OK] Downloaded PDF for PMID {pmid}")
        return True
    except requests.RequestException as e:
        print(f"[ERROR] PMID {pmid}: Failed to download PDF: {e}")
        return False

def download_pmid_documents(pmid_to_doi: Dict[str, str], data_location: Path) -> Dict[str, Optional[str]]:
    user_data_dir = Path(os.path.join(data_location, "user_data"))

    pdfs_dir = Path(os.path.join(data_location, "pdfs"))
    if not os.path.isdir(pdfs_dir):
        os.makedirs(pdfs_dir)


    if not os.path.exists(user_data_dir):
        print("[INFO] User data directory not found. Launching interactive bypass...")
        interactive_bypass(user_data_dir)

    results: Dict[str, Optional[str]] = {}
    for pmid, doi in pmid_to_doi.items():
        output_path = os.path.abspath(os.path.join(pdfs_dir, f"{pmid}.pdf"))
        if os.path.exists(output_path):
            print(f"[SKIP] PDF already exists for PMID {pmid}")
            results[pmid] = output_path
            continue

        print(f"[INFO] Resolving PDF URL for PMID {pmid} / DOI {doi}")
        pdf_url = get_pdf_url_with_playwright(doi, user_data_dir)
        if not pdf_url:
            print(f"[WARN] Could not resolve PDF URL for PMID {pmid}")
            results[pmid] = None
            continue

        if download_pdf(pmid, pdf_url, pdfs_dir):
            results[pmid] = output_path
        else:
            results[pmid] = None

    return results
