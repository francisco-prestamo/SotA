from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import requests
import json
from typing import List, Dict, Any, Mapping, Tuple
import time

from .models.metadata_result import MetadataResult

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OUTPUT_FILE = "pmid_articleids_map.json"
BATCH_SIZE = 5


def fetch_article_metadata(
    pmids: List[str], cache_root: Path
) -> dict[str, MetadataResult | None]:
    """
    Given a list of PMIDs, fetch the articleids from PubMed ESummary API.

    Returns a dictionary mapping PMID -> articleids list (or None if failed).
    """

    print(f"[INFO] Fetching metadata for {pmids}")

    cache_location = Path(os.path.join(cache_root, OUTPUT_FILE))

    results = load_cache(cache_location)
    results = filter_pmids(pmids, results)

    # Filter out already-processed PMIDs
    pending_pmids = list(set(pmids) - set(results.keys()))
    print(f"[INFO] {len(pending_pmids)} PMIDs left to process")

    if not pending_pmids:
        print("[INFO] All PMIDs are already processed.")
        return results

    results = to_dicts(results)

    i = 0
    while i < len(pmids):
        batch = pmids[i : i + BATCH_SIZE]
        print(f"[INFO] Getting metadata for {batch}...")

        batch_results = fetch_aticle_metadata_batched(batch)
        for k, v in batch_results.items():
            if v:
                results[k] = v.model_dump()
            else:
                results[k] = v

        dump_cache(cache_location, results)

        i += BATCH_SIZE

    return to_pydantic_models(results)


def filter_pmids(
    pmids: List[str], data: dict[str, MetadataResult | None]
) -> dict[str, MetadataResult | None]:
    result = {}
    pmid_set = set(pmids)
    for k, v in data.items():
        if k not in pmid_set:
            continue
        if not v:
            continue
        result[k] = v

    return result


def load_cache(cache_location: Path) -> dict[str, MetadataResult | None]:
    if os.path.isfile(cache_location):
        with open(cache_location, "r") as f:
            results = json.load(f)
            for k, v in results.items():
                results[k] = MetadataResult.model_validate(v) if v != None else None
        print(f"[INFO] Loaded {len(results)} existing entries from {OUTPUT_FILE}")
    else:
        results = {}
        print("[INFO] No existing output file found. Starting fresh.")

    return results


def dump_cache(cache_location: Path, data: dict[str, Any]):
    d = load_cache(cache_location)
    d = to_dicts(d)

    for k, v in data.items():
        d[k] = v

    with open(cache_location, "w") as f:
        f.write(json.dumps(d, indent=2))


def to_dicts(data: Dict[str, MetadataResult | None]) -> dict[str, Any]:
    return {k: v.model_dump() if v else None for k, v in data.items()}


def to_pydantic_models(data: dict[str, Any]) -> Dict[str, MetadataResult | None]:
    return {k: MetadataResult.model_validate(v) if v else None for k, v in data.items()}


def fetch_aticle_metadata_batched(pmids: List[str]) -> Dict[str, MetadataResult | None]:
    n = len(pmids)
    answ = {}
    with ThreadPoolExecutor(max_workers=n) as executor:
        future_to_input = {
            executor.submit(fetch_metadata_for_pmid, pmid): pmid for pmid in pmids
        }

        for future in as_completed(future_to_input):
            pmid, result = future.result()
            answ[pmid] = result

    return answ


def fetch_metadata_for_pmid(pmid: str) -> Tuple[str, MetadataResult | None]:
    params = {"db": "pubmed", "id": pmid, "retmode": "json"}

    max_retries = 5
    delay = 0.5  # initial delay in seconds
    max_delay = 10  # maximum delay cap in seconds

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            if resp.status_code == 429:
                wait_time = min(delay * (2 ** (attempt - 1)), max_delay)
                print(
                    f"PMID {pmid}: Received 429 Too Many Requests. Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                continue  # retry the request
            resp.raise_for_status()

            return pmid, MetadataResult.parse_from_response(pmid, resp)

        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"Failed to fetch data for PMID {pmid}: {e}")
            return pmid, None

    print(f"PMID {pmid}: Failed after {max_retries} retries")
    return pmid, None
