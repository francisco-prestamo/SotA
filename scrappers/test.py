from scrappers.pubMed_scrapper.pubMed_scrapper import PubMedScraper

example_url = "https://pubmed.ncbi.nlm.nih.gov/33316383/"

def main():
    scraper = PubMedScraper()

    print("\n--- Testing get_metadata ---")
    metadata = scraper.get_metadata(example_url)
    print(f"Title: {metadata['title']}")
    print(f"Abstract: {metadata['abstract'][:300]}...")
    print(f"Authors: {', '.join(metadata['authors'])}\n")

    print("\n--- Testing get_full_text ---")
    fulltext = scraper.get_full_text(example_url)
    print(f"Text: \b {fulltext}\n")

    print("\n--- Testing get_similar_urls ---")
    similar_urls = scraper.get_similar_urls(example_url)
    print(f"Found {len(similar_urls)} reference URLs:")
    for url in similar_urls[:10]:
        print(f"- {url}")

    print("\n--- Testing get_cited_by_urls ---")
    get_cited_by_urls = scraper.get_cited_by_urls(example_url)
    print(f"Found {len(get_cited_by_urls)} reference URLs:")
    for url in get_cited_by_urls[:10]:
        print(f"- {url}")

    print("\n--- Testing get_reference_urls ---")
    reference_urls = scraper.get_reference_urls(example_url)
    print(f"Found {len(reference_urls)} reference URLs:")
    for url in reference_urls[:10]:
        print(f"- {url}")


if __name__ == "__main__":
    main()
