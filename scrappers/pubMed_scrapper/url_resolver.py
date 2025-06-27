import re
from bs4 import BeautifulSoup
from typing import Optional

def find_citedby_section(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    section = soup.find(id="citedby")
    if section:
        return section
    section = soup.find("div", class_="citedby-section")
    if section:
        return section
    citedby_tab = soup.find("a", href="#citedby")
    if citedby_tab:
        target_id = citedby_tab.get('href', '').lstrip('#')
        if target_id:
            section = soup.find(id=target_id)
            if section:
                return section
    citedby_heading = soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4']
                                            and 'Cited by' in tag.get_text())
    if citedby_heading:
        container = citedby_heading.find_parent('section') or citedby_heading.find_parent('div')
        if container:
            return container
        next_sib = citedby_heading.find_next_sibling()
        while next_sib:
            if next_sib.name in ['div', 'section']:
                return next_sib
            next_sib = next_sib.find_next_sibling()
    return None

def is_pubmed_article_url(href: str) -> bool:
    if not href.startswith('/'):
        return False
    pattern = r'^/(?:pubmed/)?\d+(?:[/?]|$)'
    return re.match(pattern, href) is not None

def is_see_all_link(a_tag: BeautifulSoup) -> bool:
    if 'see all' in a_tag.get_text().lower():
        return True
    if 'class' in a_tag.attrs and 'show-all' in a_tag['class']:
        return True
    return False