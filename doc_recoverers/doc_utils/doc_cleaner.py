import re


class DocumentContentCleaner:
    """Clean and normalize document content extracted from PDFs."""

    @staticmethod
    def clean_document(raw_content: str) -> str:
        """Process raw document content into clean text.

        Args:
            raw_content: Text extracted directly from PDF

        Returns:
            Cleaned and normalized text content
        """
        if not raw_content or not raw_content.strip():
            return ""

        cleaned = raw_content
        replacements = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu',
            'π': 'pi', 'σ': 'sigma', 'φ': 'phi', 'ω': 'omega',
            '±': '+-', '≈': '~', '×': 'x', '÷': '/', '°': 'deg',
            '−': '-', '–': '-', '—': '-', '‐': '-', '‑': '-', '⁄': '/'
        }

        for symbol, replacement in replacements.items():
            cleaned = cleaned.replace(symbol, replacement)

        cleaning_patterns = [
            (r'[^\x00-\x7F]+', ' '),  # Non-ASCII characters
            (r'\[[\w\s,.;:\-–+&]+\]', ''),  # Reference citations
            (r'http[s]?://\S+', ''),  # URLs
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ''),  # Emails
            (r'\b(Fig|Figure|Fig\.|Table|Tab|Tbl|Tbl\.)\s*[\dA-Za-z]+\b', ''),  # Figure/table labels
            (r'\([\d\.]+\)', ''),  # Numbered references
            (r'(?<!\w)[^\w\s.,()\[\]%\-+/:;=<>](?!\w)', ''),  # Isolated symbols
        ]

        for pattern, replacement in cleaning_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)

        return re.sub(r'\s+', ' ', cleaned).strip()