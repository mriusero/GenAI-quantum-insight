# arxiv_parser.py
import xml.etree.ElementTree as ET


class ArxivXMLParser:
    """Parses XML data from the Arxiv API."""

    def __init__(self):
        """Initialize the parser with the required namespaces."""
        self.namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

    def parse_entries(self, xml_data):
        """Parse XML data and return a list of arXiv entries."""
        try:
            root = ET.fromstring(xml_data)
            entries = []
            for entry in root.findall('atom:entry', self.namespace):
                arxiv_entry = {
                    'id': entry.find('atom:id', self.namespace).text,
                    'title': entry.find('atom:title', self.namespace).text,
                    'summary': entry.find('atom:summary', self.namespace).text,
                    'author': entry.find('atom:author/atom:name', self.namespace).text,
                    'published': entry.find('atom:published', self.namespace).text,
                    'updated': entry.find('atom:updated', self.namespace).text,
                    'pdf_link': entry.find('atom:link[@title="pdf"]', self.namespace).attrib['href']
                }
                entries.append(arxiv_entry)
            return entries
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            return []