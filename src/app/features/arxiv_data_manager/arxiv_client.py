# arxiv_client.py
import urllib.request


class ArxivAPIClient:
    """Client for fetching data from the Arxiv API."""

    def __init__(self):
        """Initialize the API client with the base URL."""
        self.base_url = "http://arxiv.org/api/query"

    def fetch_arxiv_data(self, query="all:quantum", max_results=100, total_results_limit=250):
        """Fetch Arxiv data based on a search query."""

        total_results = 0
        start = 0
        all_data = []

        while total_results < total_results_limit:
            sort_by = "relevance"  # or "submitted" or "relevance" or "lastUpdatedDate"
            url = (f'{self.base_url}?search_query={query}&start={start}&max_results={max_results}'
                   f'&sortBy={sort_by}&sortOrder=descending')
            try:
                with urllib.request.urlopen(url) as response:
                    xml_data = response.read().decode('utf-8')
                    all_data.append(xml_data)

                    if "<opensearch:totalResults" in xml_data:  # Find the total number of results
                        total_results = int(
                            xml_data.split("<opensearch:totalResults>")[1].split("</opensearch:totalResults>")[0])
                    else:
                        print(
                            "Error: Unable to find the total number of results in the XML response.")  # Error: Unable to find the total number of results in the XML response.
                        break

                    start += max_results  # Increment the start point for the next request
                    if start >= total_results:  # Stop if all results have been fetched
                        break

            except urllib.error.URLError as e:
                print(f"Network error: {e}")
                break
            except Exception as e:
                print(f"Unknown error: {e}")
                break

        return "".join(all_data)