
from censys.search import CensysHosts
from aetherveil_sentinel.config import get_secret

class CensysClient:
    def __init__(self):
        self.api_id = get_secret("CENSYS_API_ID")
        self.api_secret = get_secret("CENSYS_API_SECRET")
        if not self.api_id or not self.api_secret:
            raise ValueError("Censys API ID or secret not found in configuration.")
        self.api = CensysHosts({"CENSYS_API_ID": self.api_id, "CENSYS_API_SECRET": self.api_secret})

    def search(self, query: str, pages: int = 1):
        """
        Searches Censys for a given query.

        Args:
            query: The search query.
            pages: The number of pages to retrieve.

        Returns:
            A list of search results.
        """
        try:
            results = list(self.api.search(query, pages=pages))
            return results
        except Exception as e:
            print(f"Error: {e}")
            return None
