
import shodan
from aetherveil_sentinel.config import get_secret

class ShodanClient:
    def __init__(self):
        self.api_key = get_secret("SHODAN_API_KEY")
        if not self.api_key:
            raise ValueError("Shodan API key not found in configuration.")
        self.api = shodan.Shodan(self.api_key)

    def search(self, query: str):
        """
        Searches Shodan for a given query.

        Args:
            query: The search query.

        Returns:
            A dictionary of search results.
        """
        try:
            results = self.api.search(query)
            return results
        except shodan.APIError as e:
            print(f"Error: {e}")
            return None
