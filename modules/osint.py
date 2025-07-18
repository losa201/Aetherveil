
from aetherveil_sentinel.modules.osint.shodan.shodan_client import ShodanClient
from aetherveil_sentinel.modules.osint.censys.censys_client import CensysClient

class OSINTModule:
    def __init__(self):
        self.shodan_client = ShodanClient()
        self.censys_client = CensysClient()

    def gather_intel(self, target: str):
        """
        Gathers intelligence from all available OSINT sources.

        Args:
            target: The target to gather intelligence on.

        Returns:
            A dictionary of intelligence, with each key representing an OSINT source.
        """
        intel = {}
        intel["shodan"] = self.shodan_client.search(target)
        intel["censys"] = self.censys_client.search(target)
        return intel
