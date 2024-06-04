import requests

class JinaAiTools:
  def __init__(self):
    self.links_scrapped = []

  def scrape(self, url: str):
    """
    Scrapes content from a specified URL using Jina AI.
    Args:
        url (str): The URL to scrape.
    Returns:
        str: The scraped content in markdown format, or None if scraping fails.
    """
    response = requests.get("https://r.jina.ai/" + url)

    self.links_scrapped.append(url)
    
    return response.text

  def get_links_scrapped(self):
    """
    Returns the list of links that have been scrapped.

    Returns:
        list: A list of links that have been scrapped.
    """
    return self.links_scrapped