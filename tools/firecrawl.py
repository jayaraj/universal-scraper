
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from termcolor import colored

load_dotenv()

class FireCrawlTools:
  def __init__(self, openai, model, data_points):
    self.app = FirecrawlApp()
    self.links_scrapped = []
    self.data_points = [{"name": dp, "value": None, "reference": None} for dp in data_points]
    self.openai = openai
    self.model = model

  def scrape(self, url: str):
    """
    Scrapes content from a specified URL using FirecrawlApp.
    Args:
        url (str): The URL to scrape.
    Returns:
        str: The scraped content in markdown format, or None if scraping fails.
    """
    try:
      scraped_data = self.app.scrape_url(url)
    except Exception as e:
      print(colored(f"unable to scrape the URL error: {e}", "red"))
      return None
    
    self.links_scrapped.append(url)
    return scraped_data.get("markdown", "")
  
  def search(self, query, entity_name: str):
    """
    Searches for information related to a specific entity using FirecrawlApp.

    Args:
        query (str): The search query.
        entity_name (str): The name of the entity to search for.

    Returns:
        dict: A dictionary containing information extracted from search results.
    """
    params = {"pageOptions": {"fetchPageContent": True}}

    search_result = self.app.search(query, params=params)
    search_result_str = str(search_result)
    data_keys_to_search = [obj["name"] for obj in self.data_points if obj ["value"] is None]

    prompt = f""" 
    Below are some search results from the internet about {query}:
    {search_result_str}
    -----

    Your goal is to find specific information about an entity called {entity_name} regarding {data_keys_to_search}.

    Please extract information from the search results above in the following JSON format:
    {{
        "related_urls_to_scrape_further": ["url1", "url2", "url3"],
        "info_found": [
            {{
                "research_item": "xxxx",
                "reference": "url"
            }},
            {{
                "research_item": "yyyy",
                "reference": "url"
            }}
            ...
        ]
    }}

    Where "research_item" is the actual research item name you are looking for.

    Only return research items that you actually found.
    If no research item information is found from the content provided, just don't return any research item.

    Extracted JSON:
    {{
        "related_urls_to_scrape_further": [],
        "info_found": []
    }}
    """

    response = self.openai.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    result = response.choices[0].message.content
    return result
  
  def update_data (self, data_to_update):
    """
    Update the state with new data points found.

    Args:
        data_to_update (List[dict]): The new data points found, which should follow the format 
        [{"name": "xxx", "value": "yyy", "reference": "url"}]
    """
    for data in data_to_update:
      for obj in self.data_points:
        if obj ["name"] == data ["name"]:
          obj ["value"] = data ["value"]
          obj ["reference"] = data ["reference"]

    return f"updated data: {data_to_update}"
  
  def get_data_points_to_search(self):
    """
    Returns the list of data points that need to be searched for.

    Returns:
        list: A list of data points that need to be searched for.
    """
    return [obj["name"] for obj in self.data_points if obj["value"] is None]
  
  def get_links_scrapped(self):
    """
    Returns the list of links that have been scrapped.

    Returns:
        list: A list of links that have been scrapped.
    """
    return self.links_scrapped
  
  def get_data_points(self):
    """
    Returns the list of data points found.

    Returns:
        list: A list of data points found.
    """
    return self.data_points