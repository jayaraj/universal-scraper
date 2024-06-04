from openai import OpenAI
from dotenv import load_dotenv
from agentscraper import AgentScrapper
from tools.firecrawl import FireCrawlTools
from tools.jinaai import JinaAiTools

load_dotenv()
GPT_MODEL = "gpt-4o"

data_points = ["company_name", "company_description", "company_location", 
               "company_founded", "company_founder", "company_investors", 
               "company_revenue", "company_industry", "company_contact", 
               "company_board_members"
               ]
entity_name = "Grafo Technologies"
website = "https://grafotechnologies.com/"

client = OpenAI()
firecrawl = FireCrawlTools(openai=client, model=GPT_MODEL, data_points=data_points)
jinai = JinaAiTools()
agent = AgentScrapper(openai=client, model=GPT_MODEL, fire_crawl_tools=firecrawl, jina_ai_tools=jinai)

# Step 1: Search the website ######################
response = agent.website_search(entity_name, website)

#Step 2: Search the internet ######################
response = agent.internet_search(entity_name)
print(f"internet search: {response}")

print("Data points found after internet search:")
for dp in firecrawl.get_data_points():
  print(f"{dp['name']}: {dp['value']} ({dp['reference']})")

