import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import tiktoken
from tools.firecrawl import FireCrawlTools
from tools.jinaai import JinaAiTools

class AgentScrapper:
  def __init__(self, openai, model, fire_crawl_tools: FireCrawlTools, jina_ai_tools: JinaAiTools):
    self.openai = openai
    self.model = model
    self.fire_crawl_tools = fire_crawl_tools
    self.jina_ai_tools = jina_ai_tools

  @staticmethod
  def scrape_tools():
    """
    Returns the tool definitions for the  chat request.
    """
    return [
      {
        "type": "function",
        "function": {
          "name": "update_data",
          "description": "Save data points found for later retrieval",
          "parameters": {
            "type": "object",
            "properties": {
              "data_to_update": {
                "type": "array",
                "description": "The data points to update",
                "items": {
                  "type": "object",
                  "description": "The data point to update, should follow specific JSON format: {'name': 'xxx', 'value': 'yyy', 'reference': 'url'}",
                  "properties": {
                    "name": {
                      "type": "string",
                      "description": "The name of the data point",
                    },
                    "value": {
                      "type": "string",
                      "description": "The value of the data point",
                    },
                    "reference": {
                      "type": "string",
                      "description": "The reference URL of the data point",
                    },
                  },
                  "required": ["name", "value", "reference"],
                },
              },
            },
            "required": ["data_to_update"],
          },
        },
      },
      {
        "type": "function",
        "function": {
          "name": "scrape",
          "description": "Scrape a URL for information",
          "parameters": {
            "type": "object",
            "properties": {
              "url": {
                  "type": "string",
                  "description": "The URL of the website to scrape",
              },
            },
            "required": ["url"],
          },
        },
      },
    ]
  
  @staticmethod
  def search_tools():
    """
    Returns the tool definitions for the  chat request.
    """
    return [
      {
        "type": "function",
        "function": {
          "name": "update_data",
          "description": "Save data points found for later retrieval",
          "parameters": {
            "type": "object",
            "properties": {
              "data_to_update": {
                "type": "array",
                "description": "The data points to update",
                "items": {
                  "type": "object",
                  "description": "The data point to update, should follow specific JSON format: {'name': 'xxx', 'value': 'yyy', 'reference': 'url'}",
                  "properties": {
                    "name": {
                      "type": "string",
                      "description": "The name of the data point",
                    },
                    "value": {
                      "type": "string",
                      "description": "The value of the data point",
                    },
                    "reference": {
                      "type": "string",
                      "description": "The reference URL of the data point",
                    },
                  },
                  "required": ["name", "value", "reference"],
                },
              },
            },
            "required": ["data_to_update"],
          },
        },
      },
      {
        "type": "function",
        "function": {
          "name": "search",
          "description": "Search the internet for information and related URLs",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "The query to search, should be a semantic search query as we are using AI to search",
              },
              "entity_name": {
                "type": "string",
                "description": "The name of the entity that we are researching about",
              },
            },
            "required": ["query", "entity_name"],
          },
        },
      },
    ]

  def get_tools_list(self):
    """
    Returns a dictionary of tool names and their corresponding functions.

    Returns:
        dict: A dictionary of tool names and their corresponding functions.
    """
    return {
        "scrape": self.jina_ai_tools.scrape,
        "search": self.fire_crawl_tools.search,
        "update_data": self.fire_crawl_tools.update_data,
    }
  
  @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
  def chat_request (self, messages, tool_definitions, tool_choice):
    """
    Make a chat request to the OpenAI API, with retries on failure.

    Args:
        messages (list): A list of messages for the chat request.
        tool_definitions (list): The tool definitions for the chat request, if any (default: None).
        tool_choice (str): The tool choice for the chat request, if any.

    Returns:
        response: The response from the OpenAI API, or None if the request fails.
    """
    try:
      response = self.openai.chat.completions.create(
        model=self.model,
        messages=messages, 
        tools=tool_definitions,
        tool_choice=tool_choice,
      )
      return response
    except Exception as e:
      print(colored(f"unable to generate chat response: {response}, error {e}", "red"))
      return None
  
  def print(self, message):
    """
    Prints a conversation message with color coding based on the role.

    Args:
        message (dict): The message to print, containing at least a 'role' and 'content'.
    """
    role_to_color = {
      "system": "red",
      "user": "green",
      "assistant": "blue",
      "tool": "magenta",
    }

    role = message.get("role", "unknown")
    color = role_to_color.get(role, "white")

    if role == "assistant" and message.get("tool_calls"):
      content = message.get("tool_calls", "")
    else:
      content = message.get("content", "")

    print(colored(f"{role}: {content}", color))

  def optimise_messages(self, messages: list):
    """
    Optimizes the list of messages to keep within token limits by summarizing past messages.

    Args:
        messages (list): A list of messages for the chat completion.

    Returns:
        list: The optimized list of messages.
    """
    system_prompt = messages[0]["content"]
    model = self.model
    if "gpt" not in self.model:
      model = "gpt-3.5-turbo"
    encoding = tiktoken.encoding_for_model(model)

    if len(messages) > 24 or len(encoding.encode(str(messages))) > 10000:
      latest_messages = messages [-12:]
      token_count_latest_messages = len(encoding.encode(str(latest_messages) ))
      print(f"Token count of latest messages: {token_count_latest_messages}")
      
      index = messages.index(latest_messages [0])
      early_messages = messages[:index]

      prompt = f"""
      Conversation History:
      {early_messages}
      
      -----
      
      Above is the conversation history between the user and the AI, including actions the AI has already taken.
      Please summarize the past actions taken so far, highlight any key information learned, and mention tasks that have been completed.
      Remove any redundant information and keep the summary concise. remove scrapped content from the summary.

      SUMMARY:
      """

      try:
        response = self.openai.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[{"role": "user", "content": prompt} ],
          temperature=0
        )
        summary = response.choices[0].message.content
        system_prompt = f"{system_prompt}; Here is a summary of past actions taken so far: {summary}"
        messages = [{"role": "system", "content": system_prompt}] + latest_messages

        print(colored(messages, "green"))

        return messages
      except Exception as e:
        print(colored(f"error while summarizing the past messages: {e}", "red"))
        return messages
    
    return messages

  def run(self, prompt, system_prompt, tool_definitions, plan):
    """
    Executes the main loop for interacting with the AI, making tool calls as necessary.

    Args:
        prompt (str): The user's input prompt.
        system_prompt (str): The system's initial prompt.
        tool_definitions (list): The tool definitions for the chat request.
        plan (bool): Whether to start by creating a plan.

    Returns:
        str: The final response from the AI after completing the interaction loop.
    """
    messages =[]

    if plan:
      messages.append(
        {
          "role": "user",
          "content": ( system_prompt + " " + prompt + " Let's think step by step, make a plan first")
        }
      )
      chat_response = self.chat_request(messages, tool_definitions, tool_choice=None)
      if chat_response == None:
        print(colored(f"failed to get plan response", "red"))
        return None
      messages = [
        {"role": "user", "content": (system_prompt + " " + prompt)},
        {"role": "assistant", "content": chat_response.choices[0].message.content},
      ]
    else:
      messages.append({"role": "user", "content": (system_prompt + " " + prompt)})
    
    state = "running"
    for message in messages:
      self.print(message)

    while state == "running":
      chat_response = self.chat_request(messages, tool_definitions, tool_choice=None)
      if chat_response == None:
        print(colored(f"failed to get chat response", "red"))
        state = "finished"
      else:
        current_choice = chat_response.choices[0]
        messages.append(
        {
          "role": "assistant", 
          "content": current_choice.message.content,
          "tool_calls": current_choice.message.tool_calls,
        })
        self.print(messages[-1])

        if current_choice.finish_reason == "tool_calls":
          tool_calls = current_choice.message.tool_calls
          for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_list = self.get_tools_list()
            tool_to_call = tool_list[tool_name]
            if tool_to_call:
              result = tool_to_call(**tool_args)
              messages.append({
                  "role": "tool",
                  "tool_call_id": tool_call.id,
                  "name": tool_name,
                  "content": result,
              })
              self.print(messages[-1])
            else:
              print(colored(f"tool not found", "red"))
              return None

        if current_choice.finish_reason == "stop":
          state = "finished"
        
        messages = self.optimise_messages(messages)

    return messages[-1]["content"]

  def website_search (self, entity_name: str, website: str):
    """
    Searches a given website for specific data points related to an entity.

    Args:
        entity_name (str): The name of the entity to search for.
        website (str): The URL of the website to scrape for information.

    Returns:
        str: The final response from the AI after completing the interaction loop.
    """
    data_points_to_search = self.fire_crawl_tools.get_data_points_to_search()
    
    system_prompt = """
      you are a world class web scraper, you are great at finding information on urls;

      You will keep scraping url based on information you received until information is found;

      If you can't find relevant information from the company's domain related urls, 
      Whenever you found certain data point, use "update_data" function to save the data point;
      
      You only answer questions based on results from scraper, do not make things up;
      
      You NEVER ask user for inputs or permissions, just go ahead do the best thing possible without asking for permission or guidance from user;
    """

    prompt = f"""
      Entity to search: {entity_name}

      Website to search: {website}

      Data points to search: {data_points_to_search}

      Search only the data points that are not found yet and are in the data points to search list.
      Stop if all data points are found.
      Stop if no data points are found in search list.
      Need not search for data points that are already found.
    """

    response = self.run(prompt, system_prompt, self.scrape_tools(), plan=True)

    return response

  def internet_search(self, entity_name: str):
    """
    Searches the internet for specific data points related to an entity.

    Args:
        entity_name (str): The name of the entity to search for.

    Returns:
        str: The final response from the AI after completing the interaction loop.
    """
    data_points_to_search = self.fire_crawl_tools.get_data_points_to_search()

    system_prompt = """
      you are a world class web researcher, you are great at finding information on the internet;
      You will keep scraping url based on information you received until information is found;

      You will try as hard as possible to search for all sorts of different query & source to find information; 
      if one search query didn't return any result, try another one; 
      You do not stop until all information are found, it is very important we find all information, I will give you $100, 000 tip if you find all information;
      Whenever you found certain data point, use "update_data" function to save the data point;
      
      You only answer questions based on results from scraper, do not make things up;
      You never ask user for inputs or permissions, you just do your job and provide the results;
      You ONLY run 1 function at a time, do NEVER run multiple functions at the same time
    """

    prompt = f"""
      Entity to search: {entity_name}

      Links we already scraped: {self.fire_crawl_tools.get_links_scrapped()}

      Data points to search: {data_points_to_search}
      
      Search only the data points that are not found yet and are in the data points to search list.
      Stop if all data points are found.
      Stop if no data points are found in search list.
      Need not search for data points that are already found.
    """

    response = self.run(prompt, system_prompt, self.search_tools(), plan=False)

    return response