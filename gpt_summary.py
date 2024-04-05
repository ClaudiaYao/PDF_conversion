import os
import json
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI()

def generate_prompt(section_name, section_text):
    prompt = f"""
Summarise the text with sufficient information for a technical presentation. Go straight into the summary, do not quote back the section name or start with "The section discusses..." or similar wording.
Section name:
{section_name}
Text:
{section_text}
    """
    return prompt

# Count num tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_truncated_prompt(prompt):
    max_tokens = 16385
    num_tokens = num_tokens_from_string(prompt, "cl100k_base")

    while num_tokens >= max_tokens - 1000:
        prompt = prompt[:-100]
        num_tokens = num_tokens_from_string(prompt, "cl100k_base")
    
    return prompt

def get_gpt_response(prompt):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
      ]
    )

    return response.choices[0].message.content

def get_summaries(json):
    for section in json:
        name = section['Section']
        text = section['Text']
        subsections = section['Subsections']

        if text:
            prompt = generate_prompt(name, text)
            prompt = get_truncated_prompt(prompt)
            section['Groundtruth'] = get_gpt_response(prompt)
        
        if subsections:
            get_summaries(subsections)