"""
Causal Chain Analyzer - Analyze causal relationships in news events
after this part, we should annotate factor values for each news item and analysis the causal chain based on the data 
"""

import json
import os
import time
from openai import OpenAI


def call_api(prompt: str, model: str = "model_name", max_retries: int = 3) -> str:

    # Configure proxy settings if needed
    # Replace with your actual proxy configuration
    proxy_url = 'your_proxy_url'  # REPLACE WITH YOUR PROXY URL
    proxy_port = 'your_proxy_port'  # REPLACE WITH YOUR PROXY PORT
    
    os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
    os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key="YOUR_OPENAI_API_KEY",  # REPLACE WITH YOUR API KEY
    )
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in public opinion and causal analysis"},
                    {"role": "user", "content": prompt}
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"API call failed: {e}"
            time.sleep(2 ** attempt)
    
    return "All retry attempts failed"


def main():
    prompt = """
    [PROMPT FOR CAUSAL ANALYSIS HERE]
    """
    chains_result = call_api(prompt)
    # Save results to text file
    output_file = "gpt_factors.txt"  # REPLACE WITH YOUR OUTPUT PATH
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chains_result)

        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

