# main.py
import os
from generate_summaries import generate_summaries

if __name__ == "__main__":
    repo_url = "https://github.com/department-for-transport-BODS/bods"
    base_path = os.path.expanduser(os.path.join("~", "PycharmProjects", "github-crawler"))
    local_path = os.path.join(base_path, "local_repo")
    output_folder = os.path.join(base_path, "summary_output")

    # Ensure the OPENAI_API_KEY environment variable is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")

    # Configuration variables based on the model's context window
    model_name = "gpt-4-1106-preview"
    model_context_size = 128000  # The total context size for the model
    max_tokens_for_prompt = 1024  # This remains constant as per your requirement
    max_tokens_for_response = 2000  # This remains constant as per your requirement
    max_chunk_size = model_context_size - max_tokens_for_response  # Adjust based on the context size and response token count
    chunk_sample_size = 2500  # Adjust based on your observation

    # Adjusted function call without the api_key argument
    generate_summaries(repo_url, local_path, output_folder, model_name, max_tokens_for_prompt,
                       max_tokens_for_response, max_chunk_size, chunk_sample_size)
