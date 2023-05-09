import openai
import os
from generate_commented_code import generate_commented_code

if __name__ == "__main__":
    repo_url = "https://github.com/department-for-transport-BODS/bods"

    # Use os.path.join to create platform-independent paths
    base_path = os.path.expanduser(os.path.join("~", "PycharmProjects", "github-crawler"))
    local_path = os.path.join(base_path, "local_repo")
    output_folder = os.path.join(base_path, "commented_code_output")

    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"API Key: {api_key}")

    if not api_key:
        raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
    generate_commented_code(repo_url, local_path, api_key, output_folder)  # Call generate_commented_code instead of generate_summaries
