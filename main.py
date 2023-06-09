import openai
import os
from generate_summaries import generate_summaries

if __name__ == "__main__":
    repo_url = "https://github.com/benjamesmurray/repo-crawler"

    # Use os.path.join to create platform-independent paths
    base_path = os.path.expanduser(os.path.join("~", "PycharmProjects", "github-crawler"))
    local_path = os.path.join(base_path, "local_repo")
    output_folder = os.path.join(base_path, "summary_output")

    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"API Key: {api_key}")

    if not api_key:
        raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")
    generate_summaries(repo_url, local_path, api_key, output_folder)
