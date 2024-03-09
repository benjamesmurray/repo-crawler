import os
import logging
from openai import OpenAI
import re
import time
import random
import traceback
from git import Repo
from transformers import GPT2Tokenizer

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Then, initialize the OpenAI client with the API key
client = OpenAI(api_key=api_key)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def calculate_avg_chars_per_token(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return 2.5  # Default estimated ratio if no tokens found
    avg_chars_per_token = len(text) / len(tokens)
    return avg_chars_per_token * 0.75  # Apply a reduction factor for safety margin


def contains_test_keyword(name):
    test_keywords = {"test", "tests"}
    return any(keyword.lower() in name.lower() for keyword in test_keywords)


def get_python_files(repo_path):
    python_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py") and not contains_test_keyword(file):
                file_path = os.path.join(root, file)
                print(f"Found Python file: {file_path}")
                python_files.append(file_path)
                print(f"Appending file {file_path} to the list")
    return python_files


def clone_repo(repo_url, local_path):
    try:
        Repo.clone_from(repo_url, local_path)
        logging.info(f"Cloned repository: {repo_url} to {local_path}")
        print(f"Repo cloned successfully: {repo_url} to {local_path}")
        print(f"Cloned repository: {repo_url} to {local_path}")
    except Exception as e:
        logging.error(f"Error cloning repository: {e}")
        print(
            f"Error cloning repository: {repo_url} to {local_path}. Error: {e}"
        )


def process_files(repo_path, output_folder, tokenizer):
    python_files = get_python_files(repo_path)
    for file_path in python_files:
        try:
            content = read_file_content(file_path)
            chunks = chunk_content(content, tokenizer)
            comments = []  # Initialize an empty list to collect comments
            for chunk in chunks:
                comment = generate_comments(chunk)  # Assuming this function returns the generated comment for a chunk
                comments.append(comment)
            combined_comments = "\n".join(comments)  # Combine all comments into a single string
            save_commented_file(output_folder, repo_path, file_path, combined_comments)  # Pass combined comments here
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")



def generate_commented_code(repo_url, local_path, api_key, output_folder):
    print("Entering generate_summaries function")

    # Set up your OpenAI API key

    # Set up logging
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Clone the GitHub repo to a local directory
    clone_repo(repo_url, local_path)

    # Generate summaries for the files
    print("Starting to process files...")
    process_files(local_path, output_folder, gpt2_tokenizer)
    print("Finished processing files")


def chunk_content(file_content, tokenizer):
    avg_chars_per_token = calculate_avg_chars_per_token(file_content[:5000], tokenizer)
    # Consider both the chunk of code and potential comments within 2500 tokens
    estimated_max_chunk_size_in_tokens = 2500  # Reflects max_tokens for API calls
    max_chars_per_chunk = (estimated_max_chunk_size_in_tokens) * avg_chars_per_token
    content_chunks = []
    start_idx = 0
    while start_idx < len(file_content):
        end_idx = min(start_idx + max_chars_per_chunk, len(file_content))
        if end_idx < len(file_content):
            end_idx = file_content.rfind('\n', start_idx, end_idx) + 1 or end_idx
        content_chunks.append(file_content[start_idx:end_idx])
        start_idx = end_idx
    return content_chunks


def get_sample(content, sample_size):
    content_length = len(content)
    if content_length <= sample_size:
        return content

    start_index = (content_length // 2) - (sample_size // 2)
    end_index = start_index + sample_size
    return content[start_index:end_index]


def is_response_incomplete(response: str) -> bool:
    response = response.strip()
    if not response:
        return False
    if re.match(r'\w', response[-1]):
        return True
    return not re.match(r'[.!?]', response[-1])


def generate_comments(chunk_text):
    if len(chunk_text.strip()) == 0 or len(chunk_text) < 10:
        print("Chunk content is empty or too small to generate a meaningful summary.")
        return chunk_text

    prompt = (
        "Your task is to generate a documented version of the given code snippet using docstrings. The docstring should be interspersed with the original code for real-time explanations. "
        "For each function or method or class, add a succinct docstring above it explaining its purpose, input parameters, and expected output.\n\n"
        f"{chunk_text}\n"
        "Note: The response must be the original code plus your appropriately placed docstrings. No explanatory text."
    )

    retries = 3
    max_tokens = 2500
    retry_wait = 1  # Initial waiting time in seconds
    retry_factor = 2  # Exponential backoff factor

    while retries > 0:
        print(f"Retries left: {retries}")

        try:
            response = client.chat.completions.create(model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            n=1,
            temperature=0.5)
            print("[2] API call successful.")
        except Exception as e:
            print("[2] Error during API call:", e)
            if retries > 1:  # No need to wait if there are no retries left
                wait_time = retry_wait * (retry_factor ** (3 - retries))
                jitter = random.uniform(0.5, 1.5)  # Add jitter to the waiting time
                time.sleep(wait_time * jitter)
                retries -= 1
                continue
            else:
                raise Exception(f"API call error: {e}")

        print(f"Response: {response}")  # Add this line to print the response
        summary = response.choices[0].message.content.strip()  # Update this line
        print(f"Attempt {4 - retries} summary:\n{summary}")

        return summary


def reconstruct_comments(comment_list):
    reconstructed_comments = []
    for chunk_comments in comment_list:
        lines = chunk_comments.splitlines()
        for line in lines:
            reconstructed_comments.append(line)
    return "\n".join(reconstructed_comments)


def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def process_chunks(content, tokenizer, generate_comments):
    print("Entered process_chunks function")

    # Extract a sample from the content
    sample_size = 2500  # You can adjust this value
    content_sample = content[:sample_size]
    # print("Extracted content sample:", content_sample)

    # Calculate the average characters per token for the sample
    print("Calling calculate_avg_chars_per_token function")
    avg_chars_per_token = calculate_avg_chars_per_token(content_sample, tokenizer)
    print(f"Average characters per token: {avg_chars_per_token}")

    max_chunk_size = 128000 - 4000
    print(f"Max chunk size: {max_chunk_size}")

    # Calculate the maximum number of characters per chunk
    max_chars_per_chunk = int(max_chunk_size * avg_chars_per_token)

    # Split the content into chunks based on the maximum number of characters per chunk
    content_chunks = []
    start_idx = 0

    while start_idx < len(content):
        end_idx = start_idx + max_chars_per_chunk

        # Find the closest word or paragraph boundary (whitespace or newline character)
        while end_idx < len(content) and content[end_idx] not in {'\n', ' '}:
            end_idx -= 1

        chunk = content[start_idx:end_idx]
        content_chunks.append(chunk)
        start_idx = end_idx

    print(f"Number of chunks: {len(content_chunks)}")

    summaries = []
    for chunk in content_chunks:
        print("Processing chunk for summary generation...")
        summary = generate_comments(chunk)
        print("Summary generation complete for this chunk.")
        summaries.append(summary)

    return summaries


def save_commented_file(output_folder, repo_path, file_path, comments):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a new file in the output folder with the same relative path as the original file
    relative_path = os.path.relpath(file_path, repo_path)
    output_file_path = os.path.join(output_folder, relative_path)

    # Create the parent directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Read the original file content
    with open(file_path, 'r', encoding='utf-8') as file:
        original_content = file.read()

    # Combine the original content with the generated comments
    commented_content = f"{comments}\n\n{original_content}"

    # Write the commented content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(commented_content)

    return output_file_path
