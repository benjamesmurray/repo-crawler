import json
import re
import traceback
import os
import logging
import openai
from git import Repo
from transformers import GPT2Tokenizer

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def generate_summaries(repo_url, local_path, output_folder, model_name, max_tokens_for_prompt, max_tokens_for_response,
                       max_chunk_size, chunk_sample_size):
    print("Entering generate_summaries function")

    # Fetch the API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")

    # Use the API key to configure OpenAI
    openai.api_key = api_key

    # Set up logging
    logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Clone the GitHub repo to a local directory
    clone_repo(repo_url, local_path)

    # Generate summaries for the files
    print("Starting to process files...")
    process_files(local_path, output_folder, gpt2_tokenizer, chunk_sample_size, max_chunk_size, model_name,
                  max_tokens_for_response)
    print("Finished processing files")


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


def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def save_summary(output_folder, repo_path, file_path, summary):
    # Get the relative path of the file with respect to the local repository path
    relative_path = os.path.relpath(file_path, repo_path)

    # Replace the .py extension with .txt for the summary file
    summary_filename = f"{os.path.splitext(relative_path)[0]}.txt"

    # Create the corresponding folder structure inside the output folder
    output_file_path = os.path.join(output_folder, summary_filename)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary)

    print(f"Summary for {file_path} saved at {output_file_path}")
    return output_file_path


def process_files(repo_path, output_folder, tokenizer, chunk_sample_size, max_chunk_size, model_name, max_tokens_for_response):
    print("Entering process_files function")
    python_files = get_python_files(repo_path)

    logging.info(f"Found {len(python_files)} Python files to process.")
    print(f"Found {len(python_files)} Python files to process.")

    for index, file_path in enumerate(python_files, 1):
        logging.info(f"Processing file {index}/{len(python_files)}: {file_path}")
        print(f"Processing file {index}/{len(python_files)}: {file_path}")

        try:
            content = read_file_content(file_path)

            print("Processing file content...")
            summaries = process_chunks(content, tokenizer, chunk_sample_size, max_chunk_size, model_name,
                                       max_tokens_for_response)
            print("Processing file content complete.")

            if len(summaries) == 1:
                print("Only one chunk summary generated. Using it as the output.")
                reconstructed_summary = summaries[0]
            else:
                print("Reconstructing summaries...")
                try:
                    reconstructed_summary = reconstruct_summaries(summaries)
                except Exception as e:
                    print("Error occurred in reconstruct_summaries function:")
                    print(traceback.format_exc())
                    raise e
                print("Reconstruction complete.")

            print("Reconstructed summary:\n", json.dumps(reconstructed_summary, indent=2))

            output_file_path = save_summary(output_folder, repo_path, file_path, json.dumps(reconstructed_summary, indent=2))

            logging.info(f"Generated summary for {file_path} and saved to {output_file_path}")
            print(f"Generated summary for {file_path} and saved to {output_file_path}")
            print(f"Value written to the file: {json.dumps(reconstructed_summary, indent=2)}")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")


def process_chunks(content, tokenizer, chunk_sample_size, max_chunk_size, model_name, max_tokens_for_response):
    print("Entered process_chunks function")

    # Extract a sample from the content
    sample_size=chunk_sample_size  # You can adjust this value in main.py
    content_sample = content[:sample_size]
    # print("Extracted content sample:", content_sample)

    # Calculate the average characters per token for the sample
    print("Calling calculate_avg_chars_per_token function")
    avg_chars_per_token = calculate_avg_chars_per_token(content_sample, tokenizer)
    print(f"Average characters per token: {avg_chars_per_token}")

    max_chunk_size=max_chunk_size
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
        summary = generate_summary(chunk, model_name, max_tokens_for_response)
        print("Summary generation complete for this chunk.")
        summaries.append(summary)

    return summaries


def generate_summary(chunk_text, model_name, max_tokens):
    required_keys = [
        "Overall Summary",
        "Module/Library Name",
        "ETL Processes",
        "Classes",
        "Functions/Methods",
        "Variables",
        "Data Structures",
        "Interfaces/APIs",
        "Configuration/Environment Variables",
        "Error Handling/Logging",
        "Data Inputs",
        "Data Outputs"
    ]

    # Check if the chunk is empty or too small to generate a summary
    if len(chunk_text.strip()) == 0 or len(chunk_text) < 10:  # Adjust the threshold as needed
        print("Chunk content is empty or too small to generate a meaningful summary.")
        empty_summary = {key: {} if key in {"Classes", "Functions/Methods", "Configuration/Environment Variables"} else "" for key in required_keys}
        return empty_summary

    prompt = (
        "Generate a JSON structure summary for the content below. "
        "Follow the structure outlined here:\n\n"
        'Example JSON Structure:\n'
        '{\n'
        '  "Overall Summary": "A brief summary of the code snippet...",\n'
        '  "Module/Library Name": "ExampleLib",\n'
        '  "ETL Processes": {\n'
        '    "Extract": {\n'
        '      "Description": "Details of the data extraction process...",\n'
        '      "Data Sources": ["Source1", "Source2"],\n'
        '      "Methods": {"extract_method": "Description of extraction method..."}\n'
        '    },\n'
        '    "Transform": {\n'
        '      "Description": "Details of data transformation process...",\n'
        '      "Transformations": {"transformation1": "Description of transformation..."},\n'
        '      "Dependencies": ["DependentClass/Function"],\n'
        '      "Used By": ["OtherTransformations/Processes"]\n'
        '    },\n'
        '    "Load": {\n'
        '      "Description": "Details of the data loading process...",\n'
        '      "Data Destinations": ["Destination1", "Destination2"],\n'
        '      "Methods": {"load_method": "Description of loading method..."}\n'
        '    }\n'
        '  },\n'
        '  "Classes": {\n'
        '    "ExampleClass": {\n'
        '      "Description": "A class that...",\n'
        '      "Properties": {"prop1": "Description of prop1..."},\n'
        '      "Methods": {"method1": "Description of method1..."},\n'
        '      "Dependencies": ["OtherClass"],\n'
        '      "Used By": ["YetAnotherClass"]\n'
        '    }\n'
        '  },\n'
        '  "Functions/Methods": {\n'
        '    "example_function": {\n'
        '      "Description": "A function that...",\n'
        '      "Parameters": ["param1", "param2"],\n'
        '      "Returns": "Return type and/or description...",\n'
        '      "Calls": ["another_function"],\n'
        '      "Called By": ["some_function"]\n'
        '    }\n'
        '  },\n'
        '  "Variables": {"variable1": "Description and type of variable1..."},\n'
        '  "Data Structures": "Lists, dictionaries...",\n'
        '  "Interfaces/APIs": "REST API endpoints...",\n'
        '  "Configuration/Environment Variables": {"EXAMPLE_VAR": "A variable that..."},\n'
        '  "Error Handling/Logging": "Error handling details...",\n'
        '  "Data Inputs": "Description of the data inputs...",\n'
        '  "Data Outputs": "Description of the data outputs..."\n'
        '}\n\n'
        "1. Overall Summary: Provide a brief summary of the code snippet.\n"
        "2. Module/Library Name: Provide the name of the module or library used in the file.\n"
        "3. ETL Processes: \n"
        "   a. Extract: Describe the data extraction methods, their sources, and any relevant details.\n"
        "   b. Transform: Describe the transformation logic, any dependencies, and entities that use these transformations.\n"
        "   c. Load: Describe how and where the data is loaded, including methods and destinations.\n"
        "4. Classes: Briefly describe the purpose, properties, and methods for each class in the file, "
        "including inheritance and composition relationships, if applicable.\n"
        "5. Functions/Methods: Briefly describe each function or method in the file, "
        "including input parameters, return values, and any side effects. "
        "Document any significant algorithms or logic used within the functions.\n"
        "6. Variables: Describe each variable in the file, including its type, purpose, "
        "and any functions or methods that access or modify it.\n"
        "7. Data Structures: Describe the key data structures used in the file, "
        "such as lists, dictionaries, or custom data structures, and their purposes. "
        "Include any significant relationships or interactions between these structures.\n"
        "8. Interfaces/APIs: If the file exposes any APIs or interfaces for integration with other systems, "
        "document their endpoints, input and output formats, and any authentication or authorization requirements.\n"
        "9. Configuration/Environment Variables: List any configuration files or environment variables "
        "required for the file, and describe their purposes and possible values.\n"
        "10. Error Handling/Logging: Explain the error handling and logging mechanisms used in the file, "
        "including any specific error codes or messages that developers should be aware of.\n"
        "11. Data Inputs: Describe the data inputs for the file, including any file formats, "
        "data sources, or user input requirements.\n"
        "12. Data Outputs: Describe the data outputs for the file, including any file formats, "
        "data destinations, or user output requirements.\n\n"
        f"{chunk_text}\n"
        "Important: Summarize the content without using any part of the original code. "
        "Ensure your response is a useful summary for a developer to read, "
        "and always provide a JSON structure with all keys present in the example."
    )

    retries = 5
    max_tokens=max_tokens
    while retries > 0:
        print(f"Retries left: {retries}")

        try:
            response = openai.chat.completions.create(model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that only supplies responses in JSON structures."
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
            raise Exception(f"API call error: {e}")

        summary = response.choices[0].message.content.strip()
        print(f"Attempt {6 - retries} summary:\n{summary}")

        json_pattern = re.compile(r'(?s)\{.*}')
        json_match = json_pattern.search(summary)
        if json_match:
            json_text = json_match.group()
            print("Matched JSON text:")
            print(json_text)
        else:
            print("No JSON text found in the summary.")
            json_text = ""

        try:
            summary_dict = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            summary_dict = {}

        print("JSON dictionary after loading:")
        print(summary_dict)

        if summary_dict:
            print("Sending the following JSON dictionary to the validate_summary function:")
            print(summary_dict)
            validation_result = validate_summary(summary_dict)
            if validation_result is True:
                print("Validation result: True")
                return summary_dict
            else:
                print("Validation result: False")
                is_response_incomplete_result = is_response_incomplete(
                    summary)  # Store the result of is_response_incomplete() in a variable
                print(
                    f"[4.2] is_response_incomplete() returned: {is_response_incomplete_result}")  # Print the result regardless of its value
                if is_response_incomplete_result:
                    print("[4.1] Incomplete response detected, increasing max_tokens...")
                    max_tokens += 100
                else:
                    print("[4.3] Summary validation failed, retrying...")

                retries -= 1
        else:
            print("[3] Empty summary received, retrying...")
            retries -= 1

    print("[5] Failed to generate a valid summary after 5 attempts. Returning empty JSON structure.")
    return empty_json_structure()


def reconstruct_summaries(summaries):
    def process_value(value):
        if isinstance(value, str):
            return value.strip()
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(v) for v in value]
        return value

    reconstructed_summary = {
        "Overall Summary": "",
        "Module/Library Name": "",
        "Classes": {},
        "Functions/Methods": {},
        "Data Structures": "",
        "Interfaces/APIs": "",
        "Configuration/Environment Variables": {},
        "Error Handling/Logging": "",
        "Data Inputs": "",
        "Data Outputs": ""
    }
    print(f"Input summaries: {summaries}")

    for summary in summaries:
        summary = process_value(summary)
        if not isinstance(summary, dict):
            print(f"Error: summary is not a dictionary: {summary}")
            continue

        for key in summary:
            if key in reconstructed_summary:
                if key in {"Overall Summary", "Module/Library Name", "Data Inputs", "Data Outputs", "Data Structures", "Interfaces/APIs", "Error Handling/Logging"}:
                    if summary[key] and (isinstance(summary[key], str) and summary[key].lower() not in {"none", "n/a", ""}):
                        if not reconstructed_summary[key]:
                            reconstructed_summary[key] = summary[key]
                        else:
                            reconstructed_summary[key] += f", {summary[key]}"
                elif key in {"Classes", "Functions/Methods", "Configuration/Environment Variables"}:
                    if isinstance(summary[key], dict):
                        for item_key, item_value in summary[key].items():
                            if item_value.lower() not in {"none", "n/a", ""}:
                                if item_key not in reconstructed_summary[key]:
                                    reconstructed_summary[key][item_key] = item_value
                            else:
                                print(f"Skipped {item_key} with value {item_value} for key {key}")
                    else:
                        print(f"Error at summary: {summary}, key: {key}")

    # Deduplicate and remove trailing commas and spaces
    for key in {"Overall Summary", "Module/Library Name", "Data Inputs", "Data Outputs", "Data Structures", "Interfaces/APIs", "Error Handling/Logging"}:
        values = reconstructed_summary[key].split(', ')
        deduplicated_values = list(dict.fromkeys(values))
        reconstructed_summary[key] = ', '.join(deduplicated_values).rstrip(", ")

    return reconstructed_summary


def calculate_avg_chars_per_token(text, tokenizer, max_tokens=1024, estimated_ratio=2.5):
    # Print input values and tokenizer object
    # print("Text:", text)
    print("Tokenizer:", tokenizer)
    print("Max tokens:", max_tokens)
    print("Estimated ratio:", estimated_ratio)

    # Tokenize the input text
    tokens = tokenizer.tokenize(text)

    # Calculate the average number of characters per token
    if len(tokens) == 0:
        print("Warning: No tokens found in the text. Returning default value.")
        return estimated_ratio  # return the estimated_ratio or another default value

    avg_chars_per_token = len(text) / len(tokens)

    print(f"Total characters: {len(text)}, Total tokens: {len(tokens)}")  # Add this print statement

    # Reduce the calculated average by 25% to provide some allowance
    reduced_avg_chars_per_token = avg_chars_per_token * 0.75

    return reduced_avg_chars_per_token


def contains_test_keyword(name):
    test_keywords = {"test", "tests"}
    return any(keyword.lower() in name.lower() for keyword in test_keywords)


def chunk_content(file_content, max_tokens):
    tokens = gpt2_tokenizer.encode(file_content)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = gpt2_tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


def get_sample(content, sample_size):
    content_length = len(content)
    if content_length <= sample_size:
        return content

    start_index = (content_length // 2) - (sample_size // 2)
    end_index = start_index + sample_size
    return content[start_index:end_index]


def validate_summary(summary):
    def is_empty_string(s):
        return isinstance(s, str) and s.strip() == ""

    print(f"Received summary: {summary}")
    valid_keys = [
        "Overall Summary",
        "Module/Library Name",
        "ETL Processes",
        "Classes",
        "Functions/Methods",
        "Variables",
        "Data Structures",
        "Interfaces/APIs",
        "Configuration/Environment Variables",
        "Error Handling/Logging",
        "Data Inputs",
        "Data Outputs"
    ]

    invalid_keys = [key for key in summary if key not in valid_keys]
    if invalid_keys:
        print(f"Invalid keys in summary: {invalid_keys}")
        return False

    invalid_types = []
    for key, value in summary.items():
        if not (isinstance(value, dict) or isinstance(value, str)):
            invalid_types.append(key)
        elif is_empty_string(value):
            continue
        elif isinstance(value, str) and is_empty_string(value):
            invalid_types.append(key)

    if invalid_types:
        print(f"Invalid types in summary: {invalid_types}")
        return False

    print("Summary is valid")
    return True


def empty_json_structure():
    return {
        "Overall Summary": "",
        "Module/Library Name": "",
        "Classes": {},
        "Functions/Methods": {},
        "Data Structures": "",
        "Interfaces/APIs": "",
        "Configuration/Environment Variables": {},
        "Error Handling/Logging": "",
        "Data Inputs": "",
        "Data Outputs": ""
    }


def is_response_incomplete(response: str) -> bool:
    print("Calling is_response_incomplete() function...")
    print(f"Analyzing response: {response}")

    # Remove newline characters and white spaces outside of string values
    response = re.sub(r'(?<=\})\s+|\s+(?=\{)', '', response)

    if not response:
        return False

    # Check if the response is missing a closing curly brace
    if response[-1] != '}':
        return True

    # Check if the response has an equal number of opening and closing curly braces
    if response.count('{') != response.count('}'):
        return True

    return False