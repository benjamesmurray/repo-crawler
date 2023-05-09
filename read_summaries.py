import os


def read_summaries(output_folder):
    summaries = {}

    for foldername, subfolders, filenames in os.walk(output_folder):
        for filename in filenames:
            if filename.endswith(".txt") or filename.endswith(".py"):
                file_path = os.path.join(foldername, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    summary = file.read()

                relative_path = os.path.relpath(file_path, output_folder)
                path_parts = relative_path.split(os.path.sep)

                current_level = summaries
                for part in path_parts[:-1]:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]

                current_level[path_parts[-1]] = summary

    return summaries
