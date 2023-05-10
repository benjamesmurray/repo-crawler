import os
import logging
from read_summaries import read_summaries


def generate_breadcrumbs(prefix):
    breadcrumb_links = []
    path_parts = prefix[:-1].split(os.path.sep)

    for i, part in enumerate(path_parts):
        link_target = os.path.sep.join(path_parts[:i + 1])
        breadcrumb_links.append(f"<a href='#{link_target}'>{part}</a>")

    return " > ".join(breadcrumb_links)


def generate_html(summaries, depth=0, prefix=""):
    html = ""

    if isinstance(summaries, str):
        html += f"<h3>{prefix[:-4]}</h3>"
        html += f"<pre><code>{summaries}</code></pre>"
    else:
        for key, value in summaries.items():
            if isinstance(value, dict):
                breadcrumbs = generate_breadcrumbs(prefix)
                html += f"<h1 id='{prefix}{key}'>{breadcrumbs} > {key}</h1>"
                html += "<ul>"
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        html += f"<li><a href='#{prefix}{key}{os.path.sep}{subkey}'>{subkey}</a></li>"
                html += "</ul>"
            html += generate_html(value, depth + 1, f"{prefix}{key}{os.path.sep}")

    return html


def commented_main(summary_output_path="commented_code_output"):

    # Set up logging
    logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    all_summaries = read_summaries(summary_output_path)

    # Generate HTML content
    html_content = generate_html(all_summaries)

    # Combine HTML content
    html_output = f"<html><head><title>Commented Code</title></head><body><h1>Commented Code</h1>{html_content}</body></html>"

    try:
        with open("example_output/commented_code.html", "w", encoding="utf-8") as output_file:
            output_file.write(html_output)
        logging.info("Generated commented code summary")
    except Exception as e:
        logging.error(f"Error writing commented code summary: {e}")


if __name__ == "__main__":
    commented_main()