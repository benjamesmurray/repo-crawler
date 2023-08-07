# repo-crawler

# Instructions

Update the repo URL in the main.py and the create_commented_code.py before running them. 
To generate summaries and commented code and then generate HTML's comprising of these outputs, run the following commands:

1. python main.py: running this will download the repo, and recreate the folder structure based on the file type rules (initially just python files) and create a text file for each file it finds with a summary suitable for a low level design.
2. python create_commented_code.py: running this will use the downloaded report to create a second folder structure based on the same file type rules, and create a text file for each file it finds with a commented version of that file.
3. python create_overall_summary_single_HTML.py: running this will take the folder structure and text files output from step 1 (LLD's) and create a single HTML file from it that can more easily be shared by email etc.
4. python create_commented_code_single_HTML.py: running this will take the folder structure and text files output from step 2 (commented code) and create a single HTML file from it that can more easily be shared by email etc.