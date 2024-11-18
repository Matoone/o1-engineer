import asyncio
import os
import fnmatch
import logging
import sys
import time
import traceback
import gc
from openai import OpenAI
from dotenv import load_dotenv
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
import difflib
import re
from prompt_toolkit.shortcuts.prompt import PromptSession
from model_manager import (
    ModelManager,
    ModelError,
    ModelConfigurationError,
)

MAX_FILE_SIZE = 200 * 1024  # 200 KB file size limit
MAX_TOTAL_SIZE = MAX_FILE_SIZE * 3  # 600 KB total size limit

load_dotenv()

try:
    model_manager = ModelManager()
except (ModelConfigurationError, ModelError) as e:
    logging.error(f"Error initializing model: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Pour afficher en console
        logging.FileHandler("o1_engineer.log"),  # Pour sauvegarder dans un fichier
    ],
)


CREATE_SYSTEM_PROMPT = """You are an advanced o1 engineer designed to create files and folders based on user instructions. Your primary objective is to generate the content of the files to be created as code blocks. Each code block should specify whether it's a file or folder, along with its path.

When given a user request, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to create.
2. Generate Creation Instructions: Provide the content for each file to be created within appropriate code blocks. Each code block should begin with a special comment line that specifies whether it's a file or folder, along with its path.
3. You create full functioning, complete,code files, not just snippets. No approximations or placeholders. FULL WORKING CODE.

IMPORTANT: Your response must ONLY contain the code blocks with no additional text before or after. Do not use markdown formatting outside of the code blocks. Use the following format for the special comment line. Do not include any explanations, additional text:

For folders:
```
### FOLDER: path/to/folder
```

For files:
```language
### FILE: path/to/file.extension
File content goes here...
```

Example of the expected format:

```
### FOLDER: new_app
```

```html
### FILE: new_app/index.html
<!DOCTYPE html>
<html>
<head>
    <title>New App</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

```css
### FILE: new_app/styles.css
body {
    font-family: Arial, sans-serif;
}
```

```javascript
### FILE: new_app/script.js
console.log('Hello, World!');
```

Ensure that each file and folder is correctly specified to facilitate seamless creation by the script."""


CODE_REVIEW_PROMPT = """You are an expert code reviewer. Your task is to analyze the provided code files and provide a comprehensive code review. For each file, consider:

1. Code Quality: Assess readability, maintainability, and adherence to best practices
2. Potential Issues: Identify bugs, security vulnerabilities, or performance concerns
3. Suggestions: Provide specific recommendations for improvements

Format your review as follows:
1. Start with a brief overview of all files
2. For each file, provide:
   - A summary of the file's purpose
   - Key findings (both positive and negative)
   - Specific recommendations
3. End with any overall suggestions for the codebase

Your review should be detailed but concise, focusing on the most important aspects of the code."""


EDIT_INSTRUCTION_PROMPT = """You are an advanced o1 engineer designed to analyze files and provide edit instructions based on user requests. Your task is to:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the File(s): Review the content of the provided file(s).
3. Generate Edit Instructions: Provide clear, step-by-step instructions on how to modify the file(s) to address the user's request.

Your response should be in the following format:

```
File: [file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...

File: [another_file_path]
Instructions:
1. [First edit instruction]
2. [Second edit instruction]
...
```

Only provide instructions for files that need changes. Be specific and clear in your instructions."""


APPLY_EDITS_PROMPT = """
Rewrite an entire file or files using edit instructions provided by another AI.

Ensure the entire content is rewritten from top to bottom incorporating the specified changes.

# Steps

1. **Receive Input:** Obtain the file(s) and the edit instructions. The files can be in various formats (e.g., .txt, .docx).
2. **Analyze Content:** Understand the content and structure of the file(s).
3. **Review Instructions:** Carefully examine the edit instructions to comprehend the required changes.
4. **Apply Changes:** Rewrite the entire content of the file(s) from top to bottom, incorporating the specified changes.
5. **Verify Consistency:** Ensure that the rewritten content maintains logical consistency and cohesiveness.
6. **Final Review:** Perform a final check to ensure all instructions were followed and the rewritten content meets the quality standards.
7. Do not include any explanations, additional text, or code block markers (such as ```html or ```).

Provide the output as the FULLY NEW WRITTEN file(s).
NEVER ADD ANY CODE BLOCK MARKER AT THE BEGINNING OF THE FILE OR AT THE END OF THE FILE (such as ```html or ```). 

"""


PLANNING_PROMPT = """You are an AI planning assistant. Your task is to create a detailed plan based on the user's request. Consider all aspects of the task, break it down into steps, and provide a comprehensive strategy for accomplishment. Your plan should be clear, actionable, and thorough."""

prompt_session = PromptSession()
last_ai_response = None
conversation_history = []


def is_binary_file(file_path):
    """Check if a file is binary by reading a small portion of it."""
    try:
        with open(file_path, "rb") as file:
            chunk = file.read(1024)  # Read the first 1024 bytes
            if b"\0" in chunk:
                return True  # File is binary if it contains null bytes
            # Use a heuristic to detect binary content
            text_characters = bytearray(
                {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100))
            )
            non_text = chunk.translate(None, text_characters)
            if len(non_text) / len(chunk) > 0.30:
                return True  # Consider binary if more than 30% non-text characters
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return True  # Assume binary if an error occurs
    return False  # File is likely text


# Load .gitignore patterns if in a git repository
def load_gitignore_patterns(directory):
    gitignore_path = os.path.join(directory, ".gitignore")
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def should_ignore(file_path, patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False


def add_file_to_context(file_path, added_files, action="to the chat context"):
    """Add a file to the given dictionary, applying exclusion rules."""
    excluded_dirs = {
        "__pycache__",
        ".git",
        "node_modules",
        "venv",
        "env",
        ".vscode",
        ".idea",
        "dist",
        "build",
        "__mocks__",
        "coverage",
        ".pytest_cache",
        ".mypy_cache",
        "logs",
        "temp",
        "tmp",
        "secrets",
        "private",
        "cache",
        "addons",
    }

    gitignore_patterns = []
    if os.path.exists(".gitignore"):
        gitignore_patterns = load_gitignore_patterns(".")

    if not os.path.exists(file_path):
        logging.error(f"Path does not exist: {file_path}")
        print(colored(f"Path not found: {file_path}", "red"))
        return

    if not os.access(file_path, os.R_OK):
        logging.error(f"No read permission for: {file_path}")
        print(colored(f"Cannot read file/directory: {file_path}", "red"))
        return

    if os.path.isfile(file_path):
        if any(ex_dir in file_path for ex_dir in excluded_dirs):
            logging.info(f"Skipped excluded directory file: {file_path}")
            return
        if gitignore_patterns and should_ignore(file_path, gitignore_patterns):
            logging.info(f"Skipped file matching .gitignore pattern: {file_path}")
            return
        if is_binary_file(file_path):
            logging.info(f"Skipped binary file: {file_path}")
            return

        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logging.error(
                f"File {file_path} exceeds maximum size limit of {MAX_FILE_SIZE/1024:.1f}MB"
            )
            return

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                added_files[file_path] = content
                logging.info(f"Added {file_path} {action}.")
        except IOError as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            print(colored(f"Error reading file {file_path}", "red"))
            return
        except MemoryError:
            logging.error(f"Out of memory while reading file {file_path}")
            print(colored("Out of memory error - file too large", "red"))
            return
    else:
        logging.error(f"{file_path} is not a file.")


async def apply_modifications(new_content, file_path):
    try:
        with open(file_path, "r") as file:
            old_content = file.read()

        if old_content.strip() == new_content.strip():
            print(colored(f"No changes detected in {file_path}", "red"))
            return True

        display_diff(old_content, new_content, file_path)

        confirm = (
            (
                await prompt_session.prompt_async(
                    f"Apply these changes to {file_path}? (yes/no): ",
                    style=Style.from_dict({"prompt": "orange"}),
                )
            )
            .strip()
            .lower()
        )
        if confirm == "yes":
            with open(file_path, "w") as file:
                file.write(new_content)
            logging.info(f"Modifications applied to {file_path} successfully.")
            return True
        else:
            logging.info(f"User chose not to apply changes to {file_path}.")
            return False

    except Exception as e:
        logging.error(f"Error applying modifications to {file_path}: {e}")
        return False


def display_diff(old_content, new_content, file_path):
    diff = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
            n=5,
        )
    )
    if not diff:
        logging.info(f"No changes detected in {file_path}")
        return
    console = Console()
    table = Table(title=f"Diff for {file_path}")
    table.add_column("Status", style="bold")
    table.add_column("Line")
    table.add_column("Content")
    line_number = 1
    for line in diff:
        status = line[0]
        content = line[2:].rstrip()
        if status == " ":
            continue  # Skip unchanged lines
        elif status == "-":
            table.add_row("Removed", str(line_number), content, style="red")
        elif status == "+":
            table.add_row("Added", str(line_number), content, style="green")
        line_number += 1
    console.print(table)


async def apply_creation_steps(creation_response, added_files, retry_count=0):
    max_retries = 3
    try:
        code_blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", creation_response)
        if not code_blocks:
            raise ValueError("No code blocks found in the AI response.")

        logging.info("Successfully extracted code blocks from creation response.")

        for code in code_blocks:
            info_match = re.match(r"### (FILE|FOLDER): (.+)", code.strip())

            if info_match:
                item_type, path = info_match.groups()

                if not os.path.exists(path):
                    logging.error(f"Path does not exist: {path}")
                    print(colored(f"Path not found: {path}", "red"))
                    continue

                if not os.access(path, os.W_OK):
                    logging.error(f"No write permission for: {path}")
                    print(colored(f"Cannot write to file/directory: {path}", "red"))
                    continue

                if item_type == "FOLDER":
                    os.makedirs(path, exist_ok=True)
                    logging.info(f"Folder created: {path}")
                elif item_type == "FILE":
                    file_content = re.sub(r"### FILE: .+\n", "", code, count=1).strip()

                    directory = os.path.dirname(path)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                        logging.info(f"Folder created: {directory}")

                    with open(path, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    logging.info(f"File created: {path}")
            else:
                logging.error(
                    "Could not determine the file or folder information from the code block."
                )
                continue

        return True

    except ValueError as e:
        if retry_count < max_retries:
            logging.warning(
                f"Creation parsing failed: {str(e)}. Retrying... (Attempt {retry_count + 1})"
            )
            error_message = f"{str(e)} Please provide the creation instructions again using the specified format."
            time.sleep(2**retry_count)
            new_response = await chat_with_ai(
                error_message, is_edit_request=False, added_files=added_files
            )
            if new_response:
                return await apply_creation_steps(
                    new_response, added_files, retry_count + 1
                )
            else:
                return False
        else:
            logging.error(
                f"Failed to parse creation instructions after multiple attempts: {str(e)}"
            )
            print("Creation response that failed to parse:")
            print(creation_response)
            return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during creation: {e}")
        return False


def parse_edit_instructions(response):
    instructions = {}
    current_file = None
    current_instructions = []

    for line in response.split("\n"):
        if line.startswith("File: "):
            if current_file:
                instructions[current_file] = "\n".join(current_instructions)
            current_file = line[6:].strip()
            current_instructions = []
        elif line.strip() and current_file:
            current_instructions.append(line.strip())

    if current_file:
        instructions[current_file] = "\n".join(current_instructions)

    return instructions


async def apply_edit_instructions(edit_instructions, original_files):
    modified_files = {}
    for file_path, content in original_files.items():
        if file_path in edit_instructions:
            instructions = edit_instructions[file_path]
            prompt = f"{APPLY_EDITS_PROMPT}\n\nOriginal File: {file_path}\nContent:\n{content}\n\nEdit Instructions:\n{instructions}\n\nUpdated File Content:"
            response = await chat_with_ai(prompt, is_edit_request=True)
            if response:
                modified_files[file_path] = response.strip()
        else:
            modified_files[file_path] = content
    return modified_files


async def chat_with_ai(
    user_message, is_edit_request=False, retry_count=0, added_files=None
):
    global last_ai_response, conversation_history
    try:
        if added_files:
            file_context = "Added files:\n"
            for file_path, content in added_files.items():
                file_context += f"File: {file_path}\nContent:\n{content}\n\n"
            user_message = f"{file_context}\n{user_message}"

        if not is_edit_request:
            history = "\n".join(
                [
                    f"User: {msg}" if i % 2 == 0 else f"AI: {msg}"
                    for i, msg in enumerate(conversation_history)
                ]
            )
            if history:
                user_message = f"{history}\nUser: {user_message}"

        if is_edit_request:
            prompt = EDIT_INSTRUCTION_PROMPT if retry_count == 0 else APPLY_EDITS_PROMPT
            message_content = f"{prompt}\n\nUser request: {user_message}"
        else:
            message_content = user_message

        messages = [{"role": "user", "content": message_content}]

        if is_edit_request and retry_count == 0:
            logging.info(f"Sending edit request to AI {model_manager.full_model_name}.")
        else:
            logging.info(f"Sending general query to AI {model_manager.full_model_name}")

        response = await model_manager.chat_completion(messages=messages)
        logging.info(f"Received response from AI {model_manager.full_model_name}")
        last_ai_response = response["content"]

        if not is_edit_request:
            conversation_history.append(user_message)
            conversation_history.append(last_ai_response)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

        return last_ai_response
    except Exception as e:
        logging.error(
            f"Error while communicating with AI {model_manager.full_model_name}: {e}"
        )
        logging.error(traceback.format_exc())
        return None


async def main():
    global last_ai_response, conversation_history

    print(colored("o1 engineer is ready to help you.", "cyan"))
    print("\nAvailable commands:")
    print(
        f"{colored('/edit', 'magenta'):<10} {colored('Edit files or directories (followed by paths)', 'dark_grey')}"
    )
    print(
        f"{colored('/create', 'magenta'):<10} {colored('Create files or folders (followed by instructions)', 'dark_grey')}"
    )
    print(
        f"{colored('/add', 'magenta'):<10} {colored('Add files or folders to context', 'dark_grey')}"
    )
    print(
        f"{colored('/debug', 'magenta'):<10} {colored('Print the last AI response', 'dark_grey')}"
    )
    print(
        f"{colored('/reset', 'magenta'):<10} {colored('Reset chat context and clear added files', 'dark_grey')}"
    )
    print(
        f"{colored('/review', 'magenta'):<10} {colored('Review code files (followed by file paths)', 'dark_grey')}"
    )
    print(
        f"{colored('/planning', 'magenta'):<10} {colored('Generate a detailed plan based on your request', 'dark_grey')}"
    )
    print(
        f"{colored('/quit', 'magenta'):<10} {colored('Exit the program', 'dark_grey')}"
    )

    style = Style.from_dict(
        {
            "prompt": "cyan",
        }
    )

    files = [f for f in os.listdir(".") if os.path.isfile(f)]

    completer = WordCompleter(
        [
            "/edit",
            "/create",
            "/add",
            "/quit",
            "/debug",
            "/reset",
            "/review",
            "/planning",
        ]
        + files,
        ignore_case=True,
    )

    added_files = {}
    file_contents = {}

    while True:
        print()
        user_input = (
            await prompt_session.prompt_async("You: ", style=style, completer=completer)
        ).strip()

        if user_input.lower() == "/quit":
            logging.info("User exited the program.")
            break

        elif user_input.lower() == "/debug":
            if last_ai_response:
                print(colored("Last AI Response:", "blue"))
                print(last_ai_response)
            else:
                print(colored("No AI response available yet.", "red"))

        elif user_input.lower() == "/reset":
            conversation_history.clear()
            added_files.clear()
            file_contents.clear()
            last_ai_response = None
            gc.collect()
            logging.info("Memory cleaned up and contexts reset")
            print(colored("Chat context and added files have been reset.", "green"))

        elif user_input.startswith("/add"):
            paths = user_input.split()[1:]
            if not paths:
                logging.warning("User issued /add without file or folder paths.")
                continue

            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, added_files)
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        dirs[:] = [
                            d
                            for d in dirs
                            if d not in {"__pycache__", ".git", "node_modules"}
                        ]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(file_path, added_files)
                else:
                    logging.error(f"{path} is neither a file nor a directory.")

            total_size = sum(len(content) for content in added_files.values())
            if total_size > MAX_TOTAL_SIZE:
                logging.error(
                    f"Total size of all files exceeds maximum allowed limit of {MAX_TOTAL_SIZE/1024:.1f}KB"
                )
                print(
                    colored(
                        "Total size of files is too large. Please reduce the number/size of files.",
                        "red",
                    )
                )
                added_files.clear()
                break

        elif user_input.startswith("/edit"):
            paths = user_input.split()[1:]
            if not paths:
                logging.warning("User issued /edit without file or folder paths.")
                continue
            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, added_files)
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        dirs[:] = [
                            d
                            for d in dirs
                            if d not in {"__pycache__", ".git", "node_modules"}
                        ]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(file_path, added_files)
                else:
                    logging.error(f"{path} is neither a file nor a directory.")
            if not added_files:
                print(colored("No valid files to edit.", "red"))
                continue
            edit_instruction = (
                await prompt_session.prompt_async(
                    f"Edit Instruction for all files: ", style=style
                )
            ).strip()

            edit_request = f"""User request: {edit_instruction}

Files to modify:
"""
            for file_path, content in added_files.items():
                edit_request += f"\nFile: {file_path}\nContent:\n{content}\n\n"

            ai_response = await chat_with_ai(
                edit_request, is_edit_request=True, added_files=added_files
            )

            if ai_response:
                print("o1 engineer: Here are the suggested edit instructions:")
                rprint(Markdown(ai_response))

                confirm = (
                    (
                        await prompt_session.prompt_async(
                            "Do you want to apply these edit instructions? (yes/no): ",
                            style=style,
                        )
                    )
                    .strip()
                    .lower()
                )
                if confirm == "yes":
                    edit_instructions = parse_edit_instructions(ai_response)
                    modified_files = await apply_edit_instructions(
                        edit_instructions, added_files
                    )
                    for file_path, new_content in modified_files.items():
                        await apply_modifications(new_content, file_path)
                else:
                    logging.info("User chose not to apply edit instructions.")

        elif user_input.startswith("/create"):
            creation_instruction = user_input[7:].strip()
            if not creation_instruction:
                logging.warning("User issued /create without instructions.")
                continue

            create_request = (
                f"{CREATE_SYSTEM_PROMPT}\n\nUser request: {creation_instruction}"
            )
            ai_response = await chat_with_ai(
                create_request, is_edit_request=False, added_files=added_files
            )

            if ai_response:
                while True:
                    print("o1 engineer: Here is the suggested creation structure:")
                    rprint(Markdown(ai_response))

                    confirm = (
                        (
                            await prompt_session.prompt_async(
                                "Do you want to execute these creation steps? (yes/no): ",
                                style=style,
                            )
                        )
                        .strip()
                        .lower()
                    )
                    if confirm == "yes":
                        success = await apply_creation_steps(ai_response, added_files)
                        if success:
                            break
                        else:
                            retry = (
                                (
                                    await prompt_session.prompt_async(
                                        "Creation failed. Do you want the AI to try again? (yes/no): ",
                                        style=style,
                                    )
                                )
                                .strip()
                                .lower()
                            )
                            if retry != "yes":
                                break
                            ai_response = await chat_with_ai(
                                "The previous creation attempt failed. Please try again with a different approach.",
                                is_edit_request=False,
                                added_files=added_files,
                            )
                    else:
                        logging.info("User chose not to execute creation steps.")
                        break

        elif user_input.startswith("/review"):
            paths = user_input.split()[1:]
            if not paths:
                logging.warning("User issued /review without file or folder paths.")
                continue

            file_contents = {}
            for path in paths:
                if os.path.isfile(path):
                    add_file_to_context(path, file_contents, action="to review")
                elif os.path.isdir(path):
                    for root, dirs, files_in_dir in os.walk(path):
                        dirs[:] = [
                            d
                            for d in dirs
                            if d not in {"__pycache__", ".git", "node_modules"}
                        ]
                        for file in files_in_dir:
                            file_path = os.path.join(root, file)
                            add_file_to_context(
                                file_path, file_contents, action="to review"
                            )
                else:
                    logging.error(f"{path} is neither a file nor a directory.")

            if not file_contents:
                print(colored("No valid files to review.", "red"))
                continue

            review_request = f"{CODE_REVIEW_PROMPT}\n\nFiles to review:\n"
            for file_path, content in file_contents.items():
                review_request += f"\nFile: {file_path}\nContent:\n{content}\n\n"

            print(colored("Analyzing code and generating review...", "magenta"))
            ai_response = await chat_with_ai(
                review_request, is_edit_request=False, added_files=added_files
            )

            if ai_response:
                print()
                print(colored("Code Review:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("Provided code review for requested files.")

        elif user_input.startswith("/planning"):
            planning_instruction = user_input[9:].strip()
            if not planning_instruction:
                logging.warning("User issued /planning without instructions.")
                continue
            planning_request = (
                f"{PLANNING_PROMPT}\n\nUser request: {planning_instruction}"
            )
            ai_response = await chat_with_ai(
                planning_request, is_edit_request=False, added_files=added_files
            )
            if ai_response:
                print()
                print(colored("o1 engineer: Here is your detailed plan:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("Provided planning response to user.")
            else:
                logging.error("AI failed to generate a planning response.")

        else:
            ai_response = await chat_with_ai(
                user_message=user_input, added_files=added_files
            )
            if ai_response:
                print()
                print(colored("o1 engineer:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("Provided AI response to user query.")


if __name__ == "__main__":
    asyncio.run(main())
