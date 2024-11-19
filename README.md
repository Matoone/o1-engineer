
# o1-engineer
This project is forked from [Doriandarko's o1-engineer](https://github.com/Doriandarko/o1-engineer). Special thanks to Doriandarko for setting up this project.

o1-engineer is a command-line tool for automating the creation, editing, and reviewing of project files using OpenAI's API or other models. It provides advanced workflows for planning and generating project structures.

## ✨ Features

- **Automated Code Generation**: Generate code for your projects effortlessly.

- **File Management**: Add, edit, and manage project files directly from the command line.

- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

- **Conversation History**: Save and reset conversation histories as needed.

- **Code Review**: Analyze and review code files for quality and suggestions.

- **Enhanced File and Folder Management**: The `/add` and `/edit` commands now support adding and modifying both files and folders, providing greater flexibility in managing your project structure.

- **Project Planning**: Introducing the `/planning` command, which allows users to create comprehensive project plans that can be used to generate files and directories systematically.

- **Multiple LLM provider support**: Supports openAi, anthropic and ollama models.

- **Multiple models support**: Supports setting up different models for each agent.

## Installation

### Prerequisites

- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)
- **API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/) or [Anthropic](https://www.anthropic.com/api) or use a local model without API Key with [Ollama](https://ollama.com/).

### Steps

1. Clone the repository:

    ```bash
    git clone git@github.com:Matoone/o1-engineer.git
    ```

2. Navigate to the project directory:

    ```bash
    cd o1-engineer
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure APIs:

    Add your API key(s) in `.env` file using `.env.template`.
    Setup your models in `.env`.
    You might need to adjust `MODEL_CONFIG` in `model_manager.py` to use a new model.

## Usage

Launch the application:

```bash
python o1-eng.py
```

### Available Commands

- `/edit {file_or_folder}`: Edit files or folders
- `/create {instructions}`: Create files or folders
- `/add {file_or_folder}`: Add files or folders to context
- `/planning {instructions}`: Plan project structure and tasks
- `/review {file_or_folder}`: Review and analyze code files for quality and potential improvements
- `/debug`: Print the last AI response
- `/reset`: Reset chat context and clear added files
- `/quit`: Exit the program

### Advanced Workflows

1. **Planning the Project**:

    ```bash
    You: /planning Create a basic web application with the following structure:
    - A frontend folder containing HTML, CSS, and JavaScript files.
    - A backend folder with server-side scripts.
    - A README.md file with project documentation.
    ```

2. **Creating the Project Structure**:

    ```bash
    You: /create Generate the project structure based on the above plan.
    ```

### Examples

```bash
You: /add src/main.py src/utils/helper.py src/models/
You: /planning Outline a RESTful API project with separate folders for models, views, and controllers.
You: /create Set up the basic structure for a RESTful API project with models, views, and controllers folders, including initial files, in folder ./my-folder.
You: /edit src/main.py src/models/user.py src/views/user_view.py
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## Acknowledgments

- OpenAI for providing the powerful API.
- Anthropic for another powerful language model (optional).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Doriandarko/o1-engineer&type=Date)](https://star-history.com/#Doriandarko/o1-engineer&Date)
