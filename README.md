# GenAI-quantum-insight
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

- **Python 3.10+**: This project uses Python, so you'll need to have Python installed. You can download it from [python.org](https://www.python.org/).
- **Poetry**: This project uses Poetry for dependency management. Install it by following the instructions at [python-poetry.org](https://python-poetry.org/docs/#installation).
- **Docker** (Optional): If you prefer to run the project in a Docker container, ensure Docker is installed. Instructions can be found at [docker.com](https://www.docker.com/).

## Installation

Follow these steps to install and set up the project:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mriusero/projet-sda-mlops
   cd projet-sda-mlops
   ```

2. **Install Dependencies:**

   Using Poetry:

   ```bash
   poetry install
   ```

   This will create a virtual environment and install all dependencies listed in `pyproject.toml`.

3. **Activate the Virtual Environment:**

   If Poetry does not automatically activate the virtual environment, you can activate it manually:

   ```bash
   poetry shell
   ```

## Usage

You can run the application locally or inside a Docker container.

### Running Locally

To run the application locally, execute the following command:

```bash
python streamlit run app.py
```

### Running with Docker

1. **Build the Docker Image:**

   ```bash
   docker build -t streamlit .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8501:8501 streamlit
   ```
   
   This will start the application, and you can access it in your web browser at `http://localhost:8501`.

## Project Structure

The project follows a modular structure, with the following directories and files:

```
├── Dockerfile
├── LICENSE
├── README.md
├── app.py
├── database
│   └── arxiv_data.db
├── poetry.lock
├── pyproject.toml
├── src
│   ├── __init__.py
│   └── app
│       ├── __init__.py
│       ├── components
│       │   ├── __init__.py
│       │   ├── repo_button.py
│       │   └── utils.py
│       ├── features
│       │   ├── __init__.py
│       │   ├── arxiv_data_manager
│       │   │   ├── __init__.py
│       │   │   ├── arxiv_client.py
│       │   │   ├── arxiv_db.py
│       │   │   ├── arxiv_parser.py
│       │   │   └── search_and_update.py
│       │   ├── research_assistant
│       │   │   ├── __init__.py
│       │   │   ├── checkpoints
│       │   │   │   ├── processed_pdfs.pkl
│       │   │   │   └── vector_store
│       │   │   │       ├── 973d1715-ae69-4cee-878c-dcb8a8f27efa
│       │   │   │       │   ├── data_level0.bin
│       │   │   │       │   ├── header.bin
│       │   │   │       │   ├── index_metadata.pickle
│       │   │   │       │   ├── length.bin
│       │   │   │       │   └── link_lists.bin
│       │   │   │       └── chroma.sqlite3
│       │   │   ├── document_processing.py
│       │   │   ├── processing
│       │   │   ├── qa_system.py
│       │   │   ├── skeleton.py
│       │   │   └── utilities
│       │   └── visualization
│       │       ├── __init__.py
│       │       └── visualize.py
│       ├── layout
│       │   ├── __init__.py
│       │   ├── layout0.py
│       │   ├── layout1.py
│       │   ├── layout2.py
│       │   ├── layout3.py
│       │   ├── layout4.py
│       │   ├── layout5.py
│       │   └── layout6.py
│       ├── streamlit_app.py
│       └── styles.css
└── tests
    └── test.py

16 directories, 42 files
```
## License
This project is licensed under the terms of the [MIT License](LICENSE).
