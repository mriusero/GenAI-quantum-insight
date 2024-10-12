# Use a slim Python image as the base
FROM python:3.9.13-slim

# Set the working directory+
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev

# Copy necessary files into the container
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="$HOME/.local/bin:$PATH" && \
    ~/.local/bin/poetry config virtualenvs.create true  # Optional: create dependencies in the container's global environment

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Install project dependencies via Poetry
RUN poetry install # --no-dev  Optional: install only production dependencies

# Copy the rest of the source code into the container
COPY . /app/

# Téléchargement du modèle en_core_web_sm de spaCy
RUN poetry run python -m spacy download en_core_web_sm

# Expose the port used by Streamlit
EXPOSE 8501

# Tells Docker how to test a container to check that it is still working.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

RUN mkdir -p ~/.streamlit/ && \
    echo "[general]"  > ~/.streamlit/credentials.toml && \
    echo "email = \"\""  >> ~/.streamlit/credentials.toml

# Set the default command to start the application
ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
