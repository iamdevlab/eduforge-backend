# Start from an official, lightweight Python image
FROM python:3.14-slim

# Set the working directory inside the container
WORKDIR /code

# These are necessary for the psycopg2-binary package to function properly.
# If you don't need PostgreSQL support, you can remove this step.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    # Clean up the package lists to keep the image small
    && rm -rf /var/lib/apt/lists/*

# Add the /code directory to Python's import path.
# This allows Gunicorn to find your 'app' module.
ENV PYTHONPATH /code

# --- Install uv ---
# Install uv itself using pip. We'll use this to install our app's dependencies.
RUN pip install uv

# --- Install Dependencies ---
# Copy ONLY the project file. This is a key optimization.
# Docker caches this step and will only re-run it if pyproject.toml changes.
COPY pyproject.toml ./

# Install all dependencies from your [project.dependencies]
# and [project.optional-dependencies] using uv.
# This is the replacement for "pip install -r requirements.txt"
RUN uv pip sync --system pyproject.toml


# --- Copy Application Code ---
# Now that dependencies are installed, copy the rest of your app's code
# (your 'app' folder, 'main.py', etc.)
COPY . /code/

# --- Run the Application ---
# This is the command to start your app.
# It's the same as the Heroku command, but we use the $PORT
# variable that Google Cloud Run provides automatically.
# We are starting with 1 worker to save memory, which should
# fix the crashes you had on Heroku.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker app.main:app