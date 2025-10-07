# Start from an official, lightweight Python image.
# We are using python:3.11-slim as a reliable base.
# Start from an official, lightweight Python image.
FROM python:3.11-slim

# Install system dependencies required by OpenCV (libGL, libXext, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container. All commands will run from /code.
WORKDIR /code

# Copy the file that lists all my dependencies into the container.
# This allows Docker to cache the installation step.
COPY requirements.txt /code/

# Install all the Python dependencies.
# The --no-cache-dir flag just saves space.
RUN pip install --no-cache-dir --upgrade \
    -r /code/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of my application code into the container.
# This includes api.py, src/, db/, benchmarks/, etc.
COPY . /code

# The command to run the application. Uvicorn is our server.
# We tell it to run the 'app' object inside 'api.py'.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]