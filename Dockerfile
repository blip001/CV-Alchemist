# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Create a non-root user and switch to it
RUN addgroup --system app && adduser --system --group app

# Install dependencies as root
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code from your context
COPY . .

# Change ownership of the app directory
RUN chown -R app:app /app

# Add the local user's binary directory to the PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Switch to the non-root user
USER app

# Run main.py when the container launches
# The port is dynamically set by Cloud Run via the PORT environment variable.
CMD ["sh", "-c", "exec gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT main:app"]