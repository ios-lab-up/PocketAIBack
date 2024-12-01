# Use an official Python base image
FROM python:3.12-bullseye
LABEL authors="Esteban Mayoral, Daniel Peláez & Luis Cedillo"

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc libpq-dev netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to start the application
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "1", "--reload"]
