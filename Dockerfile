# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files to the container
COPY requirements.txt /app/
COPY src /app/src/

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app/src

# Make port 80 available to the world outside this container (optional if you're exposing a service)
EXPOSE 80

# Command to run your main script
CMD ["python", "src/main.py"]
