# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies required for training
# For example, if you're using a requirements.txt file:
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Define a volume to save the trained models
VOLUME /app/models

# Command to run when the container starts
CMD ["python", "exp.py"]


