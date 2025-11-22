# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed for building packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements_v2.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_v2.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Chatbot.py when the container launches
CMD ["streamlit", "run", "Chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
