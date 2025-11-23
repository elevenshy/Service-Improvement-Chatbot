# Use the official lightweight Python image.

# https://hub.docker.com/_/python
# This lien pecifies the base image for the Docker container
# It uses the official Python image from Docker Hub, 
# specifically the version 3.12 with the "slim" variant
# which is a smaller, more lightweight version of the full Python image.
FROM python:3.13-slim

# This ensures that Python output is not buffered, 
# which is useful for real-time logging and debugging.
ENV PYTHONUNBUFFERED True
# This line is needed for the app to work in CStack 
RUN groupadd --gid 1001 app && useradd --uid 1001 --gid 1001 -ms /bin/bash app

#  Sets the working directory for the container to `/home/app`. All subsequent commands will be run from this directory.
WORKDIR /home/app
# Copies the `requirements.txt` file from the local machine to the current working directory in the container (`/home/app`).
COPY requirements.txt ./

RUN pip install -r requirements.txt
# This line is needed for the app to work in CStack 
USER 1001

# Copy local code to the container image.
#  Copies all files from the local directory to the current working directory in the container (`/home/app`), and changes the ownership of the copied files
COPY --chown=app:app . ./

# (Optional) Add any additional commands here

# Run the web service on container startup.
# Informs Docker that the container will listen on port 8501 at runtime. 
# This is used for documentation purposes and does not actually publish the port.
EXPOSE 8501
# Specifies the command to run when the container starts. In this case, it runs a Streamlit application using the `main.py` script.
CMD streamlit run main.py