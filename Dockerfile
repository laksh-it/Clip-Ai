# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 so that Hugging Face Spaces can access your app
EXPOSE 8080

# Set the port environment variable
ENV PORT 8080

# Start your Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "clip_api:app"]
