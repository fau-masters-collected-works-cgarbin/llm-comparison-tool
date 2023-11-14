FROM python:3.11-slim

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

# Install only necessary packages
#  - curl: for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Where the configuration files will be mounted
RUN mkdir -p /app/config \
    && chown -R appuser:appuser /app

USER appuser

# Python packages will be installed under USER's home directory
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install packages globally, not under the USER, to make them available in the PATH
COPY requirements.txt .
# Use only the main dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py .

# DO NOT COPY the .env file
# Create one on the same directory from where you run the container
# >> DO NOT DO THIS: COPY .env .

# Must match the port used in the other commands below
ENV PORT 8501
EXPOSE $PORT

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Mount the current directory when running: docker run -v $(pwd):/app ... or use the docker-compose.yml file
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
