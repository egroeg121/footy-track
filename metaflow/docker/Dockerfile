# Minimal Metaflow runtime image
FROM python:3.12-slim

RUN pip install --no-cache-dir metaflow

WORKDIR /app
COPY . /app

# Default command can be overridden at runtime
CMD ["bash"]
