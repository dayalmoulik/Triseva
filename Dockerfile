FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user with UID 1000 for Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    CHAINLIT_AUTH_SECRET=d96ba0f3d67f7e2a9b3d0a6498ec11f7c11f7c11f7c11f7c11f7c11f7c1 \
    CHAINLIT_COOKIE_SAMESITE=none

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose port 7860 (Hugging Face Spaces default web port)
EXPOSE 7860

# Command to run Chainlit
CMD ["chainlit", "run", "app/chainlit_app.py", "--host", "0.0.0.0", "--port", "7860"]
