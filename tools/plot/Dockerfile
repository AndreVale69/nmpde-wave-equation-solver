FROM python:3.12-alpine

# Create a non-root user to run the application
RUN apk add --no-cache libstdc++ && addgroup -S app && adduser -S app -G app

# Set working directory
WORKDIR /app

# Set environment variables
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED: Ensures that Python output is sent straight to terminal (stdout/stderr) without being buffered
# - PIP_NO_CACHE_DIR: Disables pip's cache to reduce image size
# - STREAMLIT_SERVER_HEADLESS: Runs Streamlit in headless mode
# - STREAMLIT_SERVER_ADDRESS: Binds Streamlit server to all network interfaces
# - STREAMLIT_BROWSER_GATHER_USAGE_STATS: Disables usage stats gathering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER app

# Expose the port Streamlit will run on
EXPOSE 8501

CMD ["streamlit", "run", "mms_viewer.py", "--server.port=8501"]
