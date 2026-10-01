FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install nodejs for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy requirements and setup
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY reports/ ./reports/
COPY data/ ./data/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir -e .

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "medipredict.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
