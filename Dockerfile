# Stage 1: Build React Frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
COPY --from=frontend-builder /app/dist ./dist/
COPY --from=frontend-builder /app/dist ./frontend/dist/

EXPOSE 8000
ENV PORT=8000

CMD ["python", "main.py"]

