FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r server/requirements.txt

# Set default environment variables for scaling
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=10
ENV PORT=8000

EXPOSE $PORT

CMD ["sh", "-c", "python -m server.app --port $PORT"]
