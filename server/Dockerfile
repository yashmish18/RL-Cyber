FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r server/requirements.txt

EXPOSE 8000

CMD ["python", "-m", "server.app", "--port", "8000"]
