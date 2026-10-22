FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir . &&     python -m compileall -q tevforge

EXPOSE 8090

CMD ["python", "-m", "tevforge", "serve"]
