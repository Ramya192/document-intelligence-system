FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_OFFLINE=0
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}