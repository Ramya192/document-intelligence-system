FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]