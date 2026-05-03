FROM python:3.10-slim
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]