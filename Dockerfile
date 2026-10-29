# Works for Hugging Face Spaces (Docker SDK) and Render
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "gunicorn app.app:app --bind 0.0.0.0:${PORT} --timeout 120"]
