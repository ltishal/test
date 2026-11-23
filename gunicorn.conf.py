"""Production settings for Railway and other Gunicorn hosts."""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
# One worker keeps the in-process LLM semaphore effective across all requests.
# Use threads for simultaneous users while model calls are bounded by LLM_MAX_CONCURRENT.
workers = 1
worker_class = "gthread"
threads = max(4, min(32, int(os.getenv("WEB_THREADS", "8"))))
timeout = max(
    int(os.getenv("GUNICORN_TIMEOUT", "360")),
    int(float(os.getenv("LLM_TIMEOUT", "300")) +
        float(os.getenv("LLM_QUEUE_TIMEOUT", "120")) + 30),
)
keepalive = 5
accesslog = "-"
errorlog = "-"
