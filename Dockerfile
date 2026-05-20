FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HF Spaces runs containers as UID 1000 by convention; create matching user
# so uploads/ and output/ remain writable at runtime.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app

COPY --chown=app:app requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=app:app . .

# Writable working dirs for per-session uploads and exported fit results.
RUN mkdir -p uploads output && chown -R app:app uploads output

USER app

# HF Spaces routes traffic to app_port declared in README (7860 here).
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
