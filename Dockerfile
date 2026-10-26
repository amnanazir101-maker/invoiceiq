FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ONNX Runtime and OpenCV (behind RapidOCR) need the OpenMP and GLib runtimes.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[app]"
COPY app.py ./
COPY examples ./examples
COPY results ./results

EXPOSE 8000 8501
CMD ["invoiceiq", "serve", "--host", "0.0.0.0"]
