# === P7 Predictive Maintenance API - container recipe ===
# Mirrors the CI sequence: install deps -> install project -> data -> model -> serve.

# Start FROM an official slim Python 3.12 base (minimal Linux + Python).
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy dependency manifests first (Docker caches this layer; deps only reinstall
# when requirements change, not on every code edit -> faster rebuilds).
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install third-party packages (the runtime requirements).
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project in.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install OUR project so `import src...` works (the CI lesson, in the container).
RUN pip install --no-cache-dir -e .

# Generate the seeded dataset and train the model artifact, so the container
# ships ready-to-serve (the API loads models/anomaly_model.joblib at startup).
RUN python scripts/generate_sensor_data.py && \
    python scripts/train_and_save_model.py

# Document the port the service listens on.
EXPOSE 8000

# Start the API. Bind to 0.0.0.0 so the container accepts external connections
# (127.0.0.1 would only accept connections from INSIDE the container).
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
