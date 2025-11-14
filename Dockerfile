# Hurricane Harvey Building Damage Classification - Inference Server
# Dockerfile for x86 architecture

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY app.py .
COPY best_model.h5 .
COPY best_model_info.json .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
