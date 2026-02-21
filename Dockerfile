# 1. Start with a lightweight Python base image
FROM python:3.13.7

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies first
# (Doing this first saves time when rebuilding the image later)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the essential folders into the container
COPY models/ ./models/
COPY src/ ./src/
COPY api/ ./api/

# 5. Expose the port that FastAPI runs on
EXPOSE 8000

# 6. The command to run when the container starts
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]