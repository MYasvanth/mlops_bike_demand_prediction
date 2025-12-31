# Deployment Options for Bike Demand MLOps API

This directory contains multiple deployment strategies for the Bike Demand Prediction API, implementing senior-level ML engineering practices.

## Deployment Options

### 1. Docker Containerization

**Files:**
- `Dockerfile`: Multi-stage build for optimized production image
- `docker-compose.yml`: Local development and testing setup

**Features:**
- Health checks
- Non-root user execution
- Minimal base image
- Volume mounts for models and configs

**Usage:**
```bash
# Build and run locally
docker-compose up --build

# Access API at http://localhost:8000
```

### 2. Serverless (AWS Lambda)

**Files:**
- `serverless.yml`: Serverless Framework configuration
- `handler.py`: Lambda handler functions

**Features:**
- Auto-scaling
- Pay-per-use pricing
- S3 integration for model storage
- CORS enabled

**Prerequisites:**
- AWS CLI configured
- Serverless Framework installed

**Usage:**
```bash
# Deploy to AWS
serverless deploy

# Test endpoint
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/predict \
  -H "Content-Type: application/json" \
  -d '{"temp": 0.5, "atemp": 0.6, "hum": 0.7, "windspeed": 0.2, "season": 1, "yr": 0, "mnth": 1, "holiday": 0, "weekday": 1, "workingday": 1, "weathersit": 1}'
```

### 3. Kubernetes

**Files:**
- `k8s-deployment.yaml`: Complete K8s deployment manifest

**Features:**
- Horizontal Pod Autoscaling
- Rolling updates
- Persistent storage for models
- Ingress for external access
- Health and readiness probes
- Resource limits and requests

**Usage:**
```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Access via ingress or load balancer
```

### 4. FastAPI Standalone

**Files:**
- `app.py`: FastAPI application with automatic OpenAPI docs
- `requirements.txt`: Minimal dependencies for deployment

**Features:**
- Automatic API documentation at `/docs`
- Pydantic models for request/response validation
- Async support
- Built-in health checks

**Usage:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

All deployment options expose the same API:

- `GET /health`: Health check
- `POST /predict`: Make predictions
- `GET /model-info`: Model metadata

## Model Serving Best Practices

- **Versioning**: Models are versioned and stored separately
- **Monitoring**: Health checks and metrics endpoints
- **Security**: Input validation and rate limiting
- **Scalability**: Auto-scaling configurations
- **Observability**: Structured logging and error handling

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to trained model | `models/bike_demand_model.joblib` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | `http://localhost:5000` |

## Monitoring and Alerting

- Health checks every 30 seconds
- Automatic restarts on failure
- Resource usage monitoring
- Log aggregation (integrate with ELK stack)

## Security Considerations

- Non-root container execution
- Minimal attack surface
- Input validation with Pydantic
- CORS configuration for web clients
- API authentication (add as needed)

## Performance Optimization

- Model caching in memory
- Batch prediction support (future enhancement)
- Connection pooling
- Resource limits to prevent OOM
