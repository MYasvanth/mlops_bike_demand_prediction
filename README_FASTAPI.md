# Bike Demand Prediction API - FastAPI Implementation

This document describes the FastAPI implementation of the Bike Demand Prediction API, which provides modern, async REST endpoints with automatic OpenAPI documentation.

## Features

- **Modern Async API**: Built with FastAPI for high performance and async support
- **Automatic Documentation**: Interactive API docs at `/docs` and `/redoc`
- **Request Validation**: Pydantic models ensure data validation and type safety
- **Error Handling**: Comprehensive error responses with proper HTTP status codes
- **Performance Monitoring**: Built-in processing time tracking
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Checks**: Dedicated health check endpoint

## API Endpoints

### Base URL
- **FastAPI**: `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and welcome message |
| GET | `/health` | Health check with system status |
| GET | `/welcome` | Welcome message with logging |
| POST | `/predict` | Single bike demand prediction |
| POST | `/predict/batch` | Batch predictions (up to 100) |
| GET | `/model/info` | Model information and metadata |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |
| GET | `/openapi.json` | OpenAPI JSON specification |

## Request/Response Models

### Prediction Request
```json
{
  "season": 1,
  "yr": 0,
  "mnth": 1,
  "holiday": 0,
  "weekday": 1,
  "workingday": 1,
  "weathersit": 1,
  "temp": 0.25,
  "atemp": 0.3,
  "hum": 0.6,
  "windspeed": 0.2
}
```

### Prediction Response
```json
{
  "prediction": 609.24,
  "input_features": {...},
  "model_version": "1.0.0",
  "timestamp": "2025-11-05T10:17:58.643174",
  "processing_time_ms": 77.21
}
```

## Running the FastAPI Server

### Option 1: Using Uvicorn (Recommended)
```bash
# Install dependencies
pip install fastapi uvicorn[standard] pydantic

# Run the server
python -m uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Direct Python execution
```bash
python src/api/fastapi_app.py
```

## Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "season": 1,
       "yr": 0,
       "mnth": 1,
       "holiday": 0,
       "weekday": 1,
       "workingday": 1,
       "weathersit": 1,
       "temp": 0.25,
       "atemp": 0.3,
       "hum": 0.6,
       "windspeed": 0.2
     }'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "predictions": [
         {
           "season": 1, "yr": 0, "mnth": 1, "holiday": 0, "weekday": 1,
           "workingday": 1, "weathersit": 1, "temp": 0.25, "atemp": 0.3,
           "hum": 0.6, "windspeed": 0.2
         }
       ]
     }'
```

## API Documentation

- **Swagger UI**: Visit `http://localhost:8000/docs` for interactive API documentation
- **ReDoc**: Visit `http://localhost:8000/redoc` for alternative documentation
- **OpenAPI JSON**: Access `http://localhost:8000/openapi.json` for the API specification

## Key Differences from Flask API

| Feature | Flask API | FastAPI |
|---------|-----------|---------|
| Framework | Synchronous | Asynchronous |
| Documentation | Manual | Automatic (OpenAPI) |
| Validation | Manual | Pydantic models |
| Type Hints | Optional | Required |
| Performance | Good | Excellent (async) |
| Error Handling | Basic | Advanced with proper status codes |
| Request Models | JSON only | Typed Pydantic models |

## Configuration

The FastAPI app uses the same configuration as the Flask API (`configs/model_config.yaml`) and loads the same trained model and transformers.

## Production Deployment

For production deployment, consider:

1. **Gunicorn with Uvicorn workers**:
   ```bash
   pip install gunicorn
   gunicorn src.api.fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Docker deployment** with proper environment variables
3. **Load balancer** for multiple instances
4. **API Gateway** for rate limiting and authentication

## Monitoring and Logging

- All requests are logged with timestamps
- Processing time is tracked for performance monitoring
- Health checks provide system status
- Error responses include detailed error information

## Comparison: Flask vs FastAPI

Both APIs provide the same functionality but FastAPI offers:

- **Better Performance**: Async support and optimization
- **Type Safety**: Pydantic validation prevents invalid requests
- **Auto Documentation**: No need to maintain separate docs
- **Modern Standards**: Follows OpenAPI 3.0 specification
- **Better IDE Support**: Type hints enable better autocomplete

Choose FastAPI for new projects requiring high performance and modern API standards, or stick with Flask for simpler deployments.
