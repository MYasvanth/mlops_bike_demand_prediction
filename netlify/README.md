# Bike Demand Prediction - Netlify Deployment

This is a serverless deployment of the Bike Demand Prediction API on Netlify, featuring a modern web interface and serverless functions.

## 🚀 Features

- **Serverless API**: Deployed as Netlify Functions (AWS Lambda)
- **Modern Web UI**: Responsive interface built with Bootstrap
- **HTTPS Enabled**: Automatic SSL certificate
- **Fast Global CDN**: Distributed worldwide
- **CORS Support**: Cross-origin requests enabled
- **Real-time Predictions**: Instant bike demand forecasting

## 📁 Project Structure

```
netlify/
├── index.html              # Main web interface
├── netlify.toml           # Netlify configuration
├── package.json           # Node.js dependencies
├── README.md              # This file
└── netlify/
    └── functions/
        ├── api.py         # Serverless API function
        └── requirements.txt # Python dependencies
```

## 🛠️ Local Development

### Prerequisites

1. **Install Netlify CLI**:
   ```bash
   npm install -g netlify-cli
   ```

2. **Install dependencies**:
   ```bash
   cd netlify
   npm install
   ```

3. **Copy model files**:
   Make sure the `models/` directory from your project is accessible to the functions.

### Running Locally

```bash
# Start local development server
npm run dev
# or
netlify dev
```

The site will be available at `http://localhost:8888`

## 🚀 Deployment to Netlify

### Option 1: Git-based Deployment (Recommended)

1. **Push to Git repository**:
   ```bash
   git add .
   git commit -m "Add Netlify deployment"
   git push origin main
   ```

2. **Connect to Netlify**:
   - Go to [Netlify](https://netlify.com)
   - Click "New site from Git"
   - Connect your repository
   - Set build settings:
     - **Build command**: `echo "No build required"`
     - **Publish directory**: `.`
   - Click "Deploy site"

### Option 2: Manual Deployment

```bash
# Login to Netlify
netlify login

# Deploy to production
npm run deploy
# or
netlify deploy --prod
```

## 🔧 Configuration

### Environment Variables

Set these in your Netlify dashboard (Site settings > Environment variables):

- `PYTHON_VERSION`: `3.9` (or your preferred version)

### Custom Domain (Optional)

1. Go to Site settings > Domain management
2. Add your custom domain
3. Configure DNS records as instructed

## 📡 API Endpoints

Once deployed, your API will be available at:

- **Base URL**: `https://your-site-name.netlify.app`
- **Health Check**: `GET /.netlify/functions/api/health`
- **Prediction**: `POST /.netlify/functions/api/predict`
- **Model Info**: `GET /.netlify/functions/api/model/info`

### API Usage Examples

#### Health Check
```bash
curl https://your-site-name.netlify.app/.netlify/functions/api/health
```

#### Make Prediction
```bash
curl -X POST "https://your-site-name.netlify.app/.netlify/functions/api/predict" \
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

## 🎨 Web Interface

The web interface provides:

- **Interactive Form**: Easy input for all features
- **Real-time Validation**: Client-side form validation
- **Loading States**: Visual feedback during API calls
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: User-friendly error messages

## 🔒 Security Features

- **HTTPS Only**: Automatic SSL/TLS encryption
- **CORS Protection**: Configured for web access
- **Input Validation**: Server-side validation
- **Rate Limiting**: Built-in Netlify protection
- **Secure Headers**: Security headers configured

## 📊 Monitoring

### Netlify Analytics

- **Function Logs**: Monitor API usage and errors
- **Performance Metrics**: Response times and success rates
- **Traffic Analytics**: Visitor statistics

### Custom Monitoring

Add monitoring to your functions:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log in your function
logger.info(f"Prediction request: {body}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Function Timeout**:
   - Netlify functions have a 10-second timeout
   - Optimize model loading and prediction speed

2. **Memory Limits**:
   - Functions have memory limits
   - Consider model optimization or smaller models

3. **Cold Starts**:
   - First request may be slower due to cold start
   - Subsequent requests are faster

### Debug Locally

```bash
# Enable debug logging
export NETLIFY_DEV_LOG=true
netlify dev
```

## 🔄 CI/CD

### GitHub Actions Example

```yaml
name: Deploy to Netlify
on:
  push:
    branches: [ main ]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Netlify
        uses: netlify/actions/cli@master
        with:
          args: deploy --prod --dir=netlify
        env:
          NETLIFY_AUTH_TOKEN: ${{ secrets.NETLIFY_AUTH_TOKEN }}
          NETLIFY_SITE_ID: ${{ secrets.NETLIFY_SITE_ID }}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📞 Support

For issues and questions:

- Check the [Netlify Functions documentation](https://docs.netlify.com/functions/overview/)
- Review function logs in the Netlify dashboard
- Open an issue in this repository

---

**Happy deploying! 🚀**
