#!/bin/bash
# Deployment preparation script for Vercel

echo "🚀 Preparing TDS-TA for Vercel deployment..."

# Create deployment directory
mkdir -p deploy
cd deploy

# Copy essential files only
echo "📁 Copying essential files..."
cp ../api/vercel_main.py ./main.py
cp ../data_source/main.npz ./
cp ../requirements.txt ./
cp ../vercel.json ./

# Create data_source directory for the API to find the embeddings
mkdir -p data_source
cp ../data_source/main.npz ./data_source/

# Create a minimal .env template
cat > .env.example << 'EOF'
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
EOF

# Create README for deployment
cat > README.md << 'EOF'
# TDS-TA API - Vercel Deployment

This is a minimal deployment version of the TDS Teaching Assistant API.

## Setup

1. Deploy to Vercel
2. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/` - Main query endpoint

## Usage

```bash
curl -X POST "https://your-vercel-app.vercel.app/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is TDS course about?"}'
```
EOF

echo "✅ Deployment files ready in ./deploy directory"
echo "📊 Directory size:"
du -sh .
echo ""
echo "🔧 Next steps:"
echo "1. cd deploy"
echo "2. Initialize git: git init"
echo "3. Add files: git add ."
echo "4. Commit: git commit -m 'Initial deployment'"
echo "5. Push to GitHub and connect to Vercel"
echo "6. Set environment variables in Vercel dashboard"
