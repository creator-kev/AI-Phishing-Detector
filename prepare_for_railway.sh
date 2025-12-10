#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║  Preparing for Railway Deployment          ║"
echo "║  Domain: ai-phishing-detector.com          ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Step 1: Create Procfile
echo -e "${BLUE}📝 Creating Procfile...${NC}"
cat > Procfile << 'EOF'
web: gunicorn app.main:app --workers 4 --bind 0.0.0.0:$PORT --timeout 120
EOF
echo -e "${GREEN}✅ Procfile created${NC}"

# Step 2: Update requirements.txt
echo -e "${BLUE}📦 Updating requirements.txt...${NC}"
if ! grep -q "Flask-Login" requirements.txt; then
    echo "Flask-Login>=0.6.2" >> requirements.txt
fi
if ! grep -q "werkzeug" requirements.txt; then
    echo "werkzeug>=2.3.0" >> requirements.txt
fi
echo -e "${GREEN}✅ requirements.txt updated${NC}"

# Step 3: Create .railwayignore
echo -e "${BLUE}📝 Creating .railwayignore...${NC}"
cat > .railwayignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.env
.venv
*.log
*.pot
*.pyc
.git/
.gitignore
*.md
tests/
.DS_Store
*.sqlite
*.db
.idea/
.vscode/
EOF
echo -e "${GREEN}✅ .railwayignore created${NC}"

# Step 4: Create runtime.txt (optional)
echo -e "${BLUE}📝 Creating runtime.txt...${NC}"
echo "python-3.11.0" > runtime.txt
echo -e "${GREEN}✅ runtime.txt created${NC}"

# Step 5: Verify railway.json exists
if [ -f "railway.json" ]; then
    echo -e "${GREEN}✅ railway.json exists${NC}"
else
    echo -e "${YELLOW}⚠️  Creating railway.json...${NC}"
    cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app.main:app --workers 4 --bind 0.0.0.0:$PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
fi

# Step 6: Check if model exists
if [ -f "models/best_model.pkl" ]; then
    echo -e "${GREEN}✅ Model file found${NC}"
else
    echo -e "${YELLOW}⚠️  Model file not found. You may need to train the model first.${NC}"
    echo "   Run: python run_pipeline.py"
fi

# Step 7: Create .env.example
echo -e "${BLUE}📝 Creating .env.example...${NC}"
cat > .env.example << 'EOF'
# Generate a secret key: python3 -c "import secrets; print(secrets.token_hex(32))"
FLASK_SECRET_KEY=your-super-secret-key-here
FLASK_ENV=production
DEBUG=False
DATABASE_PATH=/data/app.db
EOF
echo -e "${GREEN}✅ .env.example created${NC}"

# Step 8: Generate a secret key
echo ""
echo -e "${BLUE}🔑 Generated Secret Key:${NC}"
python3 -c "import secrets; print(secrets.token_hex(32))"
echo ""
echo -e "${YELLOW}📋 Save this key! You'll need it for Railway environment variables.${NC}"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║           Setup Complete! ✅               ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. Commit these changes:"
echo "   ${BLUE}git add .${NC}"
echo "   ${BLUE}git commit -m 'Prepare for Railway deployment'${NC}"
echo "   ${BLUE}git push origin main${NC}"
echo ""
echo "2. Go to Railway: https://railway.app"
echo "3. Deploy from GitHub repo"
echo "4. Add environment variables (use the secret key above)"
echo "5. Add custom domain: ai-phishing-detector.com"
echo "6. Configure Cloudflare DNS (see deployment guide)"
echo ""
echo -e "${YELLOW}📚 Full guide: See the Deployment Guide artifact${NC}"
