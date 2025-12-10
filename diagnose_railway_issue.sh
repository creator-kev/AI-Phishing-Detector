#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║   Railway Deployment Diagnostics           ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ISSUES_FOUND=0

# Check 1: Model file exists
echo -e "${BLUE}[1/8] Checking for trained model...${NC}"
if [ -f "models/best_model.pkl" ]; then
    echo -e "${GREEN}✅ Model file found${NC}"
else
    echo -e "${RED}❌ Model file missing${NC}"
    echo -e "${YELLOW}   Fix: Run 'python run_pipeline.py' to train model${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi
echo ""

# Check 2: Dockerfile exists
echo -e "${BLUE}[2/8] Checking for Dockerfile...${NC}"
if [ -f "Dockerfile" ]; then
    echo -e "${GREEN}✅ Dockerfile found${NC}"
    
    # Check if PORT is set
    if grep -q "PORT" Dockerfile; then
        echo -e "${GREEN}   ✅ PORT variable configured${NC}"
    else
        echo -e "${YELLOW}   ⚠️  PORT variable not set in Dockerfile${NC}"
    fi
else
    echo -e "${RED}❌ Dockerfile missing${NC}"
    echo -e "${YELLOW}   Creating Dockerfile...${NC}"
    
cat > Dockerfile << 'DOCKERFILE_CONTENT'
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p logs models data

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD gunicorn app.main:app \
    --workers 2 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
DOCKERFILE_CONTENT
    
    echo -e "${GREEN}   ✅ Dockerfile created${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi
echo ""

# Check 3: Gunicorn in requirements
echo -e "${BLUE}[3/8] Checking requirements.txt...${NC}"
if grep -q "gunicorn" requirements.txt; then
    echo -e "${GREEN}✅ gunicorn found in requirements.txt${NC}"
else
    echo -e "${RED}❌ gunicorn missing from requirements.txt${NC}"
    echo "gunicorn>=21.2.0" >> requirements.txt
    echo -e "${GREEN}   ✅ Added gunicorn to requirements.txt${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi

if grep -q "Flask-Login" requirements.txt; then
    echo -e "${GREEN}✅ Flask-Login found${NC}"
else
    echo -e "${YELLOW}⚠️  Flask-Login missing${NC}"
    echo "Flask-Login>=0.6.2" >> requirements.txt
    echo -e "${GREEN}   ✅ Added Flask-Login${NC}"
fi

if grep -q "werkzeug" requirements.txt; then
    echo -e "${GREEN}✅ werkzeug found${NC}"
else
    echo -e "${YELLOW}⚠️  werkzeug missing${NC}"
    echo "werkzeug>=2.3.0" >> requirements.txt
    echo -e "${GREEN}   ✅ Added werkzeug${NC}"
fi
echo ""

# Check 4: ProxyFix import
echo -e "${BLUE}[4/8] Checking ProxyFix import in main.py...${NC}"
if grep -q "from werkzeug.middleware.proxy_fix import ProxyFix" app/main.py; then
    echo -e "${GREEN}✅ ProxyFix import found${NC}"
else
    echo -e "${RED}❌ ProxyFix import missing${NC}"
    echo -e "${YELLOW}   This will cause deployment to fail!${NC}"
    echo -e "${YELLOW}   Add this line after Flask imports:${NC}"
    echo -e "   ${BLUE}from werkzeug.middleware.proxy_fix import ProxyFix${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi
echo ""

# Check 5: ProxyFix usage
echo -e "${BLUE}[5/8] Checking ProxyFix configuration...${NC}"
if grep -q "ProxyFix(app.wsgi_app" app/main.py; then
    echo -e "${GREEN}✅ ProxyFix configured${NC}"
else
    echo -e "${YELLOW}⚠️  ProxyFix not configured (optional but recommended)${NC}"
fi
echo ""

# Check 6: railway.json
echo -e "${BLUE}[6/8] Checking railway.json...${NC}"
if [ -f "railway.json" ]; then
    echo -e "${GREEN}✅ railway.json found${NC}"
    
    # Check healthcheck timeout
    if grep -q "healthcheckTimeout" railway.json; then
        TIMEOUT=$(grep "healthcheckTimeout" railway.json | grep -o '[0-9]*')
        if [ "$TIMEOUT" -ge 200 ]; then
            echo -e "${GREEN}   ✅ Healthcheck timeout: ${TIMEOUT}s (good)${NC}"
        else
            echo -e "${YELLOW}   ⚠️  Healthcheck timeout: ${TIMEOUT}s (may be too short)${NC}"
            echo -e "${YELLOW}   Recommended: 300s or higher${NC}"
        fi
    fi
else
    echo -e "${RED}❌ railway.json missing${NC}"
    echo -e "${YELLOW}   Creating railway.json...${NC}"
    
cat > railway.json << 'RAILWAY_CONTENT'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "gunicorn app.main:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
RAILWAY_CONTENT
    
    echo -e "${GREEN}   ✅ railway.json created${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi
echo ""

# Check 7: Database directory
echo -e "${BLUE}[7/8] Checking data directory...${NC}"
if [ -d "data" ]; then
    echo -e "${GREEN}✅ data directory exists${NC}"
else
    echo -e "${YELLOW}⚠️  data directory missing (will be created at runtime)${NC}"
    mkdir -p data
    echo -e "${GREEN}   ✅ Created data directory${NC}"
fi
echo ""

# Check 8: Git status
echo -e "${BLUE}[8/8] Checking git status...${NC}"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Git repository initialized${NC}"
    
    # Check if there are uncommitted changes
    if [[ -n $(git status -s) ]]; then
        echo -e "${YELLOW}⚠️  You have uncommitted changes${NC}"
        echo -e "${YELLOW}   Changes need to be committed and pushed to Railway${NC}"
    else
        echo -e "${GREEN}✅ All changes committed${NC}"
    fi
else
    echo -e "${RED}❌ Not a git repository${NC}"
    echo -e "${YELLOW}   Initialize with: git init${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND+1))
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════╗"
echo "║             Diagnostic Summary             ║"
echo "╚════════════════════════════════════════════╝"
echo ""

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ No critical issues found!${NC}"
    echo ""
    echo -e "${BLUE}Your project looks ready for deployment.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Commit any changes: git add . && git commit -m 'Fix deployment'"
    echo "2. Push to GitHub: git push origin main"
    echo "3. Railway will auto-deploy"
    echo "4. Check Railway logs if deployment fails"
else
    echo -e "${YELLOW}⚠️  Found $ISSUES_FOUND issue(s) that need attention${NC}"
    echo ""
    echo "Please review the issues above and fix them."
    echo ""
    echo "After fixing, run:"
    echo "  git add ."
    echo "  git commit -m 'Fix deployment issues'"
    echo "  git push origin main"
fi
echo ""

# Additional Railway environment variables check
echo -e "${BLUE}Required Railway Environment Variables:${NC}"
echo ""
echo "Make sure these are set in Railway Dashboard > Variables:"
echo ""
echo "  FLASK_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo 'RUN-THIS-COMMAND-TO-GENERATE')"
echo "  FLASK_ENV=production"
echo "  DEBUG=False"
echo "  DATABASE_PATH=/data/app.db"
echo "  PORT=8080"
echo ""

# Test local startup
echo -e "${BLUE}Test local deployment (optional):${NC}"
echo ""
echo "  export PORT=8080"
echo "  gunicorn app.main:app --bind 0.0.0.0:8080 --timeout 120"
echo "  curl http://localhost:8080/health"
echo ""

echo "╔════════════════════════════════════════════╗"
echo "║          Diagnostics Complete              ║"
echo "╚════════════════════════════════════════════╝"
