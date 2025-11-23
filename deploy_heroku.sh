#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║      Deploy to Heroku                      ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI is not installed"
    echo "Install from: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

echo "✅ Heroku CLI found"

# Login to Heroku
echo ""
echo "🔐 Logging in to Heroku..."
heroku login

# Create app (if not exists)
echo ""
read -p "Enter your Heroku app name: " APP_NAME

if heroku apps:info --app $APP_NAME &> /dev/null; then
    echo "ℹ️  App $APP_NAME already exists"
else
    echo "📦 Creating Heroku app..."
    heroku create $APP_NAME
fi

# Set environment variables
echo ""
echo "⚙️  Setting environment variables..."
heroku config:set FLASK_ENV=production --app $APP_NAME
heroku config:set DEBUG=False --app $APP_NAME
heroku config:set SECRET_KEY=$(openssl rand -hex 32) --app $APP_NAME

# Add buildpack
echo ""
echo "🔧 Adding Python buildpack..."
heroku buildpacks:set heroku/python --app $APP_NAME

# Deploy
echo ""
echo "🚀 Deploying to Heroku..."
git add .
git commit -m "Deploy to Heroku" || true
git push heroku main

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║       Deployment Successful!               ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
    echo "🌐 Your app: https://$APP_NAME.herokuapp.com"
    echo ""
    echo "Commands:"
    echo "  View logs:   heroku logs --tail --app $APP_NAME"
    echo "  Open app:    heroku open --app $APP_NAME"
    echo "  Scale:       heroku ps:scale web=1 --app $APP_NAME"
else
    echo "❌ Deployment failed"
    exit 1
fi
