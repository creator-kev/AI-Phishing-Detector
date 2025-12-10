#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║   AI Phishing Detector - Production Deploy║"
echo "║   Domain: ai-phishing-detector.com         ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOMAIN="ai-phishing-detector.com"
APP_NAME="ai-phishing-detector"
PORT=5000

echo -e "${BLUE}🚀 Starting production deployment...${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  This script should be run as root or with sudo${NC}"
    echo "Run: sudo ./deploy_production.sh"
    exit 1
fi

# Update system
echo -e "${BLUE}📦 Updating system packages...${NC}"
apt-get update -qq
apt-get upgrade -y -qq

# Install required packages
echo -e "${BLUE}📦 Installing required packages...${NC}"
apt-get install -y -qq python3-pip python3-venv nginx certbot python3-certbot-nginx ufw

# Create application user
if ! id "$APP_NAME" &>/dev/null; then
    echo -e "${BLUE}👤 Creating application user...${NC}"
    useradd -m -s /bin/bash $APP_NAME
fi

# Setup application directory
APP_DIR="/opt/$APP_NAME"
echo -e "${BLUE}📁 Setting up application directory: $APP_DIR${NC}"

if [ ! -d "$APP_DIR" ]; then
    mkdir -p $APP_DIR
fi

# Copy application files
echo -e "${BLUE}📋 Copying application files...${NC}"
cp -r . $APP_DIR/
chown -R $APP_NAME:$APP_NAME $APP_DIR

# Setup Python virtual environment
cd $APP_DIR
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"
sudo -u $APP_NAME python3 -m venv venv
sudo -u $APP_NAME ./venv/bin/pip install --upgrade pip -q
sudo -u $APP_NAME ./venv/bin/pip install -r requirements.txt -q
sudo -u $APP_NAME ./venv/bin/pip install gunicorn -q

# Check/train model
if [ ! -f "$APP_DIR/models/best_model.pkl" ]; then
    echo -e "${YELLOW}⚠️  Model not found. Training model...${NC}"
    sudo -u $APP_NAME ./venv/bin/python run_pipeline.py
fi

# Create necessary directories
mkdir -p $APP_DIR/logs
mkdir -p $APP_DIR/data
chown -R $APP_NAME:$APP_NAME $APP_DIR/logs
chown -R $APP_NAME:$APP_NAME $APP_DIR/data

# Create systemd service
echo -e "${BLUE}⚙️  Creating systemd service...${NC}"
cat > /etc/systemd/system/$APP_NAME.service << EOF
[Unit]
Description=AI Phishing Detector Web Application
After=network.target

[Service]
Type=notify
User=$APP_NAME
Group=$APP_NAME
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn \\
    --workers 4 \\
    --worker-class sync \\
    --bind 127.0.0.1:$PORT \\
    --timeout 120 \\
    --access-logfile $APP_DIR/logs/access.log \\
    --error-logfile $APP_DIR/logs/error.log \\
    --log-level info \\
    app.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
echo -e "${BLUE}🌐 Configuring Nginx...${NC}"
cat > /etc/nginx/sites-available/$APP_NAME << EOF
# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN;
    
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}

# HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $DOMAIN www.$DOMAIN;
    
    # SSL certificates (will be configured by Certbot)
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Logs
    access_log /var/log/nginx/$APP_NAME.access.log;
    error_log /var/log/nginx/$APP_NAME.error.log;
    
    # Max upload size
    client_max_body_size 10M;
    
    # Proxy settings
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files
    location /static/ {
        alias $APP_DIR/app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Favicon
    location /favicon.ico {
        alias $APP_DIR/app/static/images/favicon.png;
        expires 30d;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
echo -e "${BLUE}🔍 Testing Nginx configuration...${NC}"
nginx -t

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Nginx configuration test failed${NC}"
    exit 1
fi

# Configure firewall
echo -e "${BLUE}🔥 Configuring firewall...${NC}"
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw reload

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
systemctl daemon-reload
systemctl enable $APP_NAME
systemctl restart $APP_NAME
systemctl restart nginx

# Wait for service to start
sleep 5

# Check service status
if systemctl is-active --quiet $APP_NAME; then
    echo -e "${GREEN}✅ Application service started successfully${NC}"
else
    echo -e "${RED}❌ Application service failed to start${NC}"
    echo "Check logs: journalctl -u $APP_NAME -n 50"
    exit 1
fi

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx started successfully${NC}"
else
    echo -e "${RED}❌ Nginx failed to start${NC}"
    exit 1
fi

# SSL Certificate setup
echo ""
echo -e "${BLUE}🔒 Setting up SSL certificate...${NC}"
echo -e "${YELLOW}⚠️  Make sure your DNS records are properly configured:${NC}"
echo "   A Record: $DOMAIN → Your Server IP"
echo "   A Record: www.$DOMAIN → Your Server IP"
echo ""
read -p "Press Enter when DNS is configured and ready to continue..."

certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --redirect --email admin@$DOMAIN

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ SSL certificate installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  SSL certificate installation failed or skipped${NC}"
    echo "You can run it manually later: sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
fi

# Setup auto-renewal
echo -e "${BLUE}🔄 Setting up SSL auto-renewal...${NC}"
systemctl enable certbot.timer
systemctl start certbot.timer

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║       🎉 Deployment Successful! 🎉        ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}🌐 Your application is now live at:${NC}"
echo "   https://$DOMAIN"
echo "   https://www.$DOMAIN"
echo ""
echo -e "${BLUE}📊 Management Commands:${NC}"
echo "   View logs:        journalctl -u $APP_NAME -f"
echo "   Restart app:      systemctl restart $APP_NAME"
echo "   Stop app:         systemctl stop $APP_NAME"
echo "   Start app:        systemctl start $APP_NAME"
echo "   App status:       systemctl status $APP_NAME"
echo "   Nginx logs:       tail -f /var/log/nginx/$APP_NAME.access.log"
echo "   Nginx restart:    systemctl restart nginx"
echo ""
echo -e "${BLUE}🔒 SSL Certificate:${NC}"
echo "   Auto-renewal:     Enabled"
echo "   Test renewal:     certbot renew --dry-run"
echo "   Renew manually:   certbot renew"
echo ""
echo -e "${BLUE}🔥 Firewall Status:${NC}"
ufw status
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "   1. Verify your site: https://$DOMAIN"
echo "   2. Test the AI scanner: https://$DOMAIN/scan"
echo "   3. Check analytics: https://$DOMAIN/analytics"
echo "   4. Monitor logs for any issues"
echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
