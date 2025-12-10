#!/bin/bash
#
# AI Phishing Detector - Quick Launch Script
# Domain: ai-phishing-detector.com
#
# Usage: curl -fsSL https://raw.githubusercontent.com/YOUR_REPO/main/quick_launch.sh | sudo bash
#

set -e

# Configuration
DOMAIN="ai-phishing-detector.com"
APP_NAME="ai-phishing-detector"
REPO_URL="https://github.com/creator-kev/ai-phishing-detector.git"  # Update this

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ASCII Art Banner
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🛡️  AI Phishing Detector - Quick Launch                  ║
║                                                               ║
║     Domain: ai-phishing-detector.com                          ║
║     Deploying to production...                                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF

echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Error: This script must be run as root${NC}"
    echo "Usage: sudo ./quick_launch.sh"
    exit 1
fi

# Get server IP
SERVER_IP=$(curl -s ifconfig.me)
echo -e "${BLUE}📍 Server IP detected: ${GREEN}$SERVER_IP${NC}"
echo ""

# Prompt for DNS confirmation
echo -e "${YELLOW}⚠️  DNS Configuration Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Before continuing, ensure your Cloudflare DNS is configured:"
echo ""
echo "  1. A Record: $DOMAIN → $SERVER_IP"
echo "  2. A Record: www.$DOMAIN → $SERVER_IP"
echo "  3. Proxy Status: Enabled (Orange Cloud)"
echo "  4. SSL/TLS Mode: Full (strict)"
echo ""
read -p "Have you configured DNS records? (yes/no): " dns_ready

if [[ ! "$dns_ready" =~ ^[Yy][Ee][Ss]$ ]]; then
    echo ""
    echo -e "${YELLOW}Please configure DNS first, then run this script again.${NC}"
    echo ""
    echo "Quick Setup:"
    echo "  1. Go to Cloudflare Dashboard"
    echo "  2. Select ai-phishing-detector.com"
    echo "  3. Click 'DNS' in left sidebar"
    echo "  4. Add the A records shown above"
    echo "  5. Wait 2-5 minutes for propagation"
    echo ""
    exit 0
fi

echo ""
echo -e "${GREEN}✅ Proceeding with deployment...${NC}"
echo ""
sleep 2

# Update system
echo -e "${BLUE}[1/8]${NC} Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq
echo -e "${GREEN}✅ System updated${NC}"
echo ""

# Install dependencies
echo -e "${BLUE}[2/8]${NC} Installing dependencies..."
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw \
    git \
    curl \
    htop
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Clone repository
echo -e "${BLUE}[3/8]${NC} Cloning application repository..."
APP_DIR="/opt/$APP_NAME"
if [ -d "$APP_DIR" ]; then
    echo "Directory exists, updating..."
    cd "$APP_DIR"
    git pull
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"
echo -e "${GREEN}✅ Repository cloned${NC}"
echo ""

# Setup Python environment
echo -e "${BLUE}[4/8]${NC} Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install gunicorn -q
echo -e "${GREEN}✅ Python environment ready${NC}"
echo ""

# Train model if needed
echo -e "${BLUE}[5/8]${NC} Checking ML model..."
if [ ! -f "models/best_model.pkl" ]; then
    echo "Training model (this may take 2-5 minutes)..."
    python run_pipeline.py
fi
echo -e "${GREEN}✅ Model ready${NC}"
echo ""

# Create systemd service
echo -e "${BLUE}[6/8]${NC} Configuring system service..."
cat > /etc/systemd/system/$APP_NAME.service << EOF
[Unit]
Description=AI Phishing Detector Web Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn \\
    --workers 4 \\
    --bind 127.0.0.1:5000 \\
    --timeout 120 \\
    --access-logfile $APP_DIR/logs/access.log \\
    --error-logfile $APP_DIR/logs/error.log \\
    app.main:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
chown -R www-data:www-data "$APP_DIR"
mkdir -p "$APP_DIR/logs" "$APP_DIR/data"
chown -R www-data:www-data "$APP_DIR/logs" "$APP_DIR/data"

systemctl daemon-reload
systemctl enable $APP_NAME
echo -e "${GREEN}✅ Service configured${NC}"
echo ""

# Configure Nginx
echo -e "${BLUE}[7/8]${NC} Configuring web server..."
cat > /etc/nginx/sites-available/$APP_NAME << 'EOF'
server {
    listen 80;
    server_name ai-phishing-detector.com www.ai-phishing-detector.com;
    
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
echo -e "${GREEN}✅ Nginx configured${NC}"
echo ""

# Configure firewall
echo -e "${BLUE}[8/8]${NC} Configuring firewall..."
ufw --force enable
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw reload
echo -e "${GREEN}✅ Firewall configured${NC}"
echo ""

# Start services
echo -e "${PURPLE}🚀 Starting services...${NC}"
systemctl restart $APP_NAME
systemctl restart nginx
sleep 3

# Check service status
if systemctl is-active --quiet $APP_NAME; then
    echo -e "${GREEN}✅ Application started${NC}"
else
    echo -e "${RED}❌ Application failed to start${NC}"
    journalctl -u $APP_NAME -n 20
    exit 1
fi

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx started${NC}"
else
    echo -e "${RED}❌ Nginx failed to start${NC}"
    exit 1
fi

echo ""

# SSL Setup
echo -e "${PURPLE}🔒 Setting up SSL certificate...${NC}"
echo ""
echo "This will:"
echo "  - Obtain free SSL certificate from Let's Encrypt"
echo "  - Configure automatic HTTPS redirect"
echo "  - Setup auto-renewal"
echo ""
read -p "Proceed with SSL setup? (yes/no): " ssl_ready

if [[ "$ssl_ready" =~ ^[Yy][Ee][Ss]$ ]]; then
    certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --redirect --email admin@$DOMAIN
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ SSL certificate installed${NC}"
        systemctl enable certbot.timer
    else
        echo -e "${YELLOW}⚠️  SSL setup failed. You can run it manually later:${NC}"
        echo "sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
    fi
else
    echo -e "${YELLOW}⚠️  SSL setup skipped${NC}"
    echo "Run manually: sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              🎉 DEPLOYMENT SUCCESSFUL! 🎉                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF

echo ""
echo -e "${GREEN}🌐 Your application is live at:${NC}"
echo "   https://ai-phishing-detector.com"
echo "   https://www.ai-phishing-detector.com"
echo ""
echo -e "${BLUE}📊 Quick Status Check:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test local endpoint
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    echo -e "   Local API:     ${GREEN}✅ Healthy${NC}"
else
    echo -e "   Local API:     ${YELLOW}⚠️  Check logs${NC}"
fi

# Show service status
if systemctl is-active --quiet $APP_NAME; then
    echo -e "   App Service:   ${GREEN}✅ Running${NC}"
else
    echo -e "   App Service:   ${RED}❌ Stopped${NC}"
fi

if systemctl is-active --quiet nginx; then
    echo -e "   Web Server:    ${GREEN}✅ Running${NC}"
else
    echo -e "   Web Server:    ${RED}❌ Stopped${NC}"
fi

echo ""
echo -e "${BLUE}📝 Management Commands:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   View logs:        journalctl -u $APP_NAME -f"
echo "   Restart app:      systemctl restart $APP_NAME"
echo "   Check status:     systemctl status $APP_NAME"
echo "   Restart nginx:    systemctl restart nginx"
echo "   SSL renewal:      certbot renew"
echo ""
echo -e "${BLUE}📂 Important Paths:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Application:      $APP_DIR"
echo "   Logs:            $APP_DIR/logs/"
echo "   Database:        $APP_DIR/data/app.db"
echo "   Nginx config:    /etc/nginx/sites-available/$APP_NAME"
echo ""
echo -e "${YELLOW}🔍 Next Steps:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   1. Visit: https://ai-phishing-detector.com"
echo "   2. Create your admin account"
echo "   3. Test the URL scanner"
echo "   4. Review analytics dashboard"
echo "   5. Setup monitoring (UptimeRobot, etc.)"
echo ""
echo -e "${GREEN}✨ Deployment completed in $(date)${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
