#!/usr/bin/env python3
"""
Project Structure Diagnostic Tool
Checks if all required files and directories exist for the AI Phishing Detector
"""

import os
import sys
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for colored output
try:
    init(autoreset=True)
except:
    pass

def print_status(message, status='info'):
    """Print colored status message"""
    if status == 'success':
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    elif status == 'error':
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
    elif status == 'warning':
        print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}ℹ️  {message}{Style.RESET_ALL}")

def check_directory(path, name):
    """Check if directory exists"""
    if path.exists() and path.is_dir():
        print_status(f"{name}: {path}", 'success')
        return True
    else:
        print_status(f"{name}: {path} NOT FOUND", 'error')
        return False

def check_file(path, name):
    """Check if file exists"""
    if path.exists() and path.is_file():
        print_status(f"{name}: {path.name}", 'success')
        return True
    else:
        print_status(f"{name}: {path.name} NOT FOUND", 'error')
        return False

def main():
    print("\n" + "="*70)
    print("🔍 AI PHISHING DETECTOR - PROJECT STRUCTURE DIAGNOSTIC")
    print("="*70 + "\n")
    
    # Get project root
    root = Path(__file__).resolve().parent
    print_status(f"Project Root: {root}", 'info')
    print_status(f"Current Working Directory: {os.getcwd()}", 'info')
    print()
    
    issues = []
    
    # Check main directories
    print("📁 Checking Main Directories:")
    print("-" * 70)
    app_dir = root / 'app'
    models_dir = root / 'models'
    logs_dir = root / 'logs'
    data_dir = root / 'data'
    
    if not check_directory(app_dir, "app/"):
        issues.append("Create app/ directory")
    if not check_directory(models_dir, "models/"):
        issues.append("Create models/ directory")
    if not check_directory(logs_dir, "logs/"):
        issues.append("Create logs/ directory (will be auto-created)")
    if not check_directory(data_dir, "data/"):
        issues.append("Create data/ directory")
    
    print()
    
    # Check templates structure
    print("📄 Checking Templates Structure:")
    print("-" * 70)
    templates_dir = app_dir / 'templates'
    pages_dir = templates_dir / 'pages'
    
    if not check_directory(templates_dir, "app/templates/"):
        issues.append("Create app/templates/ directory")
    else:
        if not check_directory(pages_dir, "app/templates/pages/"):
            issues.append("Create app/templates/pages/ directory")
        else:
            # Check for HTML files
            print("\n📋 Template Files:")
            required_templates = [
                'home.html',
                'features.html',
                'documentation.html',
                'about.html',
                'contact.html',
                'pricing.html',
                'api_access.html',
                'extension.html',
                'privacy.html',
                'terms.html',
                'status.html',
                'scan.html'
            ]
            
            for template in required_templates:
                template_path = pages_dir / template
                if not check_file(template_path, f"  {template}"):
                    issues.append(f"Create app/templates/pages/{template}")
            
            # Check root templates
            print("\n📋 Root Template Files:")
            root_templates = ['batch_checker.html', 'dashboard.html']
            for template in root_templates:
                template_path = templates_dir / template
                if not check_file(template_path, f"  {template}"):
                    issues.append(f"Create app/templates/{template}")
    
    print()
    
    # Check Python files
    print("🐍 Checking Python Files:")
    print("-" * 70)
    main_py = app_dir / 'main.py'
    
    check_file(main_py, "main.py")
    
    # Check if render_template is imported
    if main_py.exists():
        with open(main_py, 'r') as f:
            content = f.read()
            if 'from flask import' in content and 'render_template' in content:
                print_status("  render_template imported", 'success')
            else:
                print_status("  render_template NOT imported", 'warning')
                issues.append("Add 'render_template' to Flask imports")
    
    print()
    
    # Check models
    print("🤖 Checking Models:")
    print("-" * 70)
    model_files = ['best_model.pkl', 'phishing_pipeline.pkl']
    model_found = False
    
    for model in model_files:
        model_path = models_dir / model
        if check_file(model_path, f"  {model}"):
            model_found = True
    
    if not model_found:
        issues.append("No model files found - run training pipeline")
    
    print()
    
    # Check static files (optional)
    print("🎨 Checking Static Files (Optional):")
    print("-" * 70)
    static_dir = app_dir / 'static'
    if check_directory(static_dir, "app/static/"):
        css_dir = static_dir / 'css'
        js_dir = static_dir / 'js'
        img_dir = static_dir / 'images'
        
        check_directory(css_dir, "  app/static/css/")
        check_directory(js_dir, "  app/static/js/")
        check_directory(img_dir, "  app/static/images/")
    
    print()
    
    # Check configuration files
    print("⚙️  Checking Configuration Files:")
    print("-" * 70)
    config_files = [
        'requirements.txt',
        'Dockerfile',
        '.env.example',
        'docker-compose.yml'
    ]
    
    for config_file in config_files:
        config_path = root / config_file
        check_file(config_path, f"  {config_file}")
    
    print()
    
    # Summary
    print("="*70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if issues:
        print(f"\n{Fore.RED}❌ Found {len(issues)} issue(s):{Style.RESET_ALL}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n{Fore.YELLOW}💡 Suggested Actions:{Style.RESET_ALL}")
        print("  1. Create missing directories:")
        print("     mkdir -p app/templates/pages")
        print("     mkdir -p app/static/css app/static/js app/static/images")
        print("  2. Create missing template files or run setup script")
        print("  3. Ensure you're running the app from project root:")
        print("     python app/main.py")
        print()
    else:
        print(f"\n{Fore.GREEN}✅ All checks passed! Your project structure looks good.{Style.RESET_ALL}\n")
    
    # Flask app test
    print("="*70)
    print("🧪 FLASK APP TEST")
    print("="*70 + "\n")
    
    try:
        sys.path.insert(0, str(root))
        from app.main import app as flask_app
        
        print_status("Flask app imported successfully", 'success')
        print_status(f"Template folder: {flask_app.template_folder}", 'info')
        print_status(f"Static folder: {flask_app.static_folder}", 'info')
        
        # List registered routes
        print("\n📍 Registered Routes:")
        routes = []
        for rule in flask_app.url_map.iter_rules():
            if rule.endpoint != 'static':
                routes.append(f"  {rule.rule} -> {rule.endpoint}")
        
        for route in sorted(routes)[:20]:  # Show first 20 routes
            print(route)
        
        if len(routes) > 20:
            print(f"  ... and {len(routes) - 20} more routes")
        
    except Exception as e:
        print_status(f"Could not import Flask app: {e}", 'error')
        issues.append("Fix Flask app import errors")
    
    print("\n" + "="*70)
    print("🏁 DIAGNOSIS COMPLETE")
    print("="*70 + "\n")
    
    if issues:
        print(f"{Fore.YELLOW}Please fix the issues above and run this script again.{Style.RESET_ALL}")
        return 1
    else:
        print(f"{Fore.GREEN}Your project is ready to run!{Style.RESET_ALL}")
        print(f"\nStart the app with: {Fore.CYAN}python app/main.py{Style.RESET_ALL}")
        return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
