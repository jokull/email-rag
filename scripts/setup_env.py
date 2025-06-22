#!/usr/bin/env python3
"""
Email RAG Environment Setup Script
Interactive script to create .env file with IMAP configuration and secrets
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Optional


def print_header():
    """Print setup header"""
    print("=" * 70)
    print("📧 EMAIL RAG SYSTEM - ENVIRONMENT SETUP")
    print("=" * 70)
    print("This script will help you create a .env file for your email RAG system.")
    print("You'll need your email account details and we'll generate secure secrets.")
    print()


def print_section(title: str):
    """Print section header"""
    print(f"\n🔧 {title}")
    print("-" * (len(title) + 4))


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_port(port: str) -> bool:
    """Validate port number"""
    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except ValueError:
        return False


def generate_secret() -> Optional[str]:
    """Generate secure secret using openssl"""
    try:
        result = subprocess.run(
            ['openssl', 'rand', '-base64', '32'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_imap_config():
    """Get IMAP configuration from user"""
    print_section("IMAP EMAIL CONFIGURATION")
    print("Enter your email account details for syncing:")
    print()
    
    # Email address
    while True:
        email = input("📧 Email address: ").strip()
        if not email:
            print("❌ Email address is required")
            continue
        if not validate_email(email):
            print("❌ Please enter a valid email address")
            continue
        break
    
    # Determine common IMAP settings based on email domain
    domain = email.split('@')[1].lower()
    
    # Common IMAP server mappings
    imap_defaults = {
        'gmail.com': ('imap.gmail.com', 993, True),
        'outlook.com': ('outlook.office365.com', 993, True),
        'hotmail.com': ('outlook.office365.com', 993, True),
        'live.com': ('outlook.office365.com', 993, True),
        'yahoo.com': ('imap.mail.yahoo.com', 993, True),
        'icloud.com': ('imap.mail.me.com', 993, True),
        'me.com': ('imap.mail.me.com', 993, True),
        'mac.com': ('imap.mail.me.com', 993, True),
    }
    
    # Suggest IMAP server
    default_host, default_port, default_tls = imap_defaults.get(domain, ('', 993, True))
    
    if default_host:
        print(f"✨ Detected {domain} - suggesting IMAP settings")
        use_defaults = input(f"Use {default_host}:{default_port} with TLS? [Y/n]: ").strip().lower()
        
        if use_defaults in ('', 'y', 'yes'):
            imap_host = default_host
            imap_port = default_port
            imap_tls = default_tls
        else:
            imap_host, imap_port, imap_tls = get_custom_imap_settings()
    else:
        print("🔧 Custom email provider - please enter IMAP settings:")
        imap_host, imap_port, imap_tls = get_custom_imap_settings()
    
    # Password/App Password
    print(f"\n🔑 Password for {email}:")
    if domain in ['gmail.com', 'outlook.com', 'hotmail.com', 'live.com']:
        print("💡 For Gmail/Outlook, use an App Password (not your regular password)")
        print("   Gmail: https://support.google.com/accounts/answer/185833")
        print("   Outlook: https://support.microsoft.com/en-us/account-billing/using-app-passwords-with-apps-that-don-t-support-two-step-verification-5896ed9b-4263-e681-128a-a6f2979a7944")
    
    while True:
        password = input("🔐 Password/App Password: ").strip()
        if not password:
            print("❌ Password is required")
            continue
        break
    
    return {
        'IMAP_USER': email,
        'IMAP_HOST': imap_host,
        'IMAP_PORT': str(imap_port),
        'IMAP_PASS': password,
        'IMAP_TLS': str(imap_tls).lower()
    }


def get_custom_imap_settings():
    """Get custom IMAP settings"""
    # IMAP Host
    while True:
        imap_host = input("🌐 IMAP Server (e.g., imap.example.com): ").strip()
        if not imap_host:
            print("❌ IMAP server is required")
            continue
        break
    
    # IMAP Port
    while True:
        port_input = input("🔌 IMAP Port [993]: ").strip()
        if not port_input:
            imap_port = 993
            break
        if not validate_port(port_input):
            print("❌ Please enter a valid port number (1-65535)")
            continue
        imap_port = int(port_input)
        break
    
    # TLS
    tls_input = input("🔒 Use TLS encryption? [Y/n]: ").strip().lower()
    imap_tls = tls_input not in ('n', 'no', 'false')
    
    return imap_host, imap_port, imap_tls


def get_zero_secret():
    """Get or generate Zero auth secret"""
    print_section("ZERO AUTHENTICATION SECRET")
    print("Zero provides real-time sync between your database and UI.")
    print("A secure secret is required for authentication.")
    print()
    
    # Try to generate secret
    generated_secret = generate_secret()
    
    if generated_secret:
        print("✨ Generated secure secret using openssl")
        use_generated = input("Use generated secret? [Y/n]: ").strip().lower()
        
        if use_generated in ('', 'y', 'yes'):
            return generated_secret
    else:
        print("⚠️  openssl not found - cannot generate secret automatically")
    
    # Manual secret entry
    print("🔧 Please provide a secure secret (32+ characters recommended)")
    print("   You can generate one with: openssl rand -base64 32")
    
    while True:
        secret = input("🔐 Zero Auth Secret: ").strip()
        if not secret:
            print("❌ Secret is required")
            continue
        if len(secret) < 16:
            print("⚠️  Secret is quite short - consider using a longer one for security")
            confirm = input("Use this secret anyway? [y/N]: ").strip().lower()
            if confirm not in ('y', 'yes'):
                continue
        break
    
    return secret


def get_optional_config():
    """Get optional configuration"""
    print_section("OPTIONAL CONFIGURATION")
    print("These settings have sensible defaults but can be customized:")
    print()
    
    config = {}
    
    # Ask if user wants to customize
    customize = input("Customize optional settings? [y/N]: ").strip().lower()
    
    if customize in ('y', 'yes'):
        print("\n📊 Email Processing Settings:")
        
        # Sync interval
        sync_input = input("Email sync interval [300s]: ").strip()
        if sync_input:
            config['SYNC_INTERVAL'] = sync_input
        
        # Log level
        log_input = input("Log level (debug/info/warn/error) [info]: ").strip().lower()
        if log_input and log_input in ['debug', 'info', 'warn', 'error']:
            config['LOG_LEVEL'] = log_input
        
        print("\n🤖 AI Processing Settings:")
        
        # Scoring batch size
        scoring_batch = input("Email scoring batch size [10]: ").strip()
        if scoring_batch and scoring_batch.isdigit():
            config['SCORING_BATCH_SIZE'] = scoring_batch
        
        # Processing batch size  
        processing_batch = input("Content processing batch size [5]: ").strip()
        if processing_batch and processing_batch.isdigit():
            config['PROCESSING_BATCH_SIZE'] = processing_batch
    
    return config


def create_env_file(config: dict):
    """Create .env file with configuration"""
    print_section("CREATING .ENV FILE")
    
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    
    # Check if .env already exists
    if env_path.exists():
        print(f"⚠️  {env_path} already exists")
        overwrite = input("Overwrite existing .env file? [y/N]: ").strip().lower()
        if overwrite not in ('y', 'yes'):
            print("❌ Setup cancelled")
            return False
        
        # Backup existing file
        backup_path = env_path.with_suffix('.env.backup')
        try:
            env_path.rename(backup_path)
            print(f"📁 Backed up existing .env to {backup_path}")
        except Exception as e:
            print(f"⚠️  Could not backup existing .env: {e}")
    
    # Create .env content
    env_content = [
        "# Email RAG System Configuration",
        "# Generated by setup script",
        "",
        "# REQUIRED: IMAP Configuration",
        f"IMAP_HOST={config['IMAP_HOST']}",
        f"IMAP_PORT={config['IMAP_PORT']}",
        f"IMAP_USER={config['IMAP_USER']}",
        f"IMAP_PASS={config['IMAP_PASS']}",
        f"IMAP_TLS={config['IMAP_TLS']}",
        "",
        "# REQUIRED: Zero Auth Secret",
        f"ZERO_AUTH_SECRET={config['ZERO_AUTH_SECRET']}",
        "",
    ]
    
    # Add optional configuration
    if any(key.startswith(('SYNC_', 'LOG_', 'SCORING_', 'PROCESSING_')) for key in config.keys()):
        env_content.extend([
            "# Optional: Service Configuration",
        ])
        
        for key, value in config.items():
            if key not in ['IMAP_HOST', 'IMAP_PORT', 'IMAP_USER', 'IMAP_PASS', 'IMAP_TLS', 'ZERO_AUTH_SECRET']:
                env_content.append(f"{key}={value}")
        
        env_content.append("")
    
    env_content.extend([
        "# Additional settings can be found in .env.example",
        "# Most services have sensible defaults and don't require configuration",
    ])
    
    # Write file
    try:
        with open(env_path, 'w') as f:
            f.write('\n'.join(env_content))
        
        print(f"✅ Created {env_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def print_next_steps():
    """Print next steps"""
    print_section("NEXT STEPS")
    print("✅ Environment configuration complete!")
    print()
    print("🚀 To start the Email RAG system:")
    print("   1. docker-compose up -d")
    print("   2. Wait for services to initialize (check logs with: docker-compose logs -f)")
    print("   3. Access UI at: http://localhost:3001")
    print()
    print("📊 To monitor processing:")
    print("   cd scripts && uv sync && pipeline-monitor --continuous")
    print()
    print("📖 For more information, see README.md")


def main():
    """Main setup function"""
    try:
        print_header()
        
        # Collect configuration
        config = {}
        
        # IMAP configuration
        imap_config = get_imap_config()
        config.update(imap_config)
        
        # Zero secret
        zero_secret = get_zero_secret()
        config['ZERO_AUTH_SECRET'] = zero_secret
        
        # Optional configuration
        optional_config = get_optional_config()
        config.update(optional_config)
        
        # Create .env file
        if create_env_file(config):
            print_next_steps()
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()