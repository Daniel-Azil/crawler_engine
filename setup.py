#!/usr/bin/env python3
"""
Setup script for Intelligent Web Extractor

This script helps users set up the Intelligent Web Extractor
with proper configuration and dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Install core dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install Playwright browsers
    if not run_command("playwright install chromium", "Installing Playwright browsers"):
        return False
    
    return True


def setup_environment():
    """Set up environment configuration"""
    print("\n⚙️ Setting up environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("📋 Creating .env file from template...")
            shutil.copy(env_example, env_file)
            print("✅ .env file created")
            print("⚠️  Please edit .env file with your API keys and settings")
        else:
            print("❌ env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "extractions",
        "cache", 
        "logs",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test import
        import sys
        sys.path.insert(0, str(Path("src")))
        
        from intelligent_web_extractor import AdaptiveContentExtractor
        print("✅ Intelligent Web Extractor import successful")
        
        # Test configuration
        from intelligent_web_extractor.models.config import ExtractorConfig
        config = ExtractorConfig()
        print("✅ Configuration system working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    # Test configuration loading
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except Exception as e:
        print(f"❌ Environment loading failed: {e}")
        return False
    
    return True


def show_next_steps():
    """Show next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - INTELLIGENT_EXTRACTOR_API_KEY")
    print("   - OPENAI_API_KEY (if using OpenAI)")
    print("   - ANTHROPIC_API_KEY (if using Anthropic)")
    
    print("\n2. Test the installation:")
    print("   python example_usage.py")
    
    print("\n3. Use the command line interface:")
    print("   intelligent-extractor extract https://example.com")
    
    print("\n4. Read the documentation:")
    print("   - README.md for basic usage")
    print("   - Check the src/ directory for advanced examples")
    
    print("\n🚀 Happy extracting!")


def main():
    """Main setup function"""
    print("🧠 Intelligent Web Extractor Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Basic tests failed")
        sys.exit(1)
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main() 