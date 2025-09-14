import subprocess
import sys
import os

def install_package(package):
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🚀 Installing EduGuardians+ dependencies...")
    
    # Read requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
    
    # Install each package
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    # Report results
    if failed_packages:
        print("\n❌ The following packages failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\nTry installing them manually with:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        print("\nNow generating the model...")
        return True

if __name__ == "__main__":
    success = main()
    if success:
        # Try to generate the model
        try:
            import generate_model
            generate_model.create_simple_model()
        except Exception as e:
            print(f"❌ Error generating model: {e}")
