import os
import subprocess
import sys
import venv
from pathlib import Path
import requests
from functools import partial
import importlib.util

def create_kes_config_dir():
    
    root_dir = Path(__file__).parent.resolve()  
    modules_dir = root_dir / "modules"
    kes_config_dir = modules_dir / "kes_config"      
    kes_config_dir.mkdir(parents=True, exist_ok=True)

    print(f"{kes_config_dir} directory created")
    return kes_config_dir 
    
def download_and_install_file(url, target_dir, filename):
    # Define the target directory and ensure it exists
    target_path = Path(target_dir) / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the file from GitHub
    response = requests.get(url)
    response.raise_for_status()  # Ensure we notice bad responses

    # Write the file content to the target location
    with open(target_path, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded and installed {filename} to {target_path}")
    
def install_requirements(venv_path, requirements_github_url, requirements_target_dir, requirements_filename):
    """Install requirements into the specified virtual environment."""
    # Determine the path to pip in the virtual environment
    pip_executable = os.path.join(venv_path, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'pip')

    # Define the requirements file path
    requirements_file = requirements_target_dir / requirements_filename

    # Check if the requirements file exists; download if missing
    if not requirements_file.is_file():
        print(f"Requirements file '{requirements_filename}' not found in root. Attempting download from GitHub.")
        download_and_install_file(requirements_github_url, requirements_target_dir, requirements_filename)

        # Recheck if download succeeded
        if not requirements_file.is_file():
            print(f"Failed to download '{requirements_filename}' from GitHub. Exiting.")
            sys.exit(1)

    # Run pip install
    try:
        print(f"Installing requirements from {requirements_file} into virtual environment at {venv_path}...")
        subprocess.check_call([pip_executable, 'install', '-r', requirements_file])
        print("Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def create_virtualenv(venv_path):
    """Create a new virtual environment at the specified path."""
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    print(f"Virtual environment created at {venv_path}")
    
def setup_simple_kes(modules_dir, kes_config_dir):
    scheduler_path = modules_dir / "simple_karras_exponential_scheduler.py"

    # Dynamically load simple_karras_exponential_scheduler from the modules folder
    if scheduler_path.exists():
        spec = importlib.util.spec_from_file_location("simple_karras_exponential_scheduler", scheduler_path)
        scheduler_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scheduler_module)
        simple_karras_exponential_scheduler = scheduler_module.simple_karras_exponential_scheduler
    else:
        raise FileNotFoundError(f"{scheduler_path} does not exist.")
    
def main():
    # Define paths
    scheduler_github_url = "https://raw.githubusercontent.com/Kittensx/Simple_KES/refs/heads/main/modules/simple_karras_exponential_scheduler.py"
    scheduler_target_dir = Path(__file__).parent.resolve() / "modules"
    scheduler_filename = "simple_karras_exponential_scheduler.py"
    scheduler_target_path = scheduler_target_dir / scheduler_filename
    config_github_url = "https://raw.githubusercontent.com/Kittensx/Simple_KES/refs/heads/main/modules/simple_kes_scheduler.yaml"
    config_target_dir = scheduler_target_dir / "kes_config" 
    config_filename = "simple_kes_scheduler.yaml"   
    config_target_path = config_target_dir / config_filename
    requirements_github_url= "https://raw.githubusercontent.com/Kittensx/Simple_KES/refs/heads/main/simple_kes_requirements.txt"
    requirements_target_dir = scheduler_target_dir / "kes_config" 
    requirements_filename = "simple_kes_requirements.txt"
    requirements_target_path = requirements_target_dir / requirements_filename    
    modules_dir = scheduler_target_dir
    kes_config_dir = create_kes_config_dir()    
    setup_simple_kes(modules_dir, kes_config_dir)  
    root_path = os.path.abspath(os.getcwd())
    venv_path = os.path.join(root_path, 'venv')
    fallback_venv_path = os.path.join(root_path, 'simple_kes_requirements')
    
    if not scheduler_target_path.is_file():
        print(f"{scheduler_filename} not found locally. Downloading from GitHub...")
        download_and_install_file(scheduler_github_url, scheduler_target_dir, scheduler_filename)
    else:
        print(f"{scheduler_filename} already exists. Skipping download.")

    if not config_target_path.is_file():
        print(f"{config_filename} not found locally. Downloading from GitHub...")
        download_and_install_file(config_github_url, config_target_dir, config_filename)
    else:
        print(f"{config_filename} already exists. Skipping download.")
      
        
    if not requirements_target_path.is_file():
        print(f"{requirements_filename} not found locally. Downloading from GitHub...")
        download_and_install_file(requirements_github_url, requirements_target_dir, requirements_filename)
    else:
        print(f"{requirements_filename} already exists. Installing requirements.")
        install_requirements(venv_path, requirements_github_url, requirements_target_dir, requirements_filename) 

    # Virtual environment setup and requirements installation
    if os.path.isdir(venv_path):
        print(f"'venv' folder found at {venv_path}")
        install_requirements(venv_path, requirements_github_url, requirements_target_dir, requirements_filename)
    else:
        print(f"'venv' folder not found. Using fallback path '{fallback_venv_path}'.")
        if not os.path.isdir(fallback_venv_path):
            create_virtualenv(fallback_venv_path)
        install_requirements(fallback_venv_path, requirements_github_url, requirements_target_dir, requirements_filename)

if __name__ == "__main__":   
    main()
