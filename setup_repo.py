import os

root = "."

directories =[
    ".github/workflows",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/api",
    "tests"
]

# Define the files to be created (relative to the root)
files = [
    ".github/workflows/ci.yml",
    "data/.gitkeep",  # just a placeholder so git tracks it
    "notebooks/1.0-eda.ipynb",
    "src/__init__.py",
    "src/data_processing.py",
    "src/train.py",
    "src/predict.py",
    "src/api/main.py",
    "src/api/pydantic_models.py",
    "tests/test_data_processing.py",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    ".gitignore",
    "README.md"
]

def create_project_structure():
 # Create directories
    for dir_path in directories:
        full_path = os.path.join(root, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created directory: {full_path}")

    # Create files
    for file_path in files:
        full_path = os.path.join(root, file_path)
        with open(file_path, "w") as f:
            if file_path.endswith(".py"):
                f.write("# Auto-generated script placeholder\n")
            elif file_path.endswith(".ipynb"):
                f.write("{}")  # Empty JSON for notebooks
            elif file_path == ".gitignore":
                f.write("data/\n__pycache__/\n.env\n*.pyc\n")
            elif file_path == "README.md":
                f.write("# Credit Risk Probability Model\n\nProject README.")
        print(f"📄 Created file: {full_path}")

if __name__ == "__main__":
    create_project_structure()
