from huggingface_hub import HfApi, Repository, login
import os
import csv

# Replace with your Hugging Face credentials
hf_token = "your_hugging_face_api_token"
repo_name = "your_username/your_dataset_name"
local_dir = "path_to_your_local_dataset_directory"
csv_file_path = "path_to_your_csv_file.csv"

# Log in to Hugging Face Hub
login(token=hf_token)

# Create the dataset repository
api = HfApi()
api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)

# Clone the repository to a local directory
repo = Repository(local_dir=local_dir, clone_from=repo_name)

# Prepare the README file
readme_content = f"""# {repo_name.split('/')[-1]}

## Description
A brief description of the dataset.

## Structure
- **data/class1/**: Images belonging to class 1.
- **data/class2/**: Images belonging to class 2.
- **labels.csv**: Contains class labels and image URLs.

## Usage
Instructions on how to use the dataset.
"""
with open(os.path.join(local_dir, "README.md"), "w") as readme_file:
    readme_file.write(readme_content)

# Prepare the dataset structure
# Ensure the CSV file paths match the local directory structure
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Assuming the first row is the header
    for row in reader:
        image_url, class_label = row
        # Perform any necessary operations on image_url and class_label
        # For example, downloading the image or verifying the path

# Add all files and push to the repository
repo.git_add(auto_lfs_track=True)  # auto_lfs_track=True for handling large files
repo.git_commit("Initial commit")
repo.git_push()

# Optionally, set the repository visibility to public
api.update_repo_visibility(repo_id=repo_name, private=False)

print(f"Dataset {repo_name} uploaded successfully.")
