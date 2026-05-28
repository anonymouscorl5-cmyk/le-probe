# --- Path Stabilization ---
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --------------------------

from huggingface_hub import HfApi
from pathlib import Path
import os


def upload_to_hf(folder_path, repo_id):
    api = HfApi()

    print(f"📦 Preparing to upload folder: {folder_path}")
    print(f"🚀 Target Repository: https://huggingface.co/datasets/{repo_id}")

    try:
        # Create the repository if it doesn't exist
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        # Use the specifically recommended method for large folders
        api.upload_large_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

        print("\n✅ Upload Complete! Your dataset is now live.")

    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        print(
            "\n💡 Make sure you have run 'huggingface-cli login' and have write access to the repo."
        )


if __name__ == "__main__":
    # Path is relative to the root of the workspace
    DATASET_DIR = "le-probe/datasets/anonymouscorl5-cmyk/gr1_reward_pred"
    REPO_ID = "gr1_reward_pred"

    if os.path.exists(DATASET_DIR):
        upload_to_hf(DATASET_DIR, REPO_ID)
    else:
        print(f"❌ Error: Folder {DATASET_DIR} not found.")
