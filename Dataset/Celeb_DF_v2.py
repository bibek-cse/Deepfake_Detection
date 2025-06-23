import kagglehub
import os

def download_celeb_df_v2():
    """
    Downloads the Celeb-DF v2 dataset from Kaggle using kagglehub.

    Returns:
        str or None: Path to the downloaded dataset files, or None if download fails.

    Notes:
        - The dataset is downloaded into a structured folder such as:
            ~/.kaggle/kagglehub/datasets/reubensuju/celeb-df-v2/
        - This folder may contain videos or extracted frames.
        - If your model expects 'real' and 'fake' subfolders, you may need to manually organize the files.
    """
    try:
        print("[INFO] Starting download of Celeb-DF v2 dataset...")
        path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
        print(f"[SUCCESS] Dataset downloaded to: {path}")
        print("[NOTE] If your model expects 'real' and 'fake' folders, please organize extracted frames accordingly inside the above path.")
        return path
    except Exception as e:
        print(f"[ERROR] An error occurred while downloading the dataset: {e}")
        print("[HINT] Make sure your Kaggle API credentials are correctly set in ~/.kaggle/kaggle.json.")
        return None

def main():
    """
    Entry point for the dataset download script.
    """
    print("=== Celeb-DF v2 Dataset Downloader ===")
    print("This script downloads the Celeb-DF v2 dataset from Kaggle using kagglehub.\n")

    dataset_path = download_celeb_df_v2()

    if dataset_path:
        print(f"\n[INFO] Dataset is ready at: {dataset_path}")
    else:
        print("\n[FAILURE] Dataset download failed. Please verify your Kaggle setup and internet connection.")

if __name__ == "__main__":
    main()
