import kagglehub
import os

def download_celeb_df_v2():
    """
    Downloads the Celeb-DF v2 dataset from Kaggle using kagglehub.
    
    Returns:
        str: Path to the downloaded dataset files.
    """
    try:
        # Download the latest version of the dataset
        print("Starting download of Celeb-DF v2 dataset...")
        path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
        print("Download completed successfully.")
        return path
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        return None

def main():
    # Ensure the script is run as a standalone program
    print("Welcome to the Celeb-DF v2 dataset downloader.")
    print("This script will download the dataset from Kaggle.")
    
    # Download the dataset
    dataset_path = download_celeb_df_v2()
    
    if dataset_path:
        print("Path to dataset files:", dataset_path)
    else:
        print("Failed to download the dataset. Please check your internet connection or the dataset identifier.")

if __name__ == "__main__":
    main()
