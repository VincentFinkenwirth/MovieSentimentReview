import os
import gdown
import zipfile
import shutil

# Path to the directories and their respective Google Drive links.
# Drive Folder directory: https://drive.google.com/drive/u/1/folders/1ZPf9ePrvo3eeZ6ESrIR6EiHgBgsE4T2y
DIRECTORIES = {
    "data": "https://drive.google.com/file/d/145kgNPYIBsM0srA2ViqZYr_eaVzvIHPc/view?usp=drive_link",
    "mlruns": "https://drive.google.com/file/d/1QB43r4gBdpnWA8cNrnsur81q-_9iyvZ0/view?usp=drive_link"
}



def download_and_extract_zip(drive_url, save_path):
    """
    Downloads a ZIP file from Google Drive and extracts it to the specified path without nesting.

    :param drive_url: Public Google Drive URL of the ZIP file.
    :param save_path: Directory where the extracted content will be stored.
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract file ID from Google Drive URL
    file_id = drive_url.split('/d/')[1].split('/')[0] if "/d/" in drive_url else drive_url.split('id=')[1]

    # Define local path for the downloaded zip file
    zip_path = os.path.join(save_path, "temp_download.zip")

    # Check if the directory is already existing (skip re-download)
    if os.listdir(save_path):  # If there are files/folders inside, assume it's already extracted
        print(f" Skipping download. Directory already exists: {save_path}")
        return

    print(f" Downloading from {drive_url} to {zip_path}...")

    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)

        # Extract ZIP file to a temporary location to inspect its structure
        temp_extract_path = os.path.join(save_path, "temp_extract")
        os.makedirs(temp_extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)

        # Find the actual extracted content
        extracted_items = os.listdir(temp_extract_path)

        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_path, extracted_items[0])):
            # If ZIP contains a single folder, move its contents up one level
            extracted_main_folder = os.path.join(temp_extract_path, extracted_items[0])
            for item in os.listdir(extracted_main_folder):
                shutil.move(os.path.join(extracted_main_folder, item), save_path)
        else:
            # If ZIP contains multiple items, move them directly
            for item in extracted_items:
                shutil.move(os.path.join(temp_extract_path, item), save_path)

        # Cleanup
        shutil.rmtree(temp_extract_path)  # Remove temp extraction folder
        os.remove(zip_path)  # Remove ZIP file after extraction

        print(f" Extraction complete. Files saved to: {save_path}")

    except Exception as e:
        print(f" Error downloading or extracting {drive_url}: {e}")

if __name__ == "__main__":
    # Loop through the directories and download/extract the files.
    for name, link in DIRECTORIES.items():
        download_and_extract_zip(link, name)
