import os
import io

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def main():
    # Read environment variables injected by GitHub Actions
    file_id = os.environ["DRIVE_FILE_ID"]
    out_path = os.environ.get("DRIVE_OUT_PATH", "data/Data_Files.zip")
    creds_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    # Create credentials from the temporary JSON file created by the auth step
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=SCOPES
    )

    # Build Drive API client
    service = build("drive", "v3", credentials=creds)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Download the file
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(out_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download {int(status.progress() * 100)}%")

    print(f"Dataset downloaded to {out_path}")

if __name__ == "__main__":
    main()
