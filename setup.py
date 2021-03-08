from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# Create data directory
if not os.path.isdir('data'):
    os.mkdir('data')

gdd.download_file_from_google_drive(file_id='1RA9n7Fj8_X1puciHhaWvf0-3cAlW1bW0',
                                    dest_path='data/font_characters.zip',
                                    unzip=True)

os.remove('data/font_characters.zip')
