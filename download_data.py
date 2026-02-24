import os
import requests
import zipfile

def download_file(url, filename):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        wrote = 0 
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def extract_zip(filename, extract_to):
    print(f"Extracting {filename} to {extract_to}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {filename}")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Download full dataset
    url = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip"
    filename = "data/full.zip"
    
    if not os.path.exists(filename):
        download_file(url, filename)
    
    extract_to = "data/mafaulda"
    if not os.path.exists(extract_to):
        extract_zip(filename, extract_to)
