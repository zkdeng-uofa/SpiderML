import csv
import os
import argparse
import time
import urllib.request
import io
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args() -> argparse.Namespace:
    """
    Parse user inputs from arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to input CSV file")
    parser.add_argument("--output_folder", type=str, help="Path to output folder for downloaded images")
    parser.add_argument("--url_column", type=str, required=True, help="Column name for photo URLs")
    parser.add_argument("--name_column", type=str, required=True, help="Column name for names")
    return parser.parse_args()

def download_image(image_data, output):
    """Download an image with urllib"""
    key, image_url, class_name = image_data
    class_name = class_name.replace("'", "").replace(" ", "_")
    file_name = f"{image_url.split('/')[-2]}.jpg"
    file_path = os.path.join(output, class_name, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    timeout = 10
    max_retries = 1
    attempts = 0
    while attempts < max_retries:
        try:
            request = urllib.request.Request(image_url, data=None, headers={"User-Agent": user_agent_string})
            with urllib.request.urlopen(request, timeout=timeout) as r:
                img_stream = io.BytesIO(r.read())
                with open(file_path, 'wb') as f:
                    f.write(img_stream.getbuffer())
                return key, file_name, class_name, None
        except Exception as err:
            attempts += 1
            time.sleep(2**attempts)  # Exponential backoff
            error_message = str(err)
    return key, None, class_name, error_message

def main():
    inputs = parse_args()
    # Read the CSV file
    with open(inputs.input_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        image_data_list = [(i, row[inputs.url_column], row[inputs.name_column]) for i, row in enumerate(reader)]
    
    errors = 0
    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=512) as executor:
        # Create a list of futures
        futures = [executor.submit(download_image, image_data, inputs.output_folder) for image_data in image_data_list]

        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            key, file_name, class_name, error = future.result()
            if error:
                errors += 1
    print(f"Completed with {errors} errors.")

if __name__ == '__main__':
    main()
