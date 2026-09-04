import re
import html
import requests
from pathlib import Path


CONTACT = "safwan.aljbaae@gmail.com"

LINEA_BASE_URL = "https://datasets.linea.org.br/gravdyn/Data"


def _parse_autoindex_links(response_text):
    """
    Extract file names from an nginx autoindex directory listing.

    Links ending with '/' are directories (or '../') and are skipped.
    """

    hrefs = re.findall(r'<a\s+href="([^"]+)"', response_text)
    return [html.unescape(h) for h in hrefs if not h.endswith("/")]


def list_linea_folder_files(path):
    """
    List files in a folder on the LInEA public dataset server
    (https://datasets.linea.org.br/gravdyn/Data/).
    """

    contact_msg = f"Please contact the package owner: {CONTACT}"

    folder = str(path).strip("/")
    url = f"{LINEA_BASE_URL}/{folder}/"

    try:
        response = requests.get(url, timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Network error while accessing the LInEA server: {e}\n{contact_msg}"
        ) from e

    if response.status_code == 404:
        raise FileNotFoundError(
            f"Folder not found on the LInEA server\n"
            f"URL: '{url}'\n"
            f"{contact_msg}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"LInEA server error {response.status_code}\n"
            f"URL: '{url}'\n"
            f"Details: {response.text}\n"
            f"{contact_msg}"
        )

    files = _parse_autoindex_links(response.text)

    if not files:
        raise FileNotFoundError(
            f"No files found in folder '{folder}'\n"
            f"URL: '{url}'\n"
            f"{contact_msg}"
        )

    return files


def download_linea_folder_files(path, output_dir="data"):
    """
    Download all files from a folder on the LInEA public dataset server
    (https://datasets.linea.org.br/gravdyn/Data/) into a local directory.
    """

    contact_msg = f"Please contact the package owner: {CONTACT}"

    folder = str(path).strip("/")
    base_url = f"{LINEA_BASE_URL}/{folder}"

    files = list_linea_folder_files(folder)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    for file_name in files:
        file_url = f"{base_url}/{file_name}"
        file_path = output_dir / file_name

        try:
            file_response = requests.get(file_url, timeout=60)
            file_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to download '{file_name}': {e}\n{contact_msg}"
            ) from e

        with open(file_path, "wb") as f:
            f.write(file_response.content)

        downloaded_files.append(file_path)

        print(f"Downloaded: {file_name}")

    if not downloaded_files:
        raise FileNotFoundError(
            f"No files found in folder '{folder}'\n{contact_msg}"
        )

    return downloaded_files


if __name__ == "__main__":
    files = list_linea_folder_files(
        path="Apophis/Pot_Expansion"
    )

    print(files)

    files = download_linea_folder_files(
        path="Apophis/Pot_Expansion",
        output_dir="./Data/Apophis/Pot_Expansion"
    )
