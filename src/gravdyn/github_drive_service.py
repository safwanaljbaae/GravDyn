import os
import requests
from pathlib import Path
from dotenv import load_dotenv


CONTACT = "safwan.aljbaae@gmail.com"

load_dotenv()  # loads .env file automatically
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

def _get_headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def list_github_folder_files(owner, repo, path, branch="main"):
    """
    List files in a GitHub repository folder using the GitHub API.
    """

    contact_msg = (
        "Please contact the package owner: safwan.aljbaae@gmail.com"
    )

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    try:
        response = requests.get(url, params=params, headers=_get_headers(), timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Network error while accessing GitHub: {e}\n{contact_msg}"
        ) from e

    if response.status_code == 404:
        raise FileNotFoundError(
            f"Folder not found in GitHub repo '{owner}/{repo}'\n"
            f"Branch: '{branch}'\n"
            f"Path: '{path}'\n"
            f"{contact_msg}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API error {response.status_code}\n"
            f"Repo: '{owner}/{repo}'\n"
            f"Branch: '{branch}'\n"
            f"Path: '{path}'\n"
            f"Details: {response.text}\n"
            f"{contact_msg}"
        )

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected a folder at path '{path}', but got a file or invalid object.\n"
            f"{contact_msg}"
        )

    files = [item["name"] for item in data if item["type"] == "file"]

    if not files:
        raise FileNotFoundError(
            f"No files found in folder '{path}'\n"
            f"Repo: '{owner}/{repo}', Branch: '{branch}'\n"
            f"{contact_msg}"
        )

    return files


def download_github_folder(owner, repo, path, branch="main", output_dir="data"):
    """
    Download all files from a GitHub folder into a local directory.
    """

    contact_msg = f"Please contact the package owner: {CONTACT}"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    try:
        response = requests.get(url, params=params, headers=_get_headers(), timeout=20)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Network error while accessing GitHub: {e}\n{contact_msg}"
        ) from e

    if response.status_code == 404:
        raise FileNotFoundError(
            f"Folder not found: '{path}' in repo '{owner}/{repo}' (branch '{branch}')\n"
            f"{contact_msg}"
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API error {response.status_code}\n{response.text}\n{contact_msg}"
        )

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Expected a folder at '{path}', got something else.\n{contact_msg}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    for item in data:
        if item["type"] != "file":
            continue

        file_name = item["name"]
        download_url = item["download_url"]

        if download_url is None:
            raise RuntimeError(
                f"Missing download URL for file '{file_name}'\n{contact_msg}"
            )

        file_path = output_dir / file_name

        try:
            file_response = requests.get(download_url, headers=_get_headers(), timeout=30)
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
            f"No files found in folder '{path}'\n{contact_msg}"
        )

    return downloaded_files

if __name__ == "__main__":
    files = list_github_folder_files(
        owner="safwanaljbaae",
        repo="GravDyn",
        path="Data/Apophis/Pot_Expansion",
        branch="feature-test"
    )

    print(files)

    files = download_github_folder(
        owner="safwanaljbaae",
        repo="GravDyn",
        path=f"Data/Apophis/Pot_Expansion",
        branch="feature-test",
        output_dir=f"./Data/Apophis/Pot_Expansion"
    )
