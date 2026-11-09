"""Download the original CAD-TCM file (CC BY 4.0). Requires Internet access."""
import argparse
import hashlib
import shutil
from tempfile import NamedTemporaryFile
from pathlib import Path
from src.config import DATA_DIR, DATASET_URL


REVISION = '897d692550985e6890980d76cf932c4d05b50eae'
SHA256 = '9fecf011bd001607b110df4c7da715d6401070c1c75c8f90babb3fe716b00eee'


def matches_source(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == SHA256


def download(force: bool = False) -> Path:
    destination = DATA_DIR / 'cad_tcm.csv'
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    if destination.exists() and not force and matches_source(destination):
        print(f'기존 CSV 사용: {destination} ({destination.stat().st_size:,} bytes)')
        return destination
    try:
        from huggingface_hub import hf_hub_download
        cached_file = hf_hub_download(repo_id='lifelonglab/CAD-TCM',
                                      filename='data.csv', revision=REVISION, repo_type='dataset')
        with NamedTemporaryFile(dir=DATA_DIR, delete=False) as tmp:
            staging = Path(tmp.name)
        try:
            shutil.copyfile(cached_file, staging)
            if not matches_source(staging):
                raise ValueError('다운로드된 파일의 SHA-256이 원본과 다릅니다.')
            staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)
    except Exception as exc:
        print(f'자동 다운로드 실패: {exc}')
        print('브라우저에서 아래 링크를 열어 data.csv를 내려받아 저장하세요:')
        print(DATASET_URL + '/tree/main')
        print('저장 위치: ' + str(destination))
        raise SystemExit(1) from exc
    print(f'다운로드 완료: {destination} ({destination.stat().st_size:,} bytes)')
    return destination

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='원본 파일 다시 다운로드')
    args = parser.parse_args()
    download(force=args.force)
