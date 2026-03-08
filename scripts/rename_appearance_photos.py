from pathlib import Path
import shutil
import argparse

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photos_dir",
        required=True,
        help="Diretório com as fotos a renomear"
    )
    args = parser.parse_args()

    root = Path(args.photos_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {root}")

    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files = sorted(files, key=lambda p: p.name.lower())

    print(f"Pasta alvo: {root}")
    print(f"Imagens encontradas: {len(files)}")

    if not files:
        print("Nenhuma imagem encontrada com extensões válidas.")
        return

    backup_dir = root / "_backup_original_names"
    backup_dir.mkdir(exist_ok=True)

    for f in files:
        backup_path = backup_dir / f.name
        if not backup_path.exists():
            shutil.copy2(f, backup_path)

    temp_files = []
    for idx, f in enumerate(files, start=1):
        temp_name = root / f"__temp_{idx:04d}{f.suffix.lower()}"
        f.rename(temp_name)
        temp_files.append(temp_name)

    for idx, f in enumerate(temp_files, start=1):
        new_name = root / f"img_{idx:03d}{f.suffix.lower()}"
        f.rename(new_name)
        print(f"{f.name} -> {new_name.name}")

    print("Renomeação concluída com sucesso.")
    print(f"Backup salvo em: {backup_dir}")

if __name__ == "__main__":
    main()
