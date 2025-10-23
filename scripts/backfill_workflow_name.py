import json
from pathlib import Path
import argparse

AUG_DIR = Path("data/workflows/training_aug")


def backfill(dir_path: Path, dry_run: bool = False):
    files = sorted(dir_path.glob("*.json"))
    updated = 0
    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Skip {f.name}: load error {e}")
            continue
        if 'workflow_name' in data:
            continue
        # derive base name from top-level name or filename stem
        base = data.get('name') or f.stem
        # ensure uniqueness: if base already ends with -augX keep it
        if not base:
            base = f.stem
        data['workflow_name'] = base
        if dry_run:
            print(f"[DRY] Would add workflow_name to {f.name} -> {base}")
        else:
            with open(f, 'w') as fh:
                json.dump(data, fh, indent=2)
            print(f"Added workflow_name to {f.name} -> {base}")
            updated += 1
    print(f"Backfill complete. Updated {updated} files.")


def main():
    parser = argparse.ArgumentParser(description="Backfill missing workflow_name field in augmented workflows.")
    parser.add_argument('--dir', type=str, default=str(AUG_DIR), help='Directory containing augmented workflows')
    parser.add_argument('--dry', action='store_true', help='Dry run only')
    args = parser.parse_args()
    backfill(Path(args.dir), dry_run=args.dry)

if __name__ == '__main__':
    main()
