import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import pandas as pd

# Add project root to sys.path
root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.foot.config as config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def inspect_dataset(raw_dir: Path):
    total_files = 0
    total_images = 0
    non_image_files = []
    
    file_extensions = Counter()
    formats = Counter()
    dimensions = Counter()
    channels_counter = Counter()
    modes_counter = Counter()
    
    images_per_class = Counter()
    images_per_split_class = defaultdict(Counter)
    images_per_split = Counter()
    
    file_sizes_bytes = []
    folder_names = set()
    folder_structure = {}
    
    image_details = []
    
    for item in raw_dir.rglob("*"):
        if item.is_dir():
            folder_names.add(item.relative_to(raw_dir).as_posix())
            continue
            
        total_files += 1
        rel_path = item.relative_to(raw_dir).as_posix()
        ext = item.suffix.lower()
        file_extensions[ext if ext else "<no_ext>"] += 1
        
        file_size = item.stat().st_size
        file_sizes_bytes.append(file_size)
        
        # Determine split and class if in standard train/valid/test/Grade X layout
        parts = item.relative_to(raw_dir).parts
        split_name = "unassigned"
        class_name = "unassigned"
        
        if len(parts) >= 2:
            potential_split = parts[0]
            potential_class = parts[1]
            if potential_split.lower() in ["train", "valid", "val", "test"]:
                split_name = potential_split
                class_name = potential_class
            elif "grade" in potential_split.lower():
                class_name = potential_split
        
        if ext in IMAGE_EXTENSIONS:
            total_images += 1
            images_per_class[class_name] += 1
            images_per_split_class[split_name][class_name] += 1
            images_per_split[split_name] += 1
            
            try:
                with Image.open(item) as img:
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format if img.format else ext[1:].upper()
                    channels = len(img.getbands())
                    
                    dim_str = f"{width} × {height}"
                    dimensions[dim_str] += 1
                    formats[fmt] += 1
                    channels_counter[f"{channels}-channel ({mode})"] += 1
                    modes_counter[mode] += 1
                    
                    image_details.append({
                        "path": rel_path,
                        "split": split_name,
                        "class": class_name,
                        "width": width,
                        "height": height,
                        "dimension": dim_str,
                        "mode": mode,
                        "channels": channels,
                        "format": fmt,
                        "size_bytes": file_size
                    })
            except Exception as e:
                non_image_files.append({"path": rel_path, "error": str(e)})
        else:
            non_image_files.append({"path": rel_path, "type": "non_image"})
            
    # Calculate file size statistics
    total_size_mb = sum(file_sizes_bytes) / (1024 * 1024) if file_sizes_bytes else 0
    avg_size_kb = (sum(file_sizes_bytes) / len(file_sizes_bytes)) / 1024 if file_sizes_bytes else 0
    min_size_kb = min(file_sizes_bytes) / 1024 if file_sizes_bytes else 0
    max_size_kb = max(file_sizes_bytes) / 1024 if file_sizes_bytes else 0
    
    inventory_data = {
        "dataset_name": "Foot Diabetic Ulcer (DFU - Wagner 4-Class)",
        "raw_directory": str(raw_dir),
        "total_files": total_files,
        "total_images": total_images,
        "total_size_mb": round(total_size_mb, 2),
        "file_size_stats_kb": {
            "min": round(min_size_kb, 2),
            "max": round(max_size_kb, 2),
            "mean": round(avg_size_kb, 2)
        },
        "file_extensions": dict(file_extensions),
        "formats": dict(formats),
        "classes": dict(images_per_class),
        "splits": dict(images_per_split),
        "splits_by_class": {k: dict(v) for k, v in images_per_split_class.items()},
        "dimensions": dict(dimensions.most_common(15)),
        "unique_dimension_count": len(dimensions),
        "channels": dict(channels_counter),
        "folder_names": sorted(list(folder_names)),
        "non_image_files_count": len(non_image_files)
    }
    
    return inventory_data, image_details

def generate_markdown_report(data: dict) -> str:
    lines = []
    lines.append("# Dataset Inventory Report: Foot DFU (Wagner 4-Class)")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("```text")
    lines.append("Dataset Inventory")
    lines.append("────────────────────────────────────────────")
    lines.append(f"Total files:       {data['total_files']:,}")
    lines.append(f"Total images:      {data['total_images']:,}")
    lines.append(f"Total size:        {data['total_size_mb']:.2f} MB")
    lines.append(f"Mean file size:    {data['file_size_stats_kb']['mean']:.2f} KB")
    lines.append("")
    lines.append("Classes (Overall):")
    for cls_name, count in sorted(data['classes'].items()):
        lines.append(f"  {cls_name:<16}: {count:,}")
    lines.append("")
    lines.append("Formats:")
    for fmt, count in sorted(data['formats'].items()):
        lines.append(f"  {fmt:<16}: {count:,}")
    lines.append("")
    lines.append("Color Channels:")
    for ch, count in sorted(data['channels'].items()):
        lines.append(f"  {ch:<16}: {count:,}")
    lines.append("")
    lines.append("Top Image Dimensions:")
    for dim, count in list(data['dimensions'].items())[:10]:
        lines.append(f"  {dim:<16}: {count:,}")
    lines.append("```")
    lines.append("")
    
    lines.append("## Breakdown by Split")
    lines.append("")
    lines.append("| Split | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total Images |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    splits_dict = data['splits_by_class']
    for split in sorted(splits_dict.keys()):
        cls_map = splits_dict[split]
        g1 = cls_map.get("Grade 1", 0)
        g2 = cls_map.get("Grade 2", 0)
        g3 = cls_map.get("Grade 3", 0)
        g4 = cls_map.get("Grade 4", 0)
        tot = sum(cls_map.values())
        lines.append(f"| **{split}** | {g1:,} | {g2:,} | {g3:,} | {g4:,} | **{tot:,}** |")
        
    lines.append("")
    lines.append("## Folder Structure")
    lines.append("```directory")
    lines.append("datasets/foot/raw/")
    for f in data['folder_names']:
        lines.append(f"├── {f}")
    lines.append("```")
    lines.append("")
    lines.append("## File Extension Breakdown")
    lines.append("")
    lines.append("| Extension | Count |")
    lines.append("| :--- | :--- |")
    for ext, count in sorted(data['file_extensions'].items()):
        lines.append(f"| `{ext}` | {count:,} |")
        
    return "\n".join(lines)

def main():
    raw_dir = config.RAW_DATA
    print(f"Inspecting raw dataset directory: {raw_dir}")
    
    if not raw_dir.exists():
        print(f"Error: Raw directory does not exist at {raw_dir}")
        sys.exit(1)
        
    inventory_data, image_details = inspect_dataset(raw_dir)
    
    # Save JSON and Markdown outputs to config.METADATA_DIR and interim/metadata
    metadata_dirs = [
        config.METADATA_DIR,
        config.DATASET_ROOT / "interim" / "metadata"
    ]
    
    for m_dir in metadata_dirs:
        m_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = m_dir / "inventory.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(inventory_data, f, indent=2)
        print(f"  [OK] Saved JSON inventory: {json_path}")
        
        md_report = generate_markdown_report(inventory_data)
        md_path = m_dir / "dataset_inventory.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  [OK] Saved Markdown report: {md_path}")
        
    # Also save detailed image-level CSV statistics
    stats_dir = config.METADATA_STATISTICS_DIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    df_details = pd.DataFrame(image_details)
    csv_path = stats_dir / "foot_image_inventory.csv"
    df_details.to_csv(csv_path, index=False)
    print(f"  [OK] Saved image-level CSV statistics: {csv_path}")
    
    print("\n==========================================")
    print("DATASET INVENTORY COMPLETED")
    print("==========================================")
    try:
        print(generate_markdown_report(inventory_data))
    except UnicodeEncodeError:
        print(generate_markdown_report(inventory_data).encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
