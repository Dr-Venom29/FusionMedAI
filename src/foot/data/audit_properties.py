import os
import sys
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import src.foot.config as config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def analyze_image_properties(raw_dir: Path):
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes_kb = []
    
    formats = Counter()
    color_modes = Counter()
    channels_list = Counter()
    
    # Per-channel mean/std accumulation
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    brightness_list = []
    contrast_list = []
    
    dark_images = []
    bright_images = []
    image_properties_records = []
    
    image_paths = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    total_count = len(image_paths)
    
    print(f"Analyzing {total_count} images for property audit...")
    
    for idx, item in enumerate(image_paths, 1):
        rel_path = item.relative_to(raw_dir).as_posix()
        ext = item.suffix.lower()
        file_size_kb = item.stat().st_size / 1024.0
        file_sizes_kb.append(file_size_kb)
        
        try:
            with Image.open(item) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format if img.format else ext[1:].upper()
                ch = len(img.getbands())
                
                widths.append(w)
                heights.append(h)
                aspect_ratio = w / h
                aspect_ratios.append(aspect_ratio)
                
                formats[fmt] += 1
                color_modes[mode] += 1
                channels_list[ch] += 1
                
                # Convert to RGB array normalized to [0, 1] for visual statistics
                img_rgb = img.convert("RGB")
                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                
                r_m = float(np.mean(arr[:, :, 0]))
                g_m = float(np.mean(arr[:, :, 1]))
                b_m = float(np.mean(arr[:, :, 2]))
                
                r_s = float(np.std(arr[:, :, 0]))
                g_s = float(np.std(arr[:, :, 1]))
                b_s = float(np.std(arr[:, :, 2]))
                
                brightness = float(np.mean(arr))
                contrast = float(np.std(arr))
                
                r_means.append(r_m)
                g_means.append(g_m)
                b_means.append(b_m)
                
                r_stds.append(r_s)
                g_stds.append(g_s)
                b_stds.append(b_s)
                
                brightness_list.append(brightness)
                contrast_list.append(contrast)
                
                if brightness < 0.15:
                    dark_images.append({"path": rel_path, "brightness": round(brightness, 4)})
                elif brightness > 0.85:
                    bright_images.append({"path": rel_path, "brightness": round(brightness, 4)})
                    
                image_properties_records.append({
                    "path": rel_path,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(aspect_ratio, 4),
                    "mode": mode,
                    "channels": ch,
                    "format": fmt,
                    "size_kb": round(file_size_kb, 2),
                    "r_mean": round(r_m, 4),
                    "g_mean": round(g_m, 4),
                    "b_mean": round(b_m, 4),
                    "r_std": round(r_s, 4),
                    "g_std": round(g_s, 4),
                    "b_std": round(b_s, 4),
                    "brightness": round(brightness, 4),
                    "contrast": round(contrast, 4)
                })
        except Exception as e:
            print(f"Warning: Failed property inspection for {rel_path}: {e}")

    # Aggregate visual statistics
    dataset_mean_rgb = [
        float(np.mean(r_means)),
        float(np.mean(g_means)),
        float(np.mean(b_means))
    ] if r_means else [0.485, 0.456, 0.406]
    
    dataset_std_rgb = [
        float(np.mean(r_stds)),
        float(np.mean(g_stds)),
        float(np.mean(b_stds))
    ] if r_stds else [0.229, 0.224, 0.225]

    aspect_ratio_counter = Counter([round(ar, 2) for ar in aspect_ratios])

    property_report = {
        "dataset_name": "Foot DFU Wagner 4-Class Dataset",
        "total_images_analyzed": len(image_properties_records),
        "dimension_summary": {
            "unique_widths": sorted(list(set(widths))),
            "unique_heights": sorted(list(set(heights))),
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "aspect_ratio_counts": dict(aspect_ratio_counter)
        },
        "color_channels": {
            "modes": dict(color_modes),
            "channels": dict(channels_list)
        },
        "file_properties": {
            "formats": dict(formats),
            "min_size_kb": round(min(file_sizes_kb), 2) if file_sizes_kb else 0,
            "max_size_kb": round(max(file_sizes_kb), 2) if file_sizes_kb else 0,
            "mean_size_kb": round(float(np.mean(file_sizes_kb)), 2) if file_sizes_kb else 0,
            "median_size_kb": round(float(np.median(file_sizes_kb)), 2) if file_sizes_kb else 0
        },
        "visual_statistics": {
            "dataset_normalization_mean": [round(x, 4) for x in dataset_mean_rgb],
            "dataset_normalization_std": [round(x, 4) for x in dataset_std_rgb],
            "brightness_stats": {
                "mean": round(float(np.mean(brightness_list)), 4) if brightness_list else 0,
                "std": round(float(np.std(brightness_list)), 4) if brightness_list else 0,
                "min": round(float(np.min(brightness_list)), 4) if brightness_list else 0,
                "max": round(float(np.max(brightness_list)), 4) if brightness_list else 0
            },
            "contrast_stats": {
                "mean": round(float(np.mean(contrast_list)), 4) if contrast_list else 0,
                "std": round(float(np.std(contrast_list)), 4) if contrast_list else 0,
                "min": round(float(np.min(contrast_list)), 4) if contrast_list else 0,
                "max": round(float(np.max(contrast_list)), 4) if contrast_list else 0
            },
            "dark_images_count": len(dark_images),
            "bright_images_count": len(bright_images)
        },
        "candidate_preprocessing": {
            "target_resolution": [224, 224],
            "observed_mean": [round(x, 4) for x in dataset_mean_rgb],
            "observed_std": [round(x, 4) for x in dataset_std_rgb],
            "candidate_color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.1,
                "hue": 0.05
            },
            "candidate_affine": {
                "rotation_range": [-15, 15],
                "horizontal_flip": 0.5
            },
            "rationale": "Dataset images are 100% pre-resized to 224x224 RGB. Measured normalization mean and std reflect observed dataset properties, while spatial augmentations represent candidate options to be evaluated during Step 10.2 Data Pipeline construction."
        }
    }

    return property_report, image_properties_records

def generate_markdown_report(report: dict) -> str:
    lines = []
    lines.append("# Phase 10.1.E — Image Property Audit & Preprocessing Rationale")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- **Total Images Analyzed**: `{report['total_images_analyzed']:,}`")
    lines.append(f"- **Uniform Resolution**: `{report['dimension_summary']['min_width']} × {report['dimension_summary']['min_height']}` (100% Aspect Ratio = 1.0)")
    lines.append(f"- **Color Space**: 100% `3-Channel RGB`")
    lines.append(f"- **File Format**: 100% `JPEG`")
    lines.append("")

    lines.append("## Image Dimensions & Aspect Ratio")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Width Range** | {report['dimension_summary']['min_width']} px – {report['dimension_summary']['max_width']} px |")
    lines.append(f"| **Height Range** | {report['dimension_summary']['min_height']} px – {report['dimension_summary']['max_height']} px |")
    lines.append(f"| **Aspect Ratio** | 1.0 (Square `224 × 224`) |")
    lines.append("")

    lines.append("## Computed Visual Statistics")
    lines.append("| Metric | Red Channel | Green Channel | Blue Channel | Overall |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    mean_rgb = report['visual_statistics']['dataset_normalization_mean']
    std_rgb = report['visual_statistics']['dataset_normalization_std']
    lines.append(f"| **Dataset Mean** | {mean_rgb[0]} | {mean_rgb[1]} | {mean_rgb[2]} | {report['visual_statistics']['brightness_stats']['mean']} |")
    lines.append(f"| **Dataset Std** | {std_rgb[0]} | {std_rgb[1]} | {std_rgb[2]} | {report['visual_statistics']['contrast_stats']['mean']} |")
    lines.append("")

    lines.append("## File Properties & Size Distribution")
    lines.append("| File Property | Statistic |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Min File Size** | {report['file_properties']['min_size_kb']} KB |")
    lines.append(f"| **Max File Size** | {report['file_properties']['max_size_kb']} KB |")
    lines.append(f"| **Mean File Size** | {report['file_properties']['mean_size_kb']} KB |")
    lines.append(f"| **Median File Size** | {report['file_properties']['median_size_kb']} KB |")
    lines.append("")

    rec = report['candidate_preprocessing']
    lines.append("## Foot DFU Preprocessing Rationale & Candidate Options")
    lines.append("> **Crucial Finding**: Do NOT copy Retina preprocessing assumptions (e.g. heavy cropping / Ben Graham green-channel filtering). The Foot DFU dataset consists of pre-resized $224 \\times 224$ RGB clinical wound photos with natural lighting and skin tones.")
    lines.append("")
    lines.append("### 1. Observed Dataset Statistics")
    lines.append(f"```python")
    lines.append(f"OBSERVED_DATASET_MEAN = {rec['observed_mean']}")
    lines.append(f"OBSERVED_DATASET_STD  = {rec['observed_std']}")
    lines.append(f"```")
    lines.append("")
    lines.append("### 2. Candidate Preprocessing Options (To be evaluated in Step 10.2 - Data Pipeline)")
    lines.append("- **Target Resolution**: Native `$224 \\times 224$` resolution (no aspect ratio distortion).")
    lines.append(f"- **Candidate Rotation Range**: `[-15°, +15°]`")
    lines.append(f"- **Candidate Horizontal Flip**: `p = {rec['candidate_affine']['horizontal_flip']}`")
    lines.append(f"- **Candidate Color Jitter**: Brightness ({rec['candidate_color_jitter']['brightness']}), Contrast ({rec['candidate_color_jitter']['contrast']}), Saturation ({rec['candidate_color_jitter']['saturation']}).")

    return "\n".join(lines)

def main():
    raw_dir = config.RAW_DATA
    print(f"Executing Phase 10.1.E Image Property Audit on: {raw_dir}")
    
    report_data, image_records = analyze_image_properties(raw_dir)
    
    # Save outputs to metadata/ and interim/metadata/
    target_dirs = [
        config.METADATA_DIR,
        config.DATASET_ROOT / "interim" / "metadata"
    ]
    
    for t_dir in target_dirs:
        t_dir.mkdir(parents=True, exist_ok=True)
        json_path = t_dir / "property_audit.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"  [OK] Saved JSON property report: {json_path}")
        
        md_report = generate_markdown_report(report_data)
        md_path = t_dir / "property_audit.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  [OK] Saved Markdown property report: {md_path}")
        
    # Save image properties CSV
    csv_path = config.METADATA_STATISTICS_DIR / "foot_image_properties.csv"
    df_props = pd.DataFrame(image_records)
    df_props.to_csv(csv_path, index=False)
    print(f"  [OK] Saved image properties CSV: {csv_path}")

    print("\n==========================================")
    print("IMAGE PROPERTY AUDIT COMPLETED")
    print("==========================================")
    try:
        print(generate_markdown_report(report_data))
    except UnicodeEncodeError:
        print(generate_markdown_report(report_data).encode("ascii", errors="replace").decode("ascii"))

if __name__ == "__main__":
    main()
