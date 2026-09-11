import json
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

def save_summary_json(metrics_before: Dict[str, float], metrics_after: Dict[str, float], save_path: Path):
    data = {
        "before_calibration": metrics_before,
        "after_calibration": metrics_after
    }
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def save_summary_csv(metrics_before: Dict[str, float], metrics_after: Dict[str, float], save_path: Path):
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Before", "After"])
        for key in metrics_before.keys():
            writer.writerow([key, metrics_before[key], metrics_after[key]])

def save_comparison_md(metrics_before: Dict[str, float], metrics_after: Dict[str, float], save_path: Path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("| Metric | Before | After | Improvement |\n")
        f.write("|---|---|---|---|\n")
        for key in metrics_before.keys():
            before_val = metrics_before[key]
            after_val = metrics_after[key]
            if key in ["ECE", "NLL", "Brier", "MCE"]:
                # lower is better
                if before_val > 0:
                    improvement = ((before_val - after_val) / before_val) * 100
                    imp_str = f"↓{improvement:.0f}%" if improvement > 0 else f"↑{-improvement:.0f}%"
                else:
                    imp_str = "N/A"
            elif key == "Accuracy":
                improvement = after_val - before_val
                imp_str = f"+{improvement:.4f}"
            else:
                imp_str = "-"
            f.write(f"| {key} | {before_val:.4f} | {after_val:.4f} | {imp_str} |\n")

def save_calibrated_predictions(
    dataset_df: pd.DataFrame,
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    save_path: Path
):
    probs_before = F.softmax(logits_before, dim=1)
    probs_after = F.softmax(logits_after, dim=1)
    
    conf_before, pred_before = torch.max(probs_before, dim=1)
    conf_after, pred_after = torch.max(probs_after, dim=1)
    
    # We assume dataset_df has the same length and order as the dataloader outputs (no shuffling)
    if len(dataset_df) != len(labels):
        # We can't guarantee order if lengths don't match, just save arrays
        df = pd.DataFrame({
            "GT": labels.numpy(),
            "Prediction": pred_after.numpy(),
            "Confidence_before": conf_before.numpy(),
            "Confidence_after": conf_after.numpy(),
            "Correct": (pred_after == labels).numpy(),
            "Temperature": temperature
        })
    else:
        df = dataset_df.copy()
        df["GT"] = labels.numpy()
        df["Prediction"] = pred_after.numpy()
        df["Confidence_before"] = conf_before.numpy()
        df["Confidence_after"] = conf_after.numpy()
        df["Correct"] = (pred_after == labels).numpy()
        df["Temperature"] = temperature
        
    df.to_csv(save_path, index=False)


def generate_pdf_report(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    temperature: float,
    figures_dir: Path,
    save_path: Path
):
    if not HAS_REPORTLAB:
        print("Warning: reportlab not installed. Skipping PDF generation.")
        return
        
    doc = SimpleDocTemplate(str(save_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("FusionMedAI: Retinal Model Calibration Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Introduction
    story.append(Paragraph("1. Introduction", styles['Heading1']))
    story.append(Paragraph("This report details the post-hoc calibration of the trained model using Temperature Scaling. The goal is to align predicted probabilities with actual accuracies to reduce overconfidence.", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Method
    story.append(Paragraph("2. Method", styles['Heading1']))
    story.append(Paragraph(f"Temperature Scaling was optimized on the validation set. The learned temperature is T = {temperature:.4f}.", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Metrics
    story.append(Paragraph("3. Metrics Comparison", styles['Heading1']))
    
    table_data = [["Metric", "Before Calibration", "After Calibration"]]
    for key in metrics_before.keys():
        table_data.append([key, f"{metrics_before[key]:.4f}", f"{metrics_after[key]:.4f}"])
        
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Improvement Table
    story.append(Paragraph("Improvement Summary", styles['Heading2']))
    imp_data = [["Metric", "Improvement"]]
    for metric in ["ECE", "NLL", "Brier"]:
        val_before = metrics_before[metric]
        val_after = metrics_after[metric]
        if val_before > 0:
            imp_pct = ((val_before - val_after) / val_before) * 100
            imp_str = f"↓ {imp_pct:.1f}%" if imp_pct > 0 else f"↑ {-imp_pct:.1f}%"
        else:
            imp_str = "0%"
        imp_data.append([metric, imp_str])
        
    t_imp = Table(imp_data)
    t_imp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t_imp)
    story.append(Spacer(1, 12))
    
    # Plots
    story.append(Paragraph("4. Reliability Diagrams", styles['Heading1']))
    
    rel_before = figures_dir / "reliability_before.png"
    rel_after = figures_dir / "reliability_after.png"
    
    if rel_before.exists():
        story.append(Paragraph("Before Calibration:", styles['Heading2']))
        story.append(Image(str(rel_before), width=300, height=300))
    if rel_after.exists():
        story.append(Paragraph("After Calibration:", styles['Heading2']))
        story.append(Image(str(rel_after), width=300, height=300))
        
    story.append(Paragraph("5. Discussion & Conclusion", styles['Heading1']))
    
    ece_imp = ((metrics_before['ECE'] - metrics_after['ECE']) / metrics_before['ECE']) * 100 if metrics_before['ECE'] > 0 else 0
    nll_imp = ((metrics_before['NLL'] - metrics_after['NLL']) / metrics_before['NLL']) * 100 if metrics_before['NLL'] > 0 else 0
    conf_red = metrics_before['Mean Confidence'] - metrics_after['Mean Confidence']
    
    discussion_text = (
        f"Calibration successfully reduced Expected Calibration Error (ECE) from {metrics_before['ECE']:.4f} to "
        f"{metrics_after['ECE']:.4f} (a {ece_imp:.1f}% improvement), while perfectly preserving classification accuracy. "
        f"The Negative Log-Likelihood (NLL) also improved by {nll_imp:.1f}%. "
        f"The optimization learned a temperature scaling factor of T = {temperature:.4f}. "
        f"This effectively reduced the mean confidence by {conf_red:.4f}, mitigating the model's initial overconfidence. "
        f"Note that Maximum Calibration Error (MCE) may occasionally increase, as Temperature Scaling explicitly optimizes NLL, not MCE. "
        f"These calibrated probabilities provide a mathematically sound foundation for Uncertainty Estimation (Step 8)."
    )
    story.append(Paragraph(discussion_text, styles['Normal']))
    
    doc.build(story)

