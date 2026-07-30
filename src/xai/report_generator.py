import os
import json
from pathlib import Path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.saveState()
    canvas.setFont('Times-Roman', 10)
    canvas.drawRightString(doc.pagesize[0] - 20, 20, text)
    canvas.restoreState()

def generate_pdf_report(pdf_path, title, metadata_dict, image_paths_list):
    """
    Generate a basic publication-ready PDF using reportlab.
    """
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    Story = []
    
    # Title
    Story.append(Paragraph(title, styles['Title']))
    Story.append(Spacer(1, 12))
    
    # Global metadata
    if metadata_dict:
        Story.append(Paragraph("Run Details & Metadata", styles['Heading2']))
        data = [[str(k), str(v)] for k, v in metadata_dict.items() if not isinstance(v, (dict, list))]
        if data:
            t = Table(data, colWidths=[200, 400], style=[
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTSIZE', (0,0), (-1,-1), 10)
            ])
            Story.append(t)
            Story.append(Spacer(1, 24))
            
    # Add images
    if image_paths_list:
        Story.append(PageBreak())
        Story.append(Paragraph("XAI Visualizations Gallery", styles['Heading2']))
        Story.append(Spacer(1, 12))
        
        for idx, img_path in enumerate(image_paths_list):
            if os.path.exists(img_path):
                img_reader = ImageReader(img_path)
                iw, ih = img_reader.getSize()
                aspect = ih / float(iw)
                target_width = 700
                target_height = target_width * aspect
                
                img = Image(img_path, width=target_width, height=target_height)
                Story.append(img)
                caption = Paragraph(f"Figure {idx+1}: {os.path.basename(os.path.dirname(img_path))} XAI Analysis", styles['Normal'])
                Story.append(caption)
                Story.append(Spacer(1, 24))
            
    doc.build(Story, onFirstPage=add_page_number, onLaterPages=add_page_number)

def generate_all_reports(metrics, manifest, xai_dir, reports_dir):
    """
    Generates summary.pdf, xai_gallery.pdf, failure_analysis.pdf
    """
    import glob
    
    xai_dir = Path(xai_dir)
    reports_dir = Path(reports_dir)
    
    all_panels = sorted(glob.glob(str(xai_dir / "image_*" / "panel.png")))
    
    # 1. Summary PDF (Metrics + a few samples)
    summary_metadata = {**metrics, **manifest}
    generate_pdf_report(reports_dir / "summary.pdf", "FusionMedAI: XAI Summary Report", summary_metadata, all_panels[:10])
    
    # 2. Complete Gallery
    generate_pdf_report(reports_dir / "xai_gallery.pdf", "FusionMedAI: Complete XAI Gallery", {}, all_panels)
    
    # 3. Failure Analysis
    failures = []
    for p in all_panels:
        meta_path = Path(p).parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if not meta.get("is_correct", True):
                    failures.append(p)
                    
    generate_pdf_report(reports_dir / "failure_analysis.pdf", "FusionMedAI: XAI Failure Analysis", {}, failures)
