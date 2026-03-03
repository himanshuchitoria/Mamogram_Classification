"""
=============================================================================
AI4BCancer - BI-RADS PDF Report Generator
=============================================================================
Generates medical-grade PDF reports following the BI-RADS standard using
reportlab. Each report includes:
  - Patient Demographics
  - Clinical Indication
  - Breast Density Assessment
  - Findings (AI Classification)
  - AI Probability Score
  - BI-RADS Assessment Category
  - Recommendations
  - Original Image + XAI Feature Importance (side-by-side)
  - Disclaimer
=============================================================================
"""

import io
import os
import base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)
from reportlab.lib import colors


# BI-RADS Categories mapping
BIRADS_CATEGORIES = {
    0: {"name": "BI-RADS 0: Incomplete", "description": "Need additional imaging evaluation",
        "recommendation": "Additional imaging evaluation and/or prior mammograms needed."},
    1: {"name": "BI-RADS 1: Negative", "description": "No significant findings",
        "recommendation": "Routine screening mammography (annual)."},
    2: {"name": "BI-RADS 2: Benign", "description": "Benign findings",
        "recommendation": "Routine screening mammography (annual)."},
    3: {"name": "BI-RADS 3: Probably Benign", "description": "Probably benign finding",
        "recommendation": "Short-interval follow-up (6 months) recommended."},
    4: {"name": "BI-RADS 4: Suspicious", "description": "Suspicious abnormality",
        "recommendation": "Tissue diagnosis (biopsy) is recommended."},
    5: {"name": "BI-RADS 5: Highly Suggestive of Malignancy", "description": "Highly suggestive of malignancy",
        "recommendation": "Appropriate action should be taken. Biopsy strongly recommended."},
    6: {"name": "BI-RADS 6: Known Malignancy", "description": "Known biopsy-proven malignancy",
        "recommendation": "Surgical excision when clinically appropriate."},
}


def determine_birads_category(prediction: str, confidence: float) -> int:
    """
    Map AI prediction and confidence to BI-RADS category.
    
    Args:
        prediction: "Benign" or "Malignant"
        confidence: probability (0-1) of the predicted class
    
    Returns:
        BI-RADS category number (1-5)
    """
    if prediction == "Benign":
        if confidence >= 0.95:
            return 1  # Negative - very confident benign
        elif confidence >= 0.85:
            return 2  # Benign finding
        else:
            return 3  # Probably benign
    else:  # Malignant
        if confidence >= 0.90:
            return 5  # Highly suggestive
        elif confidence >= 0.70:
            return 4  # Suspicious
        else:
            return 3  # Probably benign (low confidence malignant)


def generate_birads_report(
    patient_id: str,
    patient_name: str,
    patient_dob: str = "",
    clinical_notes: str = "",
    prediction: str = "Benign",
    confidence: float = 0.95,
    birads_category: int = None,
    feature_importance: list = None,
    original_image_b64: str = None,
    xai_plot_b64: str = None,
    report_date: str = None,
) -> bytes:
    """
    Generate a complete BI-RADS standard PDF report.
    
    Returns:
        bytes: PDF file content
    """
    if birads_category is None:
        birads_category = determine_birads_category(prediction, confidence)

    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y  %I:%M %p")

    birads_info = BIRADS_CATEGORIES.get(birads_category, BIRADS_CATEGORIES[0])

    # Create PDF buffer
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
    )

    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"],
        fontSize=20, textColor=HexColor("#1a365d"),
        alignment=TA_CENTER, spaceAfter=5*mm,
        fontName="Helvetica-Bold",
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#4a5568"),
        alignment=TA_CENTER, spaceAfter=8*mm,
    )
    
    section_style = ParagraphStyle(
        "SectionHeader", parent=styles["Heading2"],
        fontSize=13, textColor=HexColor("#2d3748"),
        spaceBefore=6*mm, spaceAfter=3*mm,
        fontName="Helvetica-Bold",
        borderWidth=0, borderPadding=0,
    )
    
    body_style = ParagraphStyle(
        "BodyText", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#2d3748"),
        spaceAfter=2*mm, alignment=TA_LEFT,
        leading=14,
    )
    
    body_justify = ParagraphStyle(
        "BodyJustify", parent=body_style,
        alignment=TA_JUSTIFY,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#718096"),
        alignment=TA_CENTER, spaceAfter=3*mm,
        leading=11,
    )

    # Build document elements
    elements = []

    # ----- HEADER -----
    elements.append(Paragraph("AI4BCancer", title_style))
    elements.append(Paragraph("Breast Cancer AI Classification Report", subtitle_style))
    elements.append(Paragraph(f"Report Date: {report_date}", subtitle_style))
    elements.append(HRFlowable(
        width="100%", color=HexColor("#2d3748"), thickness=1.5,
        spaceAfter=5*mm,
    ))

    # ----- PATIENT DEMOGRAPHICS -----
    elements.append(Paragraph("1. Patient Demographics", section_style))
    patient_data = [
        ["Patient ID:", patient_id or "N/A"],
        ["Patient Name:", patient_name or "N/A"],
        ["Date of Birth:", patient_dob or "N/A"],
        ["Report Date:", report_date],
    ]
    patient_table = Table(patient_data, colWidths=[40*mm, 120*mm])
    patient_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#2d3748")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 3*mm))

    # ----- CLINICAL INDICATION -----
    elements.append(Paragraph("2. Clinical Indication", section_style))
    clinical_text = clinical_notes if clinical_notes else "Routine breast cancer screening / AI-assisted analysis of provided specimen data."
    elements.append(Paragraph(clinical_text, body_justify))

    # ----- AI CLASSIFICATION RESULTS -----
    elements.append(Paragraph("3. AI Classification Results", section_style))

    # Classification result box
    pred_color = "#e53e3e" if prediction == "Malignant" else "#38a169"
    pred_bg = "#fff5f5" if prediction == "Malignant" else "#f0fff4"
    
    result_data = [
        ["Classification:", Paragraph(f'<font color="{pred_color}"><b>{prediction}</b></font>', body_style)],
        ["Confidence Score:", f"{confidence*100:.1f}%"],
        ["BI-RADS Category:", birads_info["name"]],
        ["Assessment:", birads_info["description"]],
    ]
    result_table = Table(result_data, colWidths=[45*mm, 115*mm])
    result_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#2d3748")),
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(pred_bg)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 1, HexColor(pred_color)),
        ("ROUNDEDCORNERS", [3, 3, 3, 3]),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 3*mm))

    # ----- BREAST DENSITY -----
    elements.append(Paragraph("4. Breast Density", section_style))
    elements.append(Paragraph(
        "Breast density assessment based on image analysis: "
        "<b>Not applicable</b> — classification performed on extracted morphological features. "
        "Manual density assessment by radiologist recommended.",
        body_justify
    ))

    # ----- FINDINGS -----
    elements.append(Paragraph("5. Findings", section_style))
    
    if feature_importance and len(feature_importance) > 0:
        elements.append(Paragraph(
            "The AI model identified the following features as most significant in the classification:",
            body_justify,
        ))
        elements.append(Spacer(1, 2*mm))
        
        # Top features table
        feat_header = [["Rank", "Feature", "Contribution", "Direction"]]
        feat_rows = []
        for i, feat in enumerate(feature_importance[:10], 1):
            name = feat.get("feature", "N/A")
            weight = feat.get("weight", 0)
            direction = "→ Malignant" if weight > 0 else "→ Benign"
            feat_rows.append([str(i), name, f"{weight:+.4f}", direction])
        
        feat_table = Table(feat_header + feat_rows, colWidths=[12*mm, 70*mm, 35*mm, 35*mm])
        feat_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#edf2f7")),
            ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#2d3748")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ]))
        elements.append(feat_table)
    else:
        elements.append(Paragraph("Feature importance data not available.", body_style))

    # ----- IMAGES: Original + XAI side-by-side -----
    if original_image_b64 or xai_plot_b64:
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("6. Visual Analysis", section_style))
        
        image_cells = []
        image_headers = []
        
        if original_image_b64:
            try:
                img_data = base64.b64decode(original_image_b64)
                img_buf = io.BytesIO(img_data)
                img = RLImage(img_buf, width=75*mm, height=60*mm, kind="proportional")
                image_cells.append(img)
                image_headers.append("Original Image")
            except Exception:
                image_cells.append(Paragraph("Image not available", body_style))
                image_headers.append("Original Image")

        if xai_plot_b64:
            try:
                xai_data = base64.b64decode(xai_plot_b64)
                xai_buf = io.BytesIO(xai_data)
                xai_img = RLImage(xai_buf, width=75*mm, height=60*mm, kind="proportional")
                image_cells.append(xai_img)
                image_headers.append("AI Feature Importance")
            except Exception:
                image_cells.append(Paragraph("XAI plot not available", body_style))
                image_headers.append("AI Feature Importance")

        if image_cells:
            imgs_table = Table(
                [image_headers, image_cells],
                colWidths=[80*mm] * len(image_cells),
            )
            imgs_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(imgs_table)

    # ----- BI-RADS ASSESSMENT -----
    section_num = 7 if (original_image_b64 or xai_plot_b64) else 6
    elements.append(Paragraph(f"{section_num}. BI-RADS Assessment", section_style))
    elements.append(Paragraph(f"<b>{birads_info['name']}</b>", body_style))
    elements.append(Paragraph(birads_info["description"], body_style))

    # ----- RECOMMENDATIONS -----
    elements.append(Paragraph(f"{section_num + 1}. Recommendations", section_style))
    elements.append(Paragraph(birads_info["recommendation"], body_justify))

    # ----- DISCLAIMER -----
    elements.append(Spacer(1, 10*mm))
    elements.append(HRFlowable(width="100%", color=HexColor("#cbd5e0"), thickness=0.5))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-assisted analysis system (AI4BCancer) "
        "and is intended for informational and screening assistance purposes only. "
        "It does NOT constitute a medical diagnosis. Final clinical decisions must be made by a "
        "qualified radiologist or oncologist based on comprehensive clinical evaluation. "
        "The AI model has been trained on the Wisconsin Breast Cancer dataset and its predictions "
        "should be validated against clinical findings.",
        disclaimer_style,
    ))
    elements.append(Paragraph(
        f"Generated by AI4BCancer v1.0 | {report_date}",
        disclaimer_style,
    ))

    # Build PDF
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
