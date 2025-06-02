from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from datetime import datetime
import os
import textwrap

def generate_pdf_report(
    report_path,
    prediction_label,
    confidence,
    reference_images,
    gradcam_images,
    graph_image,
    prediction_explanation=None,
    graph_explanation=None
):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    def draw_wrapped_text(text, x, y, max_width, font_size=12, leading=14):
        c.setFont("Helvetica", font_size)
        wrapped = textwrap.wrap(text, width=max_width)
        for line in wrapped:
            c.drawString(x, y, line)
            y -= leading
        return y

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Deepfake Detection Report")

    # Timestamp
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prediction Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 110, f"Prediction Result: {prediction_label.upper()}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 130, f"Confidence: {confidence:.2f}%")

    y_position = height - 160

    # Prediction Explanation
    if prediction_explanation:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y_position, "Prediction Explanation:")
        y_position -= 18
        y_position = draw_wrapped_text(prediction_explanation, 60, y_position, 90)

    # Graph Explanation
    if graph_explanation:
        y_position -= 20
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y_position, "Graph Interpretation:")
        y_position -= 18
        y_position = draw_wrapped_text(graph_explanation, 60, y_position, 90)

    y_position -= 30  # Space before image section

    def draw_image(image_path, caption):
        nonlocal y_position
        if y_position < 200:
            c.showPage()
            y_position = height - 100
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, caption)
        y_position -= 20
        if os.path.exists(image_path):
            try:
                c.drawImage(image_path, 50, y_position - 150, width=200, height=150)
                y_position -= 170
            except Exception as e:
                c.drawString(50, y_position, f"[Error displaying image: {e}]")
                y_position -= 20

    # Add Grad-CAM Images (limit to 3)
    for i, img_path in enumerate(gradcam_images[:3]):
        draw_image(img_path, f"Grad-CAM Visualization {i + 1}")

    # Add Reference Frames (limit to 3)
    for i, img_path in enumerate(reference_images[:3]):
        draw_image(img_path, f"Reference Frame {i + 1}")

    # Add Graph
    if graph_image and os.path.exists(graph_image):
        draw_image(graph_image, "Prediction Confidence Graph")

    c.save()
