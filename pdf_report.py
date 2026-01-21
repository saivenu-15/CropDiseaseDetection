from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os

def generate_pdf(report_data, output_path="disease_report.pdf"):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Crop Disease Detection Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    for key, value in report_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 25

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 100, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    c.save()
    return output_path
