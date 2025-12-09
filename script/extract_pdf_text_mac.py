import pdfplumber
from pathlib import Path
import argparse
import re
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance

"""
============================= REQUIRED DEPENDENCIES =============================

This script requires the following external tools to be installed on the system:

1) TESSERACT OCR (for image-based PDF text extraction)
   Download: https://github.com/UB-Mannheim/tesseract/wiki
   Default Windows Path:
   C:\Program Files\Tesseract-OCR\tesseract.exe

2) POPPLER (for converting PDF pages to images)
   Download: https://github.com/oschwartz10612/poppler-windows/releases/
   Example Path Used In This Script:
   C:\poppler-25.11.0\Library\bin

3) PYTHON LIBRARIES (install using pip):
   pip install pdfplumber pdf2image pytesseract pillow

===============================================================================
"""

# Set the absolute path to the Tesseract OCR executable
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def fix_duplicated_text(text: str) -> str:
    """
    Removes duplicated repeated characters caused by OCR noise.
    Example:
        "Heeelllooo" -> "Helo"
    """
    return re.sub(r"(.)\1+", r"\1", text)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocesses images before OCR for better accuracy:
    - Converts to grayscale
    - Applies binary thresholding
    - Increases contrast
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 140 else 255)  # Binary threshold
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)  # Increase contrast
    return image


def extract_text_with_pdfplumber(pdf_path: str) -> dict:
    """
    Extracts text from a text-based PDF using pdfplumber.
    Returns a dictionary:
        {
            "page_1": "text...",
            "page_2": "text..."
        }
    """
    pages_text = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True)

            if raw_text:
                clean_text = fix_duplicated_text(raw_text)
                pages_text[f"page_{i}"] = clean_text.strip()
            else:
                pages_text[f"page_{i}"] = ""

    return pages_text


def extract_text_with_ocr(pdf_path: str) -> dict:
    """
    Extracts text from image-based PDFs using OCR (Tesseract).
    Converts each page to an image before extracting text.
    """
    pages_text = {}

    images = convert_from_path(
        pdf_path,
        dpi=300
       # poppler_path=r"C:\poppler-25.11.0\Library\bin"
    )

    # Save one debug image for verification if needed
    images[1].save("debug_page2.png")

    for i, image in enumerate(images, start=1):
        processed_image = preprocess_image(image)

        custom_config = r"--oem 3 --psm 4"

        text = pytesseract.image_to_string(
            processed_image,
            lang="eng",
            config=custom_config
        )

        pages_text[f"page_{i}"] = text.strip()

    return pages_text


def extract_text_from_pdf_smart(pdf_path: str) -> dict:
    """
    Smart extractor:
    - First tries text-based extraction using pdfplumber.
    - If extracted text is too small, switches automatically to OCR.
    """
    result = extract_text_with_pdfplumber(pdf_path)
    all_text = "".join(result.values()).strip()

    if len(all_text) < 300:
        print("⚠️ Very little text found. Switching to OCR...")
        result = extract_text_with_ocr(pdf_path)
    else:
        print("✅ Text extracted using pdfplumber.")

    return result


def save_to_txt(output_path: str, text_dict: dict):
    """
    Saves extracted text into a formatted TXT file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for page, text in text_dict.items():
            f.write(f"{'='*40}\n")
            f.write(f"{page}\n")
            f.write(f"{'-'*40}\n")
            f.write(text + "\n\n")


if __name__ == "__main__":
    """
    ============================= USAGE EXAMPLE =============================

    python extract_pdf_text.py input.pdf output.txt

    =========================================================================
    """

    parser = argparse.ArgumentParser(description="Smart PDF to TXT Extractor (pdfplumber + OCR fallback)")

    parser.add_argument("pdf_file", help="Path to the input PDF file")
    parser.add_argument("output_txt", help="Path to the output TXT file")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_file).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ File not found: {pdf_path}")

    result = extract_text_from_pdf_smart(str(pdf_path))
    save_to_txt(args.output_txt, result)

    print(f"\n✅ Text successfully saved to: {args.output_txt}")
