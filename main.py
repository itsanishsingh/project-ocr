from pdf_extraction import PDFTDataExtractor
from image_extraction import ImageTextExtractor, ImageTableExtractor


def pdf_extraction_process(file_path):
    extractor = PDFTDataExtractor(file_path)
    extractor.extract_save()
    text = extractor.page_data

    return text


def image_extraction_process(file_path):
    extractor = ImageTextExtractor(file_path)
    data = extractor.extract()

    table_extractor = ImageTableExtractor(file_path)
    data = table_extractor.extract()

    return data


if __name__ == "__main__":
    pdf_file_path = "data/xii_ch_1.pdf"
    image_file_path = "data/nutrition_table.jpg"
    # pdf_extraction_process(pdf_file_path)
    image_extraction_process(image_file_path)
