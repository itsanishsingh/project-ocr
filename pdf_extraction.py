import pymupdf


class PDFTDataExtractor:
    def __init__(self, file_path):
        self.pdf = pymupdf.open(file_path)
        self.save_path = "data/output/"

    def extract_save(self):
        self.extract_text()
        self.save_text()
        self.extract_image()
        self.save_image()

    def extract_text(self):
        self.page_data = {page.number: page.get_text() for page in self.pdf}

    def extract_image(self):
        self.images = []
        for page in self.pdf:
            for im_ind, im in enumerate(page.get_images(), start=1):
                xref = im[0]
                base_image = self.pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                im_data = (image_bytes, image_ext, page.number + 1, im_ind)
                self.images.append(im_data)

    def save_text(self):
        try:
            relative_path = self.save_path
            for page_num, page_text in self.page_data.items():
                with open(relative_path + f"page_{page_num+1}.txt", "w") as f:
                    f.write(page_text)
        except AttributeError:
            self.extract_text()
            self.save_text()

    def save_image(self):
        try:
            relative_path = self.save_path
            for image_bytes, image_ext, page_number, im_ind in self.images:
                with open(
                    f"{relative_path}image{page_number}_{im_ind}.{image_ext}",
                    "wb",
                ) as image:
                    image.write(image_bytes)
        except AttributeError:
            self.extract_image()
            self.save_image()
