import cv2
from pytesseract import image_to_string
import numpy as np
import subprocess


class ImageDataExtractor:
    def __init__(self, file_path):
        self.original_image = cv2.imread(file_path)
        self.save_path = "data/image_output/"
        self.final_path = "data/output/"

    @staticmethod
    def convert_gray_scale(image):
        gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_scaled

    @staticmethod
    def convert_black_white(image):
        _, bw = cv2.threshold(image, 210, 230, cv2.THRESH_BINARY)
        return bw

    @staticmethod
    def remove_noise(image):
        kernel_dil = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel_dil, iterations=1)
        kernel_ero = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel_ero, iterations=1)
        kernel_morph = np.ones((1, 1), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_morph)
        image = cv2.medianBlur(image, 3)
        return image

    @staticmethod
    def erode(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    @staticmethod
    def dilate(image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    @staticmethod
    def remove_borders(image):
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cntSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cntSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y : y + h, x : x + w]
        return crop

    @staticmethod
    def extract_text(image):
        data = image_to_string(image)
        return data

    @staticmethod
    def save_text(text, path):
        with open(path + "result.txt", "w") as f:
            f.write(text)

    def extract(self):
        gray_scaled = self.convert_gray_scale(self.original_image)
        cv2.imwrite(self.save_path + "text/01_gray.jpg", gray_scaled)

        black_white = self.convert_black_white(gray_scaled)
        cv2.imwrite(self.save_path + "text/02_bw.jpg", black_white)

        noise_removed = self.remove_noise(black_white)
        cv2.imwrite(self.save_path + "text/03_no_noise.jpg", noise_removed)

        dilated_image = self.dilate(noise_removed)
        cv2.imwrite(self.save_path + "text/04_dilated_image.jpg", dilated_image)

        no_border_image = self.remove_borders(dilated_image)
        cv2.imwrite(self.save_path + "text/05_no_border_image.jpg", no_border_image)

        text = self.extract_text(no_border_image)
        self.save_text(text, self.final_path)

        return text
