import cv2
from pytesseract import image_to_string
import numpy as np
import subprocess


class ImageTextExtractor:
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
        _, bw = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
        return bw

    @staticmethod
    def convert_black_white_otsu(image):
        _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
        return bw

    @staticmethod
    def invert_image(image):
        invert_image = cv2.bitwise_not(image)
        return invert_image

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
        if not cntSorted:
            return image
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
        with open(path + "text.txt", "w") as f:
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


class ImageTableExtractor:
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
        _, bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return bw

    @staticmethod
    def convert_black_white_otsu(image):
        _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw

    @staticmethod
    def invert_image(image):
        invert_image = cv2.bitwise_not(image)
        return invert_image

    @staticmethod
    def dilate_image(image):
        dilated_image = cv2.dilate(image, None, iterations=5)
        return dilated_image

    @staticmethod
    def find_all_contours(image, rect_only=False):
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not rect_only:
            return contours

        rectangular_contours = []

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rectangular_contours.append(contour)

        return rectangular_contours

    @staticmethod
    def biggest_contour(contours):
        max_area = 0
        contour_with_max_area = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                contour_with_max_area = contour

        return contour_with_max_area

    @staticmethod
    def order_points(pts):
        epsilon = 0.02 * cv2.arcLength(pts, True)
        pts = cv2.approxPolyDP(pts, epsilon, True)
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def calculateDistanceBetween2Points(p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def extracting_table(self, contour_with_max_area):
        contour_with_max_area_ordered = self.order_points(contour_with_max_area)
        existing_image_width = self.original_image.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)

        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(
            contour_with_max_area_ordered[0], contour_with_max_area_ordered[1]
        )
        distance_between_top_left_and_bottom_left = (
            self.calculateDistanceBetween2Points(
                contour_with_max_area_ordered[0], contour_with_max_area_ordered[3]
            )
        )
        aspect_ratio = (
            distance_between_top_left_and_bottom_left
            / distance_between_top_left_and_top_right
        )
        new_image_width = existing_image_width_reduced_by_10_percent
        new_image_height = int(new_image_width * aspect_ratio)

        pts1 = np.float32(contour_with_max_area_ordered)
        pts2 = np.float32(
            [
                [0, 0],
                [new_image_width, 0],
                [new_image_width, new_image_height],
                [0, new_image_height],
            ]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_corrected_image = cv2.warpPerspective(
            self.original_image, matrix, (new_image_width, new_image_height)
        )

        image_height = self.original_image.shape[0]
        padding = int(image_height * 0.1)
        perspective_corrected_image = cv2.copyMakeBorder(
            perspective_corrected_image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        return perspective_corrected_image

    @staticmethod
    def extracting_structure(image):
        hor = np.array([[1, 1, 1, 1, 1, 1]])
        vertical_lines_eroded_image = cv2.erode(image, hor, iterations=10)
        vertical_lines_eroded_image = cv2.dilate(
            vertical_lines_eroded_image, hor, iterations=10
        )

        ver = np.array([[1], [1], [1], [1], [1], [1], [1]])
        horizontal_lines_eroded_image = cv2.erode(image, ver, iterations=10)
        horizontal_lines_eroded_image = cv2.dilate(
            horizontal_lines_eroded_image, ver, iterations=10
        )

        combined_image = cv2.add(
            vertical_lines_eroded_image, horizontal_lines_eroded_image
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=5)
        image_without_lines = cv2.subtract(image, combined_image_dilated)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_without_lines_noise_removed = cv2.erode(
            image_without_lines, kernel, iterations=1
        )
        image_without_lines_noise_removed = cv2.dilate(
            image_without_lines_noise_removed, kernel, iterations=1
        )

        return image_without_lines_noise_removed

    @staticmethod
    def dilating_inside_structure(image):
        kernel_to_remove_gaps_between_words = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )
        dilated_image = cv2.dilate(
            image, kernel_to_remove_gaps_between_words, iterations=5
        )
        simple_kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(dilated_image, simple_kernel, iterations=2)

        return dilated_image

    @staticmethod
    def extracting_text_from_table(contours):
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        heights = []
        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)

        mean_height = np.mean(heights)

        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

        rows = []
        half_of_mean_height = mean_height / 2
        current_row = [bounding_boxes[0]]
        for bounding_box in bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(
                current_bounding_box_y - previous_bounding_box_y
            )
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                rows.append(current_row)
                current_row = [bounding_box]
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda x: x[0])

        return rows

    @staticmethod
    def get_result_from_tersseract(image_path):
        output = subprocess.getoutput(
            "tesseract "
            + image_path
            + ' - -l eng --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "'
        )
        output = output.strip()
        return output

    def read_rows(self, rows, table_image):
        table = []
        current_row = []
        image_number = 0
        for row in rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5
                cropped_image = table_image[y : y + h, x : x + w]
                image_slice_path = (
                    self.save_path + "table/ocr_slices/" + str(image_number) + ".jpg"
                )
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = self.get_result_from_tersseract(image_slice_path)
                current_row.append(results_from_ocr)
                image_number += 1
            table.append(current_row)
            current_row = []

        return table

    def generate_csv_file(self, table):
        with open(self.final_path + "table_data.csv", "w") as f:
            for row in table:
                f.write(",".join(row) + "\n")

    def debug_save(self, image, name):
        temp_image = self.original_image.copy()
        cv2.drawContours(temp_image, image, -1, (0, 255, 0), 3)
        cv2.imwrite(self.save_path + f"table/{name}.jpg", temp_image)

    def extract(self):
        # Gray-scaling
        gray_scaled = self.convert_gray_scale(self.original_image)
        cv2.imwrite(self.save_path + "table/01_gray.jpg", gray_scaled)

        # Black and White
        black_white = self.convert_black_white_otsu(gray_scaled)
        cv2.imwrite(self.save_path + "table/02_bw.jpg", black_white)

        # Inverted Image
        inverted_image = self.invert_image(black_white)
        cv2.imwrite(self.save_path + "table/03_inv.jpg", inverted_image)

        # Dilating the image
        dilated_image = self.dilate_image(inverted_image)
        cv2.imwrite(self.save_path + "table/04_dil.jpg", dilated_image)

        # Finding all rectangular contours
        contours = self.find_all_contours(dilated_image, rect_only=True)

        # For debbuging storing image of all contours
        self.debug_save(contours, "05_all_contours")

        # The biggest contour
        contour_with_max_area = self.biggest_contour(contours=contours)

        # For debbuging storing image
        self.debug_save([contour_with_max_area], "06_big_contours")

        # Extracting the table
        table_image = self.extracting_table(contour_with_max_area)

        # For debugging
        cv2.imwrite(self.save_path + "table/07_extracted_table.jpg", table_image)

        # Extracting structure

        # Gray-scaling
        gray_scaled = self.convert_gray_scale(table_image)
        cv2.imwrite(self.save_path + "table/08_gray.jpg", gray_scaled)

        # Black and White
        black_white = self.convert_black_white(gray_scaled)
        cv2.imwrite(self.save_path + "table/09_bw.jpg", black_white)

        # Inverted Image
        inverted_image = self.invert_image(black_white)
        cv2.imwrite(self.save_path + "table/10_inv.jpg", inverted_image)

        # Extracting structure
        image_without_structure = self.extracting_structure(inverted_image)

        # For debugging
        cv2.imwrite(self.save_path + "table/11_wo_struct.jpg", image_without_structure)

        # Extracting texts
        # Black and White
        black_white = self.convert_black_white(image_without_structure)
        cv2.imwrite(self.save_path + "table/12_bw.jpg", black_white)

        # Dilating each pixel
        dilated_image = self.dilating_inside_structure(black_white)

        # Contours within the structure
        contours = self.find_all_contours(dilated_image, rect_only=False)

        # For debugging
        cv2.drawContours(table_image, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(self.save_path + f"table/13_with_text-contours.jpg", table_image)

        rows = self.extracting_text_from_table(contours)

        table = self.read_rows(rows, table_image)

        self.generate_csv_file(table)
