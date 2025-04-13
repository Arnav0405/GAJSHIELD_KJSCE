import os
import olefile
from PIL import Image
# import io
import openpyxl


class ExcelToImageConverter:
    def __init__(self, output_folder="./data/images"):
        """
        Initialize the converter with an optional output folder.
        """
        self.output_folder = output_folder
        self.supported_extensions = [".xls", ".xlsx"]

    def extract_ole_streams(self, file_path):
        """
        Extract OLE streams from an .xls file.
        Returns a list of byte streams.
        """
        streams = []
        if olefile.isOleFile(file_path):
            ole = olefile.OleFileIO(file_path)
            for stream_name in ole.listdir():
                try:
                    stream = ole.openstream(stream_name)
                    data = stream.read()
                    streams.append(data)
                except Exception:
                    pass  # Ignore streams that cannot be read
            ole.close()
        return streams

    def xlsx_to_ole_bytes(self, file_path):
        """
        Convert an .xlsx file to byte streams by reading its content.
        Returns a list of byte streams.
        """
        wb = openpyxl.load_workbook(file_path, data_only=True)
        streams = []
        for sheet in wb.worksheets:
            content = ""
            for row in sheet.iter_rows(values_only=True):
                row_str = ",".join([str(cell) if cell is not None else "" for cell in row])
                content += row_str + "\n"
            streams.append(content.encode("utf-8"))
        return streams

    def bytecode_to_image(self, byte_data, image_size=(256, 256)):
        """
        Convert byte data into a grayscale image of the specified size.
        Returns a PIL Image object.
        """
        byte_data = byte_data[:image_size[0] * image_size[1]]
        padded = byte_data + bytes([0] * (image_size[0] * image_size[1] - len(byte_data)))
        img = Image.frombytes('L', image_size, bytes(padded))
        return img

    def convert_excel_to_images(self, file_path):
        """
        Convert an Excel file (.xls or .xlsx) into images.
        Saves the images to the specified output folder.
        Returns the list of saved image file paths.
        """
        os.makedirs(self.output_folder, exist_ok=True)

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")

        # Extract streams based on file type
        if ext == ".xls":
            streams = self.extract_ole_streams(file_path)
        elif ext == ".xlsx":
            streams = self.xlsx_to_ole_bytes(file_path)

        # Convert each stream to an image and save it
        saved_images = []
        for i, stream in enumerate(streams):
            img = self.bytecode_to_image(stream)
            image_path = os.path.join(self.output_folder, f"stream_{i}.png")
            img.save(image_path)
            saved_images.append(image_path)

        return saved_images
