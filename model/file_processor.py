import os
import numpy as np
from PIL import Image
import subprocess
import olefile
import openpyxl

# Step 1: Define directories
INPUT_DIR = "data"  # Directory where input files are stored
OUTPUT_DIR = "data"  # Directory to save bytecode images
EXE_OUTPUT_DIR = "converted_exes"  # Temporary directory for converted .exe files
SUPPORTED_EXTENSIONS = [".exe", ".bat", ".xlsx", ".xls", ".doc", ".docx"]

# Step 2: Create necessary directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXE_OUTPUT_DIR, exist_ok=True)

# Step 3: Function to convert .bat files to .exe using a command-line tool
def convert_bat_to_exe(bat_file_path, exe_output_dir):
    bat_file_name = os.path.basename(bat_file_path)
    exe_file_name = os.path.splitext(bat_file_name)[0] + ".exe"
    exe_file_path = os.path.join(exe_output_dir, exe_file_name)
    
    # Replace `bat_to_exe_tool_path` with the actual path to your Bat To Exe Converter CLI tool
    bat_to_exe_tool_path = "path_to_bat_to_exe_converter_cli.exe"
    command = [
        bat_to_exe_tool_path,
        bat_file_path,  # Input .bat file
        "/target", exe_file_path,  # Output .exe file
        "/silent"  # Optional: Run the converter in silent mode
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Converted {bat_file_name} to {exe_file_name}")
        return exe_file_path
    except Exception as e:
        print(f"Error converting {bat_file_name} to .exe: {e}")
        return None

# Step 4: Function to convert binary data into a bytecode image
def convert_to_bytecode_image(file_path, output_dir):
    # Read the binary data of the file
    with open(file_path, "rb") as f:
        binary_data = f.read()
    
    # Convert binary data to a NumPy array of integers (0–255)
    byte_array = np.array(list(binary_data), dtype=np.uint8)
    
    # Calculate dimensions for a square-ish image
    size = int(np.ceil(np.sqrt(len(byte_array))))
    padded_size = size * size
    
    # Pad the byte array to fit the square dimensions
    padded_array = np.zeros(padded_size, dtype=np.uint8)
    padded_array[:len(byte_array)] = byte_array
    
    # Reshape the array into a 2D grid
    image_array = padded_array.reshape((size, size))
    
    # Normalize the pixel values to 0–255 range
    image_array = np.uint8(image_array)
    
    # Create an image from the array
    image = Image.fromarray(image_array, mode="L")  # "L" mode for grayscale
    
    # Save the image
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
    image.save(output_path)
    print(f"Saved bytecode image: {output_path}")

# Step 5: Function to extract OLE streams from .xls files
def extract_ole_streams(file_path):
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

# Step 6: Function to convert .xlsx files to byte streams
def xlsx_to_ole_bytes(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    streams = []
    for sheet in wb.worksheets:
        content = ""
        for row in sheet.iter_rows(values_only=True):
            row_str = ",".join([str(cell) if cell is not None else "" for cell in row])
            content += row_str + "\n"
        streams.append(content.encode("utf-8"))
    return streams

# Step 7: Function to process Excel files (.xls, .xlsx)
def process_excel_file(file_path, output_dir):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xls":
        streams = extract_ole_streams(file_path)
    elif ext == ".xlsx":
        streams = xlsx_to_ole_bytes(file_path)
    
    saved_images = []
    for i, stream in enumerate(streams):
        byte_array = np.frombuffer(stream, dtype=np.uint8)
        size = int(np.ceil(np.sqrt(len(byte_array))))
        padded_size = size * size
        
        padded_array = np.zeros(padded_size, dtype=np.uint8)
        padded_array[:len(byte_array)] = byte_array
        
        image_array = padded_array.reshape((size, size))
        image_array = np.uint8(image_array)
        
        image = Image.fromarray(image_array, mode="L")
        output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_stream_{i}.png")
        image.save(output_path)
        saved_images.append(output_path)
    
    return saved_images

# Step 8: Main function to process all files
def process_files(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {file_name}")
            continue
        
        print(f"Processing file: {file_name}")
        
        if ext == ".bat":
            exe_file_path = convert_bat_to_exe(file_path, EXE_OUTPUT_DIR)
            if exe_file_path:
                convert_to_bytecode_image(exe_file_path, output_dir)
        elif ext in [".exe", ".doc", ".docx"]:
            convert_to_bytecode_image(file_path, output_dir)
        elif ext in [".xls", ".xlsx"]:
            process_excel_file(file_path, output_dir)
        else:
            print(f"Skipping unsupported file: {file_name}")

if __name__ == "__main__":
    process_files(INPUT_DIR, OUTPUT_DIR)
    print("All files processed successfully!")