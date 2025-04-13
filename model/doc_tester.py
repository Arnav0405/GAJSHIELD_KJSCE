import os
import numpy as np
from PIL import Image
import subprocess

# Step 1: Define directories
INPUT_DIR = "downloads"  # Directory where .bat, .exe, and .ole files are stored
OUTPUT_DIR = "bytecode_images"  # Directory to save the bytecode images
EXE_OUTPUT_DIR = "converted_exes"  # Temporary directory to store converted .exe files

# Step 2: Create necessary directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXE_OUTPUT_DIR, exist_ok=True)

# Step 3: Function to convert .bat files to .exe using a command-line tool
def convert_bat_to_exe(bat_file_path, exe_output_dir):
    bat_file_name = os.path.basename(bat_file_path)
    exe_file_name = os.path.splitext(bat_file_name)[0] + ".exe"
    exe_file_path = os.path.join(exe_output_dir, exe_file_name)
    
    # Use a command-line tool (e.g., Bat To Exe Converter) to convert .bat to .exe
    # Replace bat_to_exe_tool_path with the path to your Bat To Exe Converter CLI tool
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

# Step 4: Function to convert a binary file into a bytecode image
def convert_to_bytecode_image(file_path, output_dir):
    # Read the binary data of the file
    with open(file_path, "rb") as f:
        binary_data = f.read()
    
    # Convert binary data to a NumPy array of integers (0–255)
    byte_array = np.array(list(binary_data), dtype=np.uint8)
    
    # Calculate the dimensions for the image (square or rectangular)
    size = int(np.ceil(np.sqrt(len(byte_array))))  # Square root of the number of bytes
    padded_size = size * size  # Total pixels in the square image
    
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

# Step 5: Process all files in the input directory
for file_name in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, file_name)
    
    # Check if the file is a .bat file
    if file_name.endswith(".bat"):
        print(f"Processing .bat file: {file_name}")
        # Convert .bat to .exe
        exe_file_path = convert_bat_to_exe(file_path, EXE_OUTPUT_DIR)
        if exe_file_path:
            # Generate bytecode image for the converted .exe file
            convert_to_bytecode_image(exe_file_path, OUTPUT_DIR)
    
    # Check if the file is already a .exe or .ole file
    elif file_name.endswith(".exe") or file_name.endswith(".ole"):
        print(f"Processing file: {file_name}")
        try:
            convert_to_bytecode_image(file_path, OUTPUT_DIR)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print("All files processed successfully!")