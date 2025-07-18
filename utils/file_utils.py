import os

def check_file_exists(filepath):
    if not os.path.isfile(filepath):
        print(f"Error: File not found : {filepath}")
        exit(1)

def getSmallestAvailableNumber():
    folder = "output"
    prefix = "output_"

    os.makedirs(folder, exist_ok=True)

    existing_files = os.listdir(folder)
    used_numbers = set()

    for filename in existing_files:
        if filename.startswith(prefix) and filename.endswith(".png"):
            try:
                num = int(filename[len(prefix):-4])
                used_numbers.add(num)
            except ValueError:
                continue

    i = 0
    while i in used_numbers:
        i += 1

    return i