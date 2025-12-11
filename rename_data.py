import os
import shutil

# Define the root folder containing the cancer type folders
root_folder = 'TOAD_dataset_testing'

# List of cancer types (folders)
cancer_types = ['BRCA', 'COAD', 'CCRCC', 'LUAD']

# Loop through each cancer type folder
for cancer_type in cancer_types:
    # Get the full path to the current cancer type folder
    cancer_folder = os.path.join(root_folder, cancer_type)
    
    # Check if the folder exists
    if os.path.exists(cancer_folder):
        # List all files in the folder
        files = os.listdir(cancer_folder)
        
        # Filter out only the .svs files
        svs_files = [f for f in files if f.endswith('.svs')]
        
        # Loop through the .svs files and rename them
        for i, filename in enumerate(svs_files, start=1):
            # Generate the new file name
            new_name = f"{cancer_type}_{i}.svs"
            
            # Get the full path to the original and new file names
            original_path = os.path.join(cancer_folder, filename)
            new_path = os.path.join(cancer_folder, new_name)
            
            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed {filename} to {new_name}")
            
            # Now move the renamed file to the main root folder
            new_location = os.path.join(root_folder, new_name)
            shutil.move(new_path, new_location)
            print(f"Moved {new_name} to {root_folder}")
    else:
        print(f"Folder {cancer_folder} does not exist")