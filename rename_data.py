import os

# define the name of our data folder
data_folder = 'TOAD_toy_data'

# our list of cancer types (subfolders)
cancer_types = ['BRCA', 'COAD', 'CCRCC', 'LUAD']

# loop through each cancer type folder
for cancer_type in cancer_types:
    # get the full path to the current cancer type folder
    cancer_folder = os.path.join(data_folder, cancer_type)
    
    if os.path.exists(cancer_folder):
        # list all files in folder
        files = os.listdir(cancer_folder)
        
        # filter out only the .svs files
        svs_files = [f for f in files if f.endswith('.svs')]
        
        # loop through the .svs files and rename them
        for i, filename in enumerate(svs_files, start=1):
            # generate the new file name
            new_name = f"{cancer_type}_{i}.svs"
            
            # get the full path to the original and new file names
            original_path = os.path.join(cancer_folder, filename)
            new_path = os.path.join(cancer_folder, new_name)
            
            # rename the file
            os.rename(original_path, new_path)
            print(f"Renamed {filename} to {new_name}")
    else:
        print(f"Folder {cancer_folder} does not exist")

