import kagglehub

# Download APTOS 2019 dataset
path = kagglehub.dataset_download("mariaherrerot/aptos2019")

print("Path to dataset files:", path)