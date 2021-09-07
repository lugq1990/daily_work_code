import zipfile

file_name = "./main.py"

with zipfile.ZipFile("./index.zip", 'w') as f:
    f.write(file_name)

print("finished zip!")