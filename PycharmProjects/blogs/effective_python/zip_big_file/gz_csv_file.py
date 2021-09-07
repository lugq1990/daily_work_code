import gzip
import shutil
import os
import time

# Noted: Change bellow parameters!
# Change this folder point to the folder of file
file_path = r"C:\Users\workings\202101"
# CSV file name
file_name = "iris_big.csv"
# Destination file name, endswith gz!
gz_file_name = "iris_gz_new_python.gz"


def gz_csv(file_path=file_path, file_name=file_name, gz_file_name=gz_file_name):
    start_time = time.time()

    if not gz_file_name.endswith('.gz'):
        gz_file_name += '.gz'

    with open(os.path.join(file_path, file_name), 'rb') as f_in:
        with gzip.open(os.path.join(file_path, gz_file_name), 'wb') as f_out:
            # This will try to copy every 16 KB file by default, and will only end if finished.
            # So it's OK for large size!
            shutil.copyfileobj(f_in, f_out)

    print("GZ process finished within: {} seconds.".format(round(time.time() - start_time, 2)))


if __name__ == "__main__":
    gz_csv()
    print("Finished!")



# gzip file unzip
import gzip
import shutil
import time

path = r"Downloads"
file_name = "Z_OH_WBS.gz"
start_time = time.time()
with gzip.open(os.path.join(path, file_name), 'rb') as f_in:
    with open(os.path.join(path, 'Z_file.csv'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("End with time: ", time.time() - start_time)