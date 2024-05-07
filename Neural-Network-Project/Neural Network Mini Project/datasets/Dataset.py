import wget
import os
import gzip, zipfile
import shutil

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self, url, data_dir, file_name):
        data_path = os.path.join(data_dir, file_name)
        if not os.path.exists(data_path):
            print("Dowloading from url: ", url)
            print("Saving to directory: ", data_dir)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            wget.download(url, out=data_path)
            print("Download complete.\n")
        else:
            print("Skipping download. File already exists: {}\n".format(data_path))
        return data_path

    
    def extract_gz(self, input_file_path, output_file_path):
        if not os.path.exists(output_file_path):
            print("Extracting: ", input_file_path)
            with gzip.open(input_file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        else:
            print("Skipping extraction. File already exists: {}\n".format(output_file_path))

    
    def extract_zip(self, file_path, location='.'):
        print('Unzipping: '+ file_path + '\n')
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(path=location)
        zip_ref.close()