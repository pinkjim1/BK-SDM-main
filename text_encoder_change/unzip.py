import os
import tarfile
import json
import pickle




def extract_json_from_tar(tar_file_path, extract_to="temp"):
    json_files = []
    with tarfile.open(tar_file_path, 'r:*') as tar_ref:
        for member in tar_ref.getmembers():
            if member.isfile() and member.name.endswith('.json') and 'syn' in member.name:
                extracted_file = tar_ref.extractfile(member)
                if extracted_file is not None:
                    json_content = json.load(extracted_file)
                    json_files.append(json_content['syn_text'])
    return json_files



def create_list_from_tars(tar_folder, batch_size=32, shuffle=True):
    all_data = []
    for tar_file in os.listdir(tar_folder):
        if tar_file.endswith('.tar') :
            tar_path = os.path.join(tar_folder, tar_file)
            json_files = extract_json_from_tar(tar_path)
            all_data.extend(json_files)

    return all_data


# 示例用法
tar_folder = 'ml-mobileclip/DataCompDR-12M/datasets--apple--DataCompDR-12M/snapshots/bd23bbc361cc4b2ee0d2bd0431e107b2906b312f'  # tar文件所在的文件夹路径
data_list= create_list_from_tars(tar_folder)


with open('train_text.pkl', 'wb') as f:
    pickle.dump(data_list,f)