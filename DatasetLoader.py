# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:51:07 2024

@author: joris
"""

import os
import json
import pandas as pd

def get_all_files(root_dir):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def add_file(file_path, dataset):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    #REMOVE THE STR AND REPLACE ARUOND METADATA ONCE WE HAVE NEW DATASET
    new_row = {'id': data['id'], 'content': data['inhoud'], 'content_type': 'text', 'meta': str(data['metadata']).replace('[', '').replace(']', ''), 'id_hash_keys': "['content']", 'score': "None", 'embedding': "None"}
    dataset.loc[len(dataset)] = new_row
    return dataset

def get_dataset(root_folder):
    dataset = pd.DataFrame(columns=['id', 'content', 'content_type', 'meta', 'id_hash_keys', 'score', 'embedding'])
    all_files = get_all_files('articles')

    for file_path in all_files:
        dataset = add_file(file_path, dataset)

    return dataset


