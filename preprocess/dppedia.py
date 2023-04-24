# (1) get every label, may be fine grained or not
# (2) process the xml file
# (3) construct dataset

import os
import pandas
import tqdm
import json

ROOT_DIR = 'dbpedia_csv'

num2class = {
    1: 'Company',
    2: 'EducationalInstitution',
    3: 'Artist',
    4: 'Athlete',
    5: 'OfficeHolder',
    6: 'MeanOfTransportation',
    7: 'Building',
    8: 'NaturalPlace',
    9: 'Village',
    10: 'Animal',
    11: 'Plant',
    12: 'Album',
    13: 'Film',
    14: 'WrittenWork',
}


def get_dataset(split, ROOT_DIR=ROOT_DIR):
    dataset = pandas.read_csv(os.path.join(
        ROOT_DIR, split + '.csv'), header=None, index_col=None).fillna('')
    dataset_csv = []
    for d in dataset:
        dataset_csv.append(dataset[d])
    dataset_csv = [dataset_csv[d] for d in range(len(dataset_csv))]
    answer2context = {}
    cnt = 0
    total = len(dataset_csv[0])
    with open(ROOT_DIR + '/' + split + '.jsonl', 'w', encoding='utf-8') as target_file:
        for label, c1, c2 in zip(dataset_csv[0],
                                 dataset_csv[1],
                                 dataset_csv[2]):
            datapoint = {}
            datapoint["question"] = " ".join(list(num2class.values()))
            context = ' '.join([c1, c2]).replace('\n', ' ').replace(
                '\"', ' ').replace('\\n', ' ').replace('\\', ' ')
            context = ' '.join(context.split())
            datapoint["context"] = context
            datapoint["answer"] = num2class[label]
            json.dump(datapoint, target_file)
            print(file=target_file)
            cnt += 1
            print('\r %.2f (%d / %d)' % (cnt / total, cnt, total), end='')


if __name__ == "__main__":
    print('getting test dataset...')
    get_dataset('test')
    print('\ndone getting test set \n')

    print('getting train dataset...')
    get_dataset('train')
    print('\ndone getting train set \n')

    with open(ROOT_DIR + '/question.jsonl', 'w', encoding='utf-8') as file:
        print({v.lower(): k - 1 for k, v in num2class.items()}, file=file)
