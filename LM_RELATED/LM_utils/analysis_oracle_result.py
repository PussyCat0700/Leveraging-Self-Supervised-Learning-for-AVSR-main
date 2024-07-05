from tqdm import tqdm
from collections import Counter
index_list=[]
path="/mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_result/decode_hybrid_oracle_bs48.txt"
path="/mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_VO_result/decode_hybird_oracle_48.txt"
with open(path,"r") as f:
    lines = f.readlines()
    for line in lines:
        line=line.strip()
        if "- oracle" in line:
            pos=line.find("oracle")
            index= line[pos+13:]
            index_list.append(index)

c = Counter(index_list)
print(len(index_list))
print(c)

"""
198
Counter({'1': 116, '2': 30, '3': 14, '4': 10, '6': 4, '7': 4, '44': 2, '29': 2, '10': 2, '19': 2, '5': 2, '77': 2, '15': 2, '9': 2, '11': 2, '12': 2})
"""
# 77 29 44

"""
636
Counter({'1': 148, '2': 95, '3': 65, '4': 52, '5': 26, '6': 25, '8': 18, '7': 15, '15': 12, '10': 10, '9': 9, '14': 8, '11': 8, '12': 8, '16': 7, '13': 6, '17': 6, '26': 5, '19': 5, '25': 5, '57': 4, '21': 4, '69': 4, '23': 4, '42': 4, '39': 4, '35': 3, '32': 3, '18': 3, '24': 3, '29': 3, '68': 3, '56': 3, '36': 2, '22': 2, '89': 2, '86': 2, '37': 2, '34': 2, '41': 2, '27': 2, '47': 2, '43': 2, '20': 2, '80': 2, '66': 2, '60': 2, '71': 2, '31': 2, '30': 2, '77': 1, '59': 1, '33': 1, '53': 1, '63': 1, '50': 1, '51': 1, '40': 1, '83': 1, '85': 1, '55': 1, '81': 1, '28': 1, '73': 1, '64': 1, '62': 1, '91': 1, '98': 1, '75': 1, '70': 1, '45': 1, '48': 1, '78': 1, '44': 1})
"""