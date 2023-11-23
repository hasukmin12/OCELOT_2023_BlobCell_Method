import json

import numpy as np

with open('/vast/AI_team/sukmin/Results_Stroma_NDM/Test_200x_CacoX_b32_e10k/summary.json','r') as my_json:
    my_code = json.load(my_json)




N_case = [] # 0 ~ 27
D_case = [] # 28 ~ 33 6개
M_case = [] # 34 ~ 37 4개
NET_case = [] # 37 ~  5개

for c in range(43):
    if c<=27:
        N_case.append(c)
    elif 27<c<=33:
        D_case.append(c)
    elif 33<c<=37:
        M_case.append(c)
    else:
        NET_case.append(c)


N_Precision = []
N_Recall = []
N_Dice_Score = []

D_Precision = []
D_Recall = []
D_Dice_Score = []

M_Precision = []
M_Recall = []
M_Dice_Score = []

NET_Precision = []
NET_Recall = []
NET_Dice_Score = []

for i in N_case:
    precision = my_code['results']['all'][i]['2']['Precision']
    N_Precision.append(precision)
    rec = my_code['results']['all'][i]['2']['Recall']
    N_Recall.append(rec)
    dice = my_code['results']['all'][i]['2']['Dice']
    N_Dice_Score.append(dice)


for i in D_case:
    precision = my_code['results']['all'][i]['2']['Precision']
    D_Precision.append(precision)
    rec = my_code['results']['all'][i]['2']['Recall']
    D_Recall.append(rec)
    dice = my_code['results']['all'][i]['2']['Dice']
    D_Dice_Score.append(dice)


for i in M_case:
    precision = my_code['results']['all'][i]['2']['Precision']
    M_Precision.append(precision)
    rec = my_code['results']['all'][i]['2']['Recall']
    M_Recall.append(rec)
    dice = my_code['results']['all'][i]['2']['Dice']
    M_Dice_Score.append(dice)


for i in NET_case:
    precision = my_code['results']['all'][i]['2']['Precision']
    NET_Precision.append(precision)
    rec = my_code['results']['all'][i]['2']['Recall']
    NET_Recall.append(rec)
    dice = my_code['results']['all'][i]['2']['Dice']
    NET_Dice_Score.append(dice)



print()
print("N_Dice_score_average : ", np.average(N_Dice_Score))
print("N_Precision_average : ", np.average(N_Precision))
print("N_Recall_average : ", np.average(N_Recall))
print()
print("D_Dice_score_average : ", np.average(D_Dice_Score))
print("D_Precision_average : ", np.average(D_Precision))
print("D_Recall_average : ", np.average(D_Recall))
print()
print("M_Dice_score_average : ", np.average(M_Dice_Score))
print("M_Precision_average : ", np.average(M_Precision))
print("M_Recall_average : ", np.average(M_Recall))
print()
print("NET_Dice_score_average : ", np.average(NET_Dice_Score))
print("NET_Precision_average : ", np.average(NET_Precision))
print("NET_Recall_average : ", np.average(NET_Recall))
print()