import os
import json
join = os.path.join
import numpy as np


# NaN을 제거하고 수치를 측정해보자!
path = "/vast/AI_team/sukmin/Results_Test_Lunit_Challenge_for_paper/val/tissue"
path_list = sorted(next(os.walk(path))[1])

for case in path_list:
    print()
    print(case)
    case_list = sorted(next(os.walk(join(path, case)))[2])
    with open(join(path, case, 'summary.json'), 'r') as my_json:
        my_code = json.load(my_json)

    dice_score_1 = []
    dice_score_2 = []
    
    for i in range(len(case_list)-1):# 60):
        dice_1 = my_code['results']['all'][i]['1']['Dice']
        if np.isnan(dice_1) == False:
            dice_score_1.append(dice_1)

        dice_2 = my_code['results']['all'][i]['2']['Dice']
        if np.isnan(dice_2) == False:
            dice_score_2.append(dice_2)

        # pred = my_code['results']['all'][i]['1']['Precision']
        # pred_score_easy.append(pred)
        # jac = my_code['results']['all'][i]['1']['Jaccard']
        # jac_score_easy.append(jac)
        # rec = my_code['results']['all'][i]['1']['Recall']
        # recal_score_easy.append(rec)

    print("Cancer Dice score : ", np.mean(dice_score_1))
    # print("TC Dice score : ", np.mean(dice_score_2))

    rst = dice_score_1 + dice_score_2
    # print("Mean Dice score : ", (np.mean(dice_score_1) + np.mean(dice_score_2))/2)
    # print("Mean Dice score : ", (np.mean(rst)))





# result = 0
# for val in pred_score:
#     result += val
# print("total_precision_average : ", result/len(pred_score))
#
# for val in recal_score:
#     result += val
# print("total_recal_average : ", result/len(pred_score))
#
# result = 0
# for val in jac_score:
#     result += val
# print("total_jac_average : ", result/len(pred_score))








# # draw plot
# import matplotlib.pyplot as plt
# plt.figure()
#
# # print(int)
# # print(int/22)
# # weight.sort()
# # print(weight)
# # # t.hist(weight, label='bins=10')
#
# plt.hist(easy_score, bins=50,  color='orange', range=(0,1))
# plt.ylim(0,6)
# plt.xlabel("Dice score",fontsize=12)
# plt.ylabel("Count",fontsize=12)
# plt.legend()
# plt.show()
#
# plt.hist(hard_score, bins=50,  color='green', range=(0,1))
# plt.ylim(0,6)
# plt.xlabel("Dice score",fontsize=12)
# plt.ylabel("Count",fontsize=12)
# plt.legend()
# plt.show()