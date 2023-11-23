
from evaluator import evaluate_folder


# 비교하려는 GT 폴더 결과
folder_with_gt = '/vast/AI_team/sukmin/datasets/Task506_Stroma_NDM_prep_100x_whole/labelsTs'
# folder_with_gt = '/home/sukmin/datasets/Task111_Carotid_side/labelsTs'


# inference 된 결과
folder_with_pred = '/vast/AI_team/sukmin/Results_Stroma_NDM/io_contrast_Test_100x_CacoX_b32_e10000'
# folder_with_pred = '/vast/AI_team/sukmin/Results/Test_ResUNet_b256_256_b'
# labels = (0, 1, 2, 3) # test 하고 싶은 라벨 입력
labels = (0, 1, 2, 3, 4, 5, 6) # test 하고 싶은 라벨 입력

evaluate_folder(folder_with_gt, folder_with_pred, labels)


# 실행이 완료되면 folder_with_pred 경로에 summary.json이 생성됌

# scp를 활용해서 로컬에서 열어보면 된다. (혹은 서버에서 vim이나 nano로 확인)
# ex) scp -r -P 22 sukmin@10.10.10.14:/nas3/sukmin/inf_rst/multi_organ_model/caddunet_focal_portion /home/sukmin/Downloads
