![OCELOT LOGO](logo/ocelot_banner.png)

# OCELOT 2023: BlobCell Mothod
 
This repository presents an approach submitted for the [Grand Challenge OCELOT 23](https://ocelot2023.grand-challenge.org/), utilizing the BlobCell Method. The forthcoming paper is titled "Generating BlobCell label from weak annotations for precise Cell Segmentation."
For more information about challenge refer to [page](https://lunit-io.github.io/research/publications/ocelot/).


# Overall process
 
An input/output interface has been implemented to load input images stored on the platform and generate cell predictions. 
The process comprises two main stages:
1. Training/Inferencing Cell Segmentation
2. Cell detection from Inferencing Cell Segmentation model


# Code

## The train code for cell segmentation is as follows:

* Training only cell
```python
train_lunit_for_paper_only_cell.py
```

* Training only tissue
```python
train_lunit_for_paper_only_cell.py
```

* Training the Tissue Injection model
```python
train_lunit_for_paper_with_2_concat_no_MONAI.py
```

* Training the Tissue-BlobCell Injection model
```python
train_lunit_for_paper_with_3_concat_no_MONAI.py
```



## The code for inferencing cell segmentation model is as follows.:

* Inferencing only cells on validation set
```python
predict_for_visualize_norm_Lunit_No_MONAI_val_with_dataloader_like_wandb.py
```

* Inferencing only cells on test set
```python
predict_for_visualize_norm_Lunit_No_MONAI_test_with_dataloader_like_wandb.py
```



* Inference on validation set with Tissue Injection model (using trained Tissue Model)
```python
predict_for_visualize_norm_Lunit_No_MONAI_val_with_dataloader_like_wandb_with_tissue.py
```

* Inference on validation set with Tissue Injection model (using trained Tissue Model)
```python
predict_for_visualize_norm_Lunit_No_MONAI_test_with_dataloader_like_wandb_with_tissue.py
```

* Inference on validation set with Tissue-BlobCell Injection model (using trained Tissue, BlobCell Model)
```python
predict_for_visualize_norm_Lunit_No_MONAI_val_with_dataloader_like_wandb_with_tissue_blobcell.py
```

* Inference on test set with Tissue-BlobCell Injection model (using trained Tissue, BlobCell Model)
```python
predict_for_visualize_norm_Lunit_No_MONAI_test_with_dataloader_like_wandb_with_tissue_blobcell.py
```




* Inference on validation set with Tissue Injection model (Use of GT in Tissue)
```python
predict_for_visualize_norm_Lunit_No_MONAI_val_with_dataloader_like_wandb_with_tissue_with_GT.py
```

* Inference on validation set with Tissue Injection model (Use of GT in Tissue)
```python
predict_for_visualize_norm_Lunit_No_MONAI_test_with_dataloader_like_wandb_with_tissue_with_GT.py
```

* Inference on validation set with Tissue-BlobCell Injection model (Use of GT in Tissue, BlobCell)
```python
predict_for_visualize_norm_Lunit_No_MONAI_val_with_dataloader_like_wandb_with_tissue_blobcell_with_GT.py
```

* Inference on test set with Tissue-BlobCell Injection model (Use of GT in Tissue, BlobCell)
```python
predict_for_visualize_norm_Lunit_No_MONAI_test_with_dataloader_like_wandb_with_tissue_blobcell_with_GT.py
```



## The code for cell detection using the cell segmentation model is as follows.
* cell detection using independently trained Tissue model and BlobCell model
```python
cell_detection_from_OCELOT/process.py
```

* cell detection using GT of Tissue and BlobCell
```python
cell_detection_from_OCELOT/process_with_GT.py
```

* evaluate inference results (json)
```python
cell_detection_from_OCELOT/evaluation/eval.py
```


