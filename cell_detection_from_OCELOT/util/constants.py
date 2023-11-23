from pathlib import Path

# # Grand Challenge folders were input files can be found
# GC_CELL_FPATH = Path("/input/images/cell_patches/")
# GC_TISSUE_FPATH = Path("/input/images/tissue_patches/")

# GC_METADATA_FPATH = Path("/input/metadata.json")

# # Grand Challenge output file
# GC_DETECTION_OUTPUT_PATH = Path("/output/cell_classification.json")

# # Sample dimensions
# SAMPLE_SHAPE = (1024, 1024, 3)


# For trainset Evaluation
GC_CELL_FPATH = Path("/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/val/cell") # cell_n
# GC_TISSUE_FPATH = Path("/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/test/tissue") 
GC_TISSUE_FPATH = Path("/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/tissue_crop2") # tissue2
GC_BlobCell_FPATH = Path("/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/BlobCell_one_label")

GC_METADATA_FPATH = Path("/vast/AI_team/sukmin/datasets/ocelot2023_v1.0.1/ocelot2023_v1.0.1/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = Path("/home/sukmin/OCELOT_2023_BlobCell_Method_for_submission/ocelot23algo_for_submit_for_Github_submit_for_paper/test/output/val_GT/3_concat.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)




