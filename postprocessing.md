# Post-processing for SegTHOR thoracic CT segmentation

**!! The scripts folder contains a Slurm job (evaluate_postprocessing_policies.sh) that is intended to be used for this whole pipeline, but may be subject to change. !!**

This markdown describes the post-processing stage for SegTHOR thoracic organ-at-risk segmentation. Purpose is to apply post-processing techniques after slice stitching for 2D/2.5D models or directly after inference for 3D models.

Important to note: anatomical topology and prediction errors differ by organ -> retain post-processing only if it improves patient-level 3D evaluation. This follows the principle used by nnUnet: connected-component post-processing should be selected from validation evidence, rather than applied automatically to every class.

The implementation is 'postprocessing.py', which is grounded in what has been done before in prior literature. It is made separate from model training & inference such that each policy can be evaluated against the same raw predictiond and same GT masks.

## Workflow

The intended workflow is:

> Train/infer model
> → reconstruct or resample to original-space NIfTI volumes
> → evaluate unchanged prediction baseline (`none`)
> → evaluate each post-processing policy
> → inspect metrics, audits, and qualitative outputs
> → retain a policy only if validation evidence supports it

For 2D/2.5D models, `postprocessing.py` must be applied only after slice stitching/reconstruction. It must never be applied directly to PNG slices.

## Preconditions

Before connected-component post-processing, the input folder must contain NIfTI prediction files named:

> Patient_XX.nii.gz

Each prediction must have the same shape, voxel spacing and affine as the corresponding GT file:

> data/segthor_train_full/train/Patient_XX/GT.nii.gz

Required because:

- physical component thresholds are interpreted in $\text{mm}^3$;
- HD95 and ASSD use physical voxel spacing;
- corresponding prediction and GT voxels must describe the same patient space locations.

## Implemented policies

Current implementations are restricted to literature grounded connected-component operations. Does net yet do something like fill holes, performing morphological closing, adding voxels or overwriting another organ label.

1. **none**: baseline; writes output volumes without changing any segmentation voxel
2. **lcc_3d_non_esophagus**: retains largest 3D connected component for heart, trachea and aorta only; esophagus left unchanged because Han et al. [2] use a different strategy for its potentially disconnected tubular predictions.
3. **esophagus_min_500mm3**: retains all esophagus components with volume at least $500\ \text{mm}^3$ and removes smaller esophageal component.
4. **heart_3d_lcc_axial_2d_lcc**: retains largest 3D heart component, then retains largest 2D heart component in each axial slice. Labels 1,3,4 are unchanged.
5. **relative_20pct_all**: independently for every foreground class, removes components smaller than $20%$% of that class's largest component; can retain more than one substantial component. $%$

*note that all policies must be evaluated relative to **none** using same raw predicitons, fixed validation patients, GT and metric implementations.*

### Safety properties

For every policy, a changed voxel can only change from its original foreground label to background. Therefore, the code cannot:

> background → organ
> organ A → organ B
> organ B → organ A

Each output folder contains `postprocessing_report.csv`, which records input/output voxels, component counts, removed voxels & the active policy parameters for every patient and foreground class.

## Literature motivation

### All organs: connected-component filtering

* 3D connected component denoising Han et al. [2] state in sec 3.5: "*After both coarse- and fine-resolution segmentation, we remove noisy
  isolated segments by picking the largest 3D connected component*."
* Retain components relative to largest; Van Harten et al. [5] write: "*Given that voxel classification may result in isolated (clusters of) voxels disconnected from the target structure, connected components smaller than 0.2 times the largest component in the class were removed using largest component selection."*
* Axis-based continuity denoising; Feng et al. [4] report that predicted organs can be disconnected despite connected GT labels, can contain "multiple organs inclusions in the same slice" and contain "background noise inside". Their method: "*The CT image is sliced along three dimensions respectively, then count the number of connected blocks of each
  organ. For each dimension and each organ, the largest connected block is retained, and the other parts are considered background noise and therefore removed."*

### Esophagus: do not apply largest-component-only filtering by default

* Do not assume largest component only is appropriate, Han et al. [2] state in sec 3.5: "*For esophagus segmentation, instead of picking the largest connected component in the fine-resolution, we pick the connected components with size >500 voxels. This post-processing will take care of possible disconnections of esophagus segmentation due to its tubular structure."*

### Heart: 3D LCC followed by slice-wise 2D LCC

* 3D LLC followed by slice-wise 2D LLC, Han et al. [2] state in sec 3.5: "*For heart segmentation, there may be small isolated segments in 2D
  slices. To remove them, we will pick the largest 2D connected component at each slice after the 3D one."*

**So based on literature research, the following policy set, but not limited to, can be useful to test:**

| Policy# | Policy                                                          | Organ(s)                 | Literature basis       |
| ------- | --------------------------------------------------------------- | ------------------------ | ---------------------- |
| P0      | No post-processing                                              | All                      | baseline               |
| P1      | 3D LCC                                                          | Each class independently | Han et al.; Kim et al. |
| P2      | Keep esophagus components$\ge 500\ \text{mm}^3$               | Esophagus                | Han et al.             |
| P3      | Heart 3D LCC + slice-wise 2D LCC                                | Heart                    | Han et al.             |
| P4      | Remove components smaller than$20\%$ of the largest component | Each class independently | van Harten et al.      |

## Literature

[1] Isensee, F., Jaeger, P.F., Kohl, S.A.A. *et al.* nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
*Nat Methods*  **18** , 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

* Relevant information: select connected-component post-processing only when validation performance supports it.

[2] Han, M., Yao, G., Zhang, W., Mu, G., Zhan, Y., Zhou, X., & Gao, Y. (2019). Segmentation of CT Thoracic Organs by Multi-resolution VB-nets.  *SegTHOR@ ISBI* ,  *2019* , 1-4.

[3] Kim, S., Jang, Y., Han, K., Shim, H., & Chang, H. J. (2019). A cascaded two-step approach for segmentation of thoracic organs. In *CEUR Workshop Proceedings* (Vol. 2349). CEUR-WS.

[4] Feng, M., Huang, W., Wang, Y., & Xie, Y. (2019, April). Multi-organ Segmentation using Simplified Dense V-net with Post-processing. In  *SegTHOR@ ISBI* .

[5] van Harten, L. D., Noothout, J. M., Verhoeff, J. J., Wolterink, J. M., & Išgum, I. (2019). Automatic segmentation of organs at risk in thoracic CT scans by combining 2D and 3D convolutional neural networks. In  *2019 SegTHOR Challenge: Segmentation of THoracic Organs at Risk in CT Images* . CEUR.
