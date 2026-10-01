# Post-processing for SegTHOR thoracic CT segmentation

Purpose is to apply post-processing techniques after slice stitching for 2D/2.5D models or directly after inference for 3D models.

Important to note: anatomical topology and prediction errors differ by organ -> retain post-processing only if it improves patient-level 3D evaluation. This follows the principle used by nnUnet: connected-component post-processing should be selected from validation evidence, rather than applied automatically to every class.

The implementation is 'postprocessing.py'. It is made separate from model training & inference such that each policy can be evaluated against the same raw predictiond and same GT masks.

## Implemented policies (wip)

Current implementations are restricted to literature grounded connected-component operations. Does net yet do something like fill holes, performing morphological closing, adding voxels or overwriting another organ label.

1. **none**: baseline; writes output volumes without changing any segmentation voxel
2. **lcc_3d_non_esophagus**: retains largest 3D connected component for heart, trachea and aorta only; esophagus left unchanged because Han et al. [2] use a different strategy for its potentially disconnected tubular predictions.
3. **esophagus_min_500mm3**: retains all esophagus components with volume at least $500\ \text{mm}^3$ and removes smaller esophageal component.
4. **heart_3d_lcc_axial_2d_lcc**: retains largest 3D heart component, then retains largest 2D heart component in each axial slice. Labels 1,3,4 are unchanged.
5. **relative_20pct_all**: independently for every foreground class, removes components smaller than $20%$% of that class's largest component; can retain more than one substantial component. $%$

*note that all policies must be evaluated relative to **none** using same raw predicitons, fixed validation patients, GT and metric implementations.*

## Post-processing methods motivations for organs-at-risk based on literature research

* All organs:

  * 3D connected component denoising Han et al. [2] state in sec 3.5: "*After both coarse- and fine-resolution segmentation, we remove noisy
    isolated segments by picking the largest 3D connected component*."
  * Retain components relative to largest; Van Harten et al. [5] write: "*Given that voxel classification may result in isolated (clusters of) voxels disconnected from the target structure, connected components smaller than 0.2 times the largest component in the class were removed using largest component selection."*
  * Axis-based continuity denoising; Feng et al. [4] report that predicted organs can be disconnected despite connected GT labels, can contain "multiple organs inclusions in the same slice" and contain "background noise inside". Their method: "*The CT image is sliced along three dimensions respectively, then count the number of connected blocks of each
    organ. For each dimension and each organ, the largest connected block is retained, and the other parts are considered background noise and therefore removed."*
* Esophagus:

  * Do not assume largest component only is appropriate, Han et al. [2] state in sec 3.5: "*For esophagus segmentation, instead of picking the largest connected component in the fine-resolution, we pick the connected components with size >500 voxels. This post-processing will take care of possible disconnections of esophagus segmentation due to its tubular structure."*
* Heart:

  * 3D LLC followed by slice-wise 2D LLC, Han et al. [2] state in sec 3.5: "*For heart segmentation, there may be small isolated segments in 2D
    slices. To remove them, we will pick the largest 2D connected component at each slice after the 3D one."*

**So based on literature research, the following policy set, but not limited to, can be useful to test:**

| Policy# | Policy                                                            | Organ(s)                 | Literature basis       |
| ------- | ----------------------------------------------------------------- | ------------------------ | ---------------------- |
| P0      | No post-processing                                                | All                      | baseline               |
| P1      | 3D LCC                                                            | Each class independently | Han et al.; Kim et al. |
| P2      | Keep esophagus components$\ge 500\ \text{mm}^3$                 | Esophagus                | Han et al.             |
| P3      | Heart 3D LCC + slice-wise 2D LCC                                  | Heart                    | Han et al.             |
| P4      | Remove components smaller than $20\%$ of the largest component | Each class independently | van Harten et al.      |
| P5      | Axis-based continuity denoise                                     | Potentially all four     | Feng et al.            |

	

Small test (outdated):

**Setup**

| Item                     | Value                                                      |
| ------------------------ | ---------------------------------------------------------- |
| Model                    | ENet3D                                                     |
| Loss                     | Compound loss                                              |
| Training duration        | 2 epochs                                                   |
| Data split               | 35 training / 5 validation patients                        |
| Validation patients      | Patient_01, Patient_14, Patient_15, Patient_16, Patient_33 |
| GPU                      | NVIDIA H100                                                |
| Post-processing policies | `none`, `heart_lcc`                                    |
| Metric backend           | DisTorch on CUDA                                           |

**Metrics: solely heart class**

| Policy        | Heart Dice | Heart HD95 (mm) | Heart ASSD (mm) | Heart precision | Heart recall |
| ------------- | ---------: | --------------: | --------------: | --------------: | -----------: |
| `none`      |     0.0437 |           49.32 |           21.34 |          0.7667 |       0.0227 |
| `heart_lcc` |     0.0209 |           74.90 |           38.84 |          0.9499 |       0.0106 |

**Interpretation**

These values are not final results (still a WIP). Model was trained for only two epochs and had poor valdiation segmentation quality. The heart_lcc policy increased precision but reduced all other metrics in this undertrained setting.

## Literature

[1] Isensee, F., Jaeger, P.F., Kohl, S.A.A. *et al.* nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
*Nat Methods*  **18** , 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

* Relevant information: select connected-component post-processing only when validation performance supports it.

[2] Han, M., Yao, G., Zhang, W., Mu, G., Zhan, Y., Zhou, X., & Gao, Y. (2019). Segmentation of CT Thoracic Organs by Multi-resolution VB-nets.  *SegTHOR@ ISBI* ,  *2019* , 1-4.

[3] Kim, S., Jang, Y., Han, K., Shim, H., & Chang, H. J. (2019). A cascaded two-step approach for segmentation of thoracic organs. In *CEUR Workshop Proceedings* (Vol. 2349). CEUR-WS.

[4] Feng, M., Huang, W., Wang, Y., & Xie, Y. (2019, April). Multi-organ Segmentation using Simplified Dense V-net with Post-processing. In  *SegTHOR@ ISBI* .

[5] van Harten, L. D., Noothout, J. M., Verhoeff, J. J., Wolterink, J. M., & Išgum, I. (2019). Automatic segmentation of organs at risk in thoracic CT scans by combining 2D and 3D convolutional neural networks. In  *2019 SegTHOR Challenge: Segmentation of THoracic Organs at Risk in CT Images* . CEUR.
