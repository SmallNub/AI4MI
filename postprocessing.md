# Post-processing for SegTHOR thoracic CT segmentation

Purpose is to apply post-processing techniques after slice stitching for 2D/2.5D models or directly after inference for 3D models. 

Important to note: anatomical topology and prediction errors differ by organ -> retain post-processing only if it improves patient-level 3D evaluation. This follows the principle used by nnUnet: connected-component post-processing should be selected from validation evidence, rather than applied automatically to every class.

The implementation is 'postprocessing.py'. It is made separate from model training & inference such that each policy can be evaluated against the same raw predictiond and same GT masks.

The workflow for now has 2 policies:

1. 'none': no-op baseline > writees output volumes w/o changing any segmentation voxel
2. 'heart_lcc': for heart label '2' retain only the largest 3D connected component and set all other disconnected heart components to background. Labels 1/3/4 unchanged.

## Candidate methods

1. Connected-component anaylsis
   A connected component algorithm identifies disconnected regins of a predicted class mask.
   Uses: i) retain solely largest component; ii) discard components below a minmimum physical volume; iii) report component counts as a diagnostic without modifying predictions
   Implementation uses 3D connectivity and records components statistics for every patient and class.
   *"Connected component-based post-processing is commonly used in medical image segmentation"* [1]
2. Small-component removal
3. Hole filling
4. Morphological closing

## Organ-specific hypotheses to test

These are not conclusions

| Organ     | Plausible policy                                            | Main risk                                                       |
| --------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| Esophagus | Remove only very small disconnected islands                  | Largest-component retention may remove true fragmented portions |
| Heart     | Largest connected component; optionally hole filling         | Aggressive morphology can alter valid boundaries                |
| Trachea   | Remove isolated islands; inspect continuity before closing   | Gap bridging can change airway topology                         |
| Aorta     | Remove tiny islands while allowing multiple major components | Largest-component-only may remove valid aortic regions          |

## Small test:

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
