Classifier validation added for the ASOC submission
===================================================

The main manuscript and Supplementary Section S9 report classifier-selection,
probability-calibration, adaptive-refitting, and permutation-importance checks.
They use only the stored initial sample and run histories; no new circuit
simulation was performed.

Files
-----
scripts/rf_classifier_analysis.py
    Recomputes the classifier comparison, library-index replay, learning curve,
    permutation importance, and ensemble-disagreement diagnostics from the
    packaged data.

data/precomputed/rf_classifier_numbers.json
    Complete numerical output from the analysis used to prepare the manuscript.
    It includes both the reproducible library-index representation and the
    component-property embedding used in the original optimization runs.

outputs/classifier_validation/
    Created when the script is run.

Reproduction
------------
From Circuit_Study/scripts, run:

    python rf_classifier_analysis.py

The default packaged-data run uses library indices for the two component-library
variables. The original optimization used an embedding based on foundry-library
inductance, quality factor, and self-resonance frequency. Those proprietary
library values cannot be distributed. The manuscript therefore reports both the
as-run embedding replay and the library-index sensitivity check; their ROC-AUC
values differ by at most 0.03 and lead to the same calibration conclusion.

Important interpretation
------------------------
The original run histories did not log classifier probabilities. Calibration is
therefore a replay: before each batch, the classifier is refitted using the
recorded configuration and random seed on the data available at that time, and
then scored on the next proposed batch. This limitation is stated explicitly in
the manuscript and supplement.
