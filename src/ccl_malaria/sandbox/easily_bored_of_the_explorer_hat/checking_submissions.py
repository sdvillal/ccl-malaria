import os.path as op
from ccl_malaria import MALARIA_DATA_ROOT


def vsc2smi(file1, file2):
    with open(file1, 'r') as reader, open(file2, 'w') as writer:
        for line in reader:
            smi = line.split(',')[1]
            writer.write(smi)
            writer.write('\n')


vsc2smi(op.join(MALARIA_DATA_ROOT, 'submission', 'final-merged-nonCalibrated-hitSelection.csv'),
        op.join(MALARIA_DATA_ROOT, 'submission', 'final-merged-nonCalibrated-hitSelection.txt'))

vsc2smi(op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-avg-scr.csv'),
        op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-avg-scr.txt'))

vsc2smi(op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-avg-unl.csv'),
        op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-avg-unl.txt'))

vsc2smi(op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-stacker=linr-scr.csv'),
        op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-stacker=linr-scr.txt'))

vsc2smi(op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-stacker=linr-unl.csv'),
        op.join(MALARIA_DATA_ROOT, 'submission', 'final-nonCalibrated-stacker=linr-unl.txt'))