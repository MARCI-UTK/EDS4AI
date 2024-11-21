import nbformat
import nbconvert
from nbconvert.preprocessors import ExecutePreprocessor

filename = 'CNN2.ipynb'
with open(filename) as ff:
    nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

(nb_out, _) = ep.preprocess(nb_in)

with open('executed_CNN2.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb_out, f)
