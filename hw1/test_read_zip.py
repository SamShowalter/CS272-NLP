import pandas as pd
import zipfile
import os 

print(os.getcwd())

zf = zipfile.ZipFile('presidential-candidate-classification-w20.zip') # having First.csv zipped file.
print(zf.__dict__)
df = pd.read_csv(zf.open('speech_basic.csv'))
