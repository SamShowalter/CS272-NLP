import numpy as np
import re
from nltk import ngrams

# s = "This            is a test sent'sence \n for ngram examinati__on"

# i_res = re.sub("\s\s+"," ",s).lower()
# print(i_res)
# res = re.sub("[^0-9a-zA-Z ]","", i_res)
# print(res)
# # gs = ngrams(s.split(" "), 3)
# # print(list( gs ))
a = [1,2,3,4]
# print(a[-2:])
a = (1,32,4)
print(a[2:] == (4,))
