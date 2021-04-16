import re
from nltk import ngrams

s = "This            is a test sent'sence \n for ngram examinati__on"

i_res = re.sub("\s\s+"," ",s).lower()
print(i_res)
res = re.sub("[^0-9a-zA-Z ]","", i_res)
print(res)
# gs = ngrams(s.split(" "), 3)
# print(list( gs ))
