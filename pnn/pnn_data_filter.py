import os
import re
import string
from string import punctuation

data_folder = r'D:\DeeplearningData\PNN测试数据\1. yelp_review_full_csv'
test_file = os.path.join(data_folder, 'test.csv')

with open(test_file) as f:
  lines = f.readlines()
  count = 0
  for line in lines:
    data = line.strip().split('","')
    label = data[0][1:]
    text = data[1][:-1]
    text = text.replace('\\n', ' <EOL> ')
    words = text.split(' ')
    s = """!():<=>?[]_{|}~"""
    c = [re.sub(r'[{}]+'.format("""#$%&*+,-./;@\^`"""), '', x) for x in words]
    print(c)

    for word in words:
      print(word)

    count += 1
    if count > 5:
      break
