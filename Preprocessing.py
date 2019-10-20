import pandas as pd
import re
import gensim # thư viện NLP
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from pandas import DataFrame


df = pd.read_csv("./Data/Test_original.csv",encoding="UTF8")

#df = pd.read_csv("test.csv",encoding="UTF8")

df1 = pd.read_csv("./Preprocessing Source/pre_processing.csv", encoding="UTF8")

for i in range(len(df)):
    value = df.iat[i, 1]

    # Biến đổi tất cả thành chữ thường
    value = value.lower()
    #Loại bỏ kí tự số
    pattern1 = '\d+'
    valueNew = re.sub(pattern1, '', value)
    #Xóa khoảng trắng đầu và cuối chuỗi
    valueNew = valueNew.strip()
    # Loại bỏ các kí tự đặc biệt
    valueNew = gensim.utils.simple_preprocess(value)
    valueNew = ' '.join(valueNew)
    # Sau khi loại bỏ kí tự đặc biệt thì file sẽ bị mất nhận diện chuỗi -> thêm dấu nháy vào chuỗi
    valueNew = '"' + str(valueNew) + '"'
    #Chuẩn hóa ngôn ngữ tiếng việt sử dụng pyvi
    value = gensim.utils.tokenize(valueNew)
    valueNew = ViTokenizer.tokenize(value)
    # Loại bỏ thẻ tag html, css, js...
    valueNew = re.sub("(<.*?>)", "", valueNew)

    valueNew = re.sub("(url photos from .*? post)", "", valueNew)
    valueNew = re.sub("updated .*? status", "", value)

    for j in range(len(df1)):
        valueReplace = df1.iat[j,0]
        if (valueReplace in valueNew):
            valueNew = re.sub(valueReplace, '', valueNew)
            print(valueNew)
            df.iat[i, 1] = valueNew

    print("New\n" + valueNew)
    df.iat[i,1] = valueNew




# Do the same with Train_orginal to have Train_Clean file
df.to_csv('./Data/Test_Clean.csv')

