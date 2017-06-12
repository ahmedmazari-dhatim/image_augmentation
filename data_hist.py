import pandas as pd

def find_group(val):
    val = str(val)
    val= val.lower()

    if val.isalpha():
        return 'Alpha'
    elif val.isdigit():
        return 'digit'
    elif val.isalnum and any(c.isalpha() for c in val):
        return 'Alphanumeric'
    else:
        return 'Special'



df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv',sep=',')
df = df.astype(str)
df=df.replace(['é','è','È','É'],'e', regex=True)
df=df.replace(['à','â','À'],'a', regex=True)
df.manual_raw_value=df.manual_raw_value.str.lower()

df.manual_raw_value.apply(find_group)

df.manual_raw_value.apply(find_group).value_counts().plot(kind='bar')