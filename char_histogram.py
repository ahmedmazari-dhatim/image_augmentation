import pandas as pd


def find_group(val):
    unique_values = set(df.manual_raw_value.apply(list).sum())
    for unique in unique_values:
        # get the number of occurence of all the unique values
        # then make a histogram



df = pd.read_csv('/home/ahmed/Pictures/cogedis/cogedis_words_3/words.csv',sep=',')
df = df.astype(str)
df=df.replace(['é','è','È','É'],'e', regex=True)
df=df.replace(['à','â','À'],'a', regex=True)
df.manual_raw_value=df.manual_raw_value.str.lower()

unique_values=set(df.manual_raw_value.apply(list).sum())



df.manual_raw_value.apply(find_group)

df.manual_raw_value.apply(find_group).value_counts().plot(kind='bar')