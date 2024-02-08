import pandas as pd

df = pd.read_csv('./data/drugbank/drugbank_DDI.tab', sep='\t')



grouped_data = df.groupby(['ID1', 'ID2'])['Y'].nunique()
df2 = grouped_data.reset_index()
two_labels = df2[df2['Y'] > 1]
print(two_labels)

label_grouped = df.groupby(['Map'])['Y'].count()
df3 = label_grouped.reset_index()

for index, row in two_labels.iterrows():
    samples = df[(df['ID1'] == row['ID1']) & (df['ID2'] == row['ID2'])]
    sample1 = samples.iloc[0]
    sample2 = samples.iloc[1]
    num_label1 = df3[df3['Map'] == sample1['Map']]['Y'].values[0]
    num_label2 = df3[df3['Map'] == sample2['Map']]['Y'].values[0]

    if num_label1 >= num_label2:
        df.drop(sample2.name, inplace=True)
    else:
        df.drop(sample1.name, inplace=True)


df.to_csv('./data/drugbank/drugbank_DDI_revised.tab', sep='\t')
revised = pd.read_csv('./data/drugbank/drugbank_DDI_revised.tab', sep='\t')
grouped_data = df.groupby(['ID1', 'ID2'])['Y'].nunique()
df2 = grouped_data.reset_index()
two_labels = df2[df2['Y'] > 1]
print(two_labels)
    




# labels = df['Map'].unique()

# for label in labels:
#     print(label)
#     print('-------------------------------------')
