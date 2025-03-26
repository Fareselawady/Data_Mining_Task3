import pandas as pd
import os 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

Dataset = [
    ['Jett', 'Phoenix', 'Brimstone', 'Sova', 'Raze', 'Sage'],
    ['Viper', 'Phoenix', 'Brimstone', 'Sova', 'Raze', 'Sage'],
    ['Jett', 'Reyna', 'Sova', 'Raze'],
    ['Jett', 'Omen', 'Killjoy', 'Sova', 'Sage'],
    ['Killjoy', 'Phoenix', 'Phoenix', 'Sova', 'Chamber', 'Raze']
]

te = TransactionEncoder()
tefordataset = te.fit(Dataset).transform(Dataset)
print(tefordataset)

print("==============================================")

df = pd.DataFrame(tefordataset, columns=te.columns_)
print(df)

print("==============================================")

from mlxtend.frequent_patterns import apriori

print(apriori(df, min_support=0.6))


print("==============================================")

frequentitemsets = apriori(df, min_support=0.6, use_colnames=True)
print(frequentitemsets)


print("==============================================")

frequentitemsets['length'] = frequentitemsets['itemsets'].apply(lambda x: len(x))
print(frequentitemsets)


print("==============================================")


print(frequentitemsets[(frequentitemsets['length'] == 2)])


print("==============================================")

target_items = {'Jett', 'Sova'}

print(frequentitemsets[(frequentitemsets['itemsets'] == target_items)])

print(association_rules(frequentitemsets, metric="confidence", min_threshold=0.7))


print("==============================================")

rules = association_rules(frequentitemsets, metric="lift", min_threshold=1.2)
print(rules)

print("==============================================")

rules['antecedent_length'] = rules["antecedents"].apply(lambda x: len(x))
print(rules)

print("==============================================")

print(rules[(rules['antecedent_length'] >= 2)])
