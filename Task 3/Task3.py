import pandas as pd

a = pd.read_excel(r"D:\3 Grade\datamining\Task 3\data\people_data.xlsx")
print(a)
print("==============================================")

print(a.isnull().sum())

print("==============================================")

print(a.describe())

print("==============================================")

print(a.dtypes)
print("==============================================")

print(a.isnull().sum().sum())
print("==============================================")

b=a.dropna(how='any')
print(b)
print("==============================================")
c=a.dropna(how='all')
print(c)
print("==============================================")
d=a.fillna(0)
print(d)
print("==============================================")
e=a.fillna({'Name':0, 'Job':50})
print(e)
print("==============================================")
f=a.fillna(method='ffill')
print(f)
print("==============================================")
z=a.fillna(a['Age'].mode())
print(z)
print("==============================================")
g=a.fillna(method='bfill')
print(g)
print("==============================================")
f=a.fillna(method='ffill')
print(f)