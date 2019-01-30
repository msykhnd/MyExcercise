Names = ["Alice", "Bob", "Charchill"]
Ages = [20, 15, 35]

for name, age in zip(Names, Ages):
    print(name, age)

Tanks = ["T-55A", "T57 heavy", "KV-2"]

for name, age, tank in zip(Names, Ages, Tanks):
    print(name, age, tank)

Rate_Colors = ["Blue", "Green", "Yellow", "Red"]

for name, color in zip(Names, Rate_Colors):
    print(name, color)
    # 小さい方にそろえる

from itertools import zip_longest

for name, color in zip_longest(Names, Rate_Colors, fillvalue="NO_NAME"):
    print(name, color)
    # fillvalueで穴埋め. default NONE

z = zip(Names, Ages, Tanks)

print("Zipped Type", type(z))
print("print z :", z)
print("print list(z)", list(z))
