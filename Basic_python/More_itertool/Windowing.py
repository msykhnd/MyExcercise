import more_itertools as mit

'''
more_itertools.windowed(seq, n, fillvalue=None, step=1)
'''

array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

all_window = mit.windowed(array, n=3, step=2)
print("type", type(all_window))
print(list(all_window))

all_window = mit.windowed(array, n=3, step=2, fillvalue="Filled")
print("type", type(all_window))
print(list(all_window))
