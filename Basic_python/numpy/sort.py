import numpy as np

'''
    numpy.sort
    ndarray.sort
    ndarray.argsort
'''

a = np.array([
    5, 7, 10, 9, 1, 0
])
a_sorted = np.sort(a)
a_sorted_reverse = np.sort(a)[::-1]

# print(a_sorted)
# print(a_sorted_reverse)

a2d = np.array([
    [20, 5, 100],
    [1, 50, 200],
    [900, 10, 9]

])
a2d_sorted_default = np.sort(a2d)
a2d_sorted_axis0 = np.sort(a2d, axis=0)
a2d_sorted_axis1 = np.sort(a2d, axis=1)

print("a2d\n", a2d)
# print("a2d_sorted_default\n", a2d_sorted_default)
# print("a2d_sorted_axis0\n", a2d_sorted_axis0)
# print("a2d_sorted_axis1\n", a2d_sorted_axis1)

a2d_sort_col_index = np.argsort(a2d)

# print("a2d_sort_col_index\n", a2d_sort_col_index)

target_col = a2d[:,1]
print(target_col)
print(np.argsort(target_col))
print(a2d[np.argsort(target_col)])
