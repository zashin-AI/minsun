import numpy as np
from pprint import pprint
import tensorflow as tf
tf.set_random_seed(42)

#배열 생성
unsorted_arr = np.random.random((3, 3))

pprint(unsorted_arr)
'''
[[0.67179746 0.57124357 0.72119571]
 [0.95811413 0.40769315 0.06067548]
 [0.34472854 0.73734489 0.32720539]]
'''
print(unsorted_arr)
'''
[[0.67179746 0.57124357 0.72119571]
 [0.95811413 0.40769315 0.06067548]
 [0.34472854 0.73734489 0.32720539]]
'''

#데모를 위한 배열 복사
unsorted_arr1 = unsorted_arr.copy()
unsorted_arr2 = unsorted_arr.copy()
unsorted_arr3 = unsorted_arr.copy()

#배열 정렬
unsorted_arr1.sort()
pprint(unsorted_arr1)
'''
array([[0.57124357, 0.67179746, 0.72119571],
       [0.06067548, 0.40769315, 0.95811413],
       [0.32720539, 0.34472854, 0.73734489]])
'''
#배열 정렬, axis=-1
unsorted_arr1.sort(axis=-1)
pprint(unsorted_arr1)
'''
array([[0.57124357, 0.67179746, 0.72119571],
       [0.06067548, 0.40769315, 0.95811413],
       [0.32720539, 0.34472854, 0.73734489]])
'''
#배열 정렬, axis=0
unsorted_arr2.sort(axis=0)
pprint(unsorted_arr2)
'''
array([[0.34472854, 0.40769315, 0.06067548],
       [0.67179746, 0.57124357, 0.32720539],
       [0.95811413, 0.73734489, 0.72119571]])
'''

#배열 정렬, axis=1
unsorted_arr3.sort(axis=1)
pprint(unsorted_arr3)
'''
array([[0.57124357, 0.67179746, 0.72119571],
       [0.06067548, 0.40769315, 0.95811413],
       [0.32720539, 0.34472854, 0.73734489]])
'''

# 결론 : sort() = sort(axis=-1) = sort(axis=1) 값은 같다.