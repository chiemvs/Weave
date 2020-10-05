import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 
from PermutationImportance import sklearn_permutation_importance

length = 1000
do_datetime = True # Whether the pandas index is datetime or not does not make a difference, only the fact that pandas aligns all indices
if do_datetime:
    index = pd.Index(pd.date_range(start = '2000-01-01', periods = length))
else:
    index = pd.RangeIndex(length)
X = pd.DataFrame({'A': np.random.normal(size = length), 'B':np.random.normal(size = length)}, index = index) # y will be a linear combination of A, B is unrelated noise
y = X.loc[:,['A']] * 2 + 3 

X_train = X.iloc[:int(0.8*length),:]
y_train = y.iloc[:int(0.8*length),:]
X_val = X.iloc[int(0.8*length):,:]
y_val = y.iloc[int(0.8*length):,:]

model = LinearRegression()
model.fit(X = X_train, y = y_train)

"""
A and B have equal singlepass importance when using pandas frames. Both very low mean_absolute_error. As if permutation did not happen.
"""
result_pd = sklearn_permutation_importance(model = model, scoring_data = (X_val, y_val), evaluation_fn = mean_absolute_error, scoring_strategy = 'argmax_of_mean', njobs = 1, nbootstrap = 1)

print('pandas:')
print(f'singlepass: {result_pd.retrieve_singlepass()}')
print(f'multipass: {result_pd.retrieve_multipass()}')

"""
A and B have unequal singlepass importance when using numpy. As expected permuting A results in very high mean_absolute_error, not for B, which is unimportant
"""
result_np = sklearn_permutation_importance(model = model, scoring_data = (X_val.values, y_val.values), variable_names = X_val.columns, evaluation_fn = mean_absolute_error, scoring_strategy = 'argmax_of_mean', njobs = 1, nbootstrap = 1)

print('numpy:')
print(f'singlepass: {result_np.retrieve_singlepass()}')
print(f'multipass: {result_np.retrieve_multipass()}')

"""
I believe the problem lies in the alignment of pandas objects
by pd.concat. Called in PermutationImportance.utils.make_data_from_columns
"""
indices = np.random.permutation(len(X_val))
A_permuted = X_val.iloc[indices,X_val.columns.tolist().index('A')]
print('A == A_permuted?', X_val['A'].equals(A_permuted))
X_val_permuted = pd.concat([A_permuted,X_val['B']], axis = 1)
print('X_val == X_val_permuted?', X_val.equals(X_val_permuted))

