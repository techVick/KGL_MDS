# KGL_MDS
1. mds_s1
ordinal encoding - converting feature variables to their corresponding integer value
Decision tree regressor with depth 2

2. mds_s2
ordinal encoding - converting feature variables to their corresponding integer value
RandomForestRegressor and GradientBoostingRegressor, with a internal commented loop for tree depth and other parameter optimization using grid search

3. mds_s3
ordinal encoding - converting feature variables to their corresponding integer value
XGBRegressor

4. mds_s4
One hot encoding - total variable in total column and each with one and rest zero
RandomForestRegressor,with a internal commented loop for tree depth and other parameter optimization

5. mds_s5
empty for now

6. mds_s6
One hot encoding - total variable in total column and each with one and rest zero
RandomForestRegressor with decomposition PCA for feature reduction
 np.setdiff1d is used to match train and test data sets after feature reduction, clumns should be same. Train test can have different features, so this is necessary.
 
7. mds_s7
ordinal encoding - converting feature variables to their corresponding integer value
RandomForestRegressor with decomposition PCA for feature reduction
 
 8. mds_s8
Binary encoding : feature to integer, integer to binary, binary to columns
AdaBoostRegressor, GradientBoostingRegressor, with a internal commented loop for tree depth and other parameter optimization
