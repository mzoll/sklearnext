Sklearnext documentation
=====================

Motivation
----------
One of the most commonly used packages for Machine Learning in the python domain in the scikit-learn or `sklearn` package. The package itself provides a profound library containing tools for dataset manipulation over preprocessing functions down to common use classifiers and regressors. This makes sklearn the predominant library in the machine learning community. Another library with a comparable reach in the ml-community is `pandas`, a library to hold and process large data tables (DataFrames). Most tutorials teaching ml with python will use both this libraries in unision, to store training data, manipulate and preprocessit and finally train a machine learning model.

While it seems that using pandas::DataFrames as input for sklearn algorithms is working for most cases, the truth is that this is only achieved by downcasting of pandas objects to their base representation as ´numpy´ arrays and martices. This process discards much of the structure of the dataframes and is prune to information loss, for example of column names, compressed data formats and other dynamic relations within the data.

Sklearnext is an attempt to truly integrate the power of the pandas library into the sklearn ecosystem, by providing wrappers and tools, which make the native sklearn components work with and handle pandas objects.
Some of the things that we can achive through this are: Correct Feature importance tables with lables for Learning algorithms, start-to-end pipelining with data manipulation through column names, complex data manipulation.

Pipelining
----------

The `sklearn` library comes with a native interface for data pipelining and  

These classes are used to assemble a complex and especially nested estimator model
by chaining Transformers and Estimators in FeatureUnions and Pipelines   
The input-output scheme is as follows:
Transformers-- In: DataFrame Out: DataFrame
Estimator-- In: DataFrame Out: numpy-array
EstimatorWrapper-- wrap Estimator or Pipeline-- In: nparray Out: DataFrame
FeatureUnions-- assembly of Tranformers-- In: DataFrame Out: DataFrame
TransformerPipe-- assembly of Transformers-- In: DataFrame Out: DataFrame
ForkingPipeline-- a pipeline-- In: DataFrame Out: nparray
Pipeline-- assembly of Transformers plus Estimator-- In: DataFrame Out:ndarray