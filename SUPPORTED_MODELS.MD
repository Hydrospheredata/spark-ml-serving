# Supported models

Here is a list of supported models.
V - Serving support
N/A - Model is not present
X - Not implemented

If model is not present here, it's unsupported.

## Classification

Model/Spark                              |    2.0    |    2.1    |   2.2
-----------------------------------------|-----------|-----------|----------
DecisionTreeClassificationModel          |     V     |     V     |    V
GBTClassificationModel                   |     V     |     V     |    V
MultilayerPerceptronClassificationModel  |     V     |     V     |    V
NaiveBayes                               |     V     |     V     |    V
RandomForestClassificationModel          |     V     |     V     |    V
LogisticRegressionModel                  |     V     |     V     |    V
LinearSVCModel                           |    N/A    |    N/A    |    V


## Regression

Model/Spark                              |    2.0    |    2.1    |   2.2
-----------------------------------------|-----------|-----------|----------
DecisionTreeRegressionModel              |     V     |     V     |    V
GBTRegressionModel                       |     V     |     V     |    V
LinearRegressionModel                    |     V     |     V     |    V
RandomForestRegressionModel              |     V     |     V     |    V


## Clustering

Model/Spark             |    2.0    |    2.1    |   2.2
------------------------|-----------|-----------|----------
GaussianMixtureModel    |     V     |     V     |    V
KMeansModel             |     V     |     V     |    V
LDAModel                |     V     |     V     |    V

## Features

Model/Spark            |    2.0    |    2.1    |   2.2
-----------------------|-----------|-----------|----------
BinarizerModel         |     V     |     V     |    V
ChiSqSelectorModel     |     V     |     V     |    V
CountVectorizerModel   |     V     |     V     |    V
DCTModel               |     V     |     V     |    V
HashingTF              |     V     |     V     |    V
IDFModel               |     V     |     V     |    V
IndexToString          |   V       |   V       |    V
MaxAbsScalerModel      |   V       |   V       |    V
MinMaxScalerModel      |   V       |   V       |    V
NGramModel             |   V       |   V       |    V
NormalizerModel        |   V       |   V       |    V
OneHotEncoderModel     |   V       |   V       |    V
PCAModel               |   V       |   V       |    V
PolynomialExpansion    |   V       |   V       |    V
RegexTokenizer         |   V       |   V       |    V
StandardScalerModel    |   V       |   V       |    V
StopWordsRemover       |   V       |   V       |    V
StringIndexerModel     |   V       |   V       |    V
Tokenizer              |   V       |   V       |    V
VectorAssembler        |   V       |   V       |    V
VectorIndexer          |   V       |   V       |    V
Word2VecModel          |   V       |   V       |    V
ImputerModel           |   N/A     |   N/A     |    V
