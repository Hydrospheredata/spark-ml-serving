package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.classification._
import io.hydrosphere.spark_ml_serving.clustering._
import io.hydrosphere.spark_ml_serving.common.LocalTransformer
import io.hydrosphere.spark_ml_serving.preprocessors._
import io.hydrosphere.spark_ml_serving.regression._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.clustering.{GaussianMixtureModel, KMeansModel, LocalLDAModel => SparkLocalLDAModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.{PipelineModel, Transformer}

object CommonTransormerConversions extends DynamicTransformerConverter {

  implicit def transformerToLocal(transformer: Transformer): LocalTransformer[_] = {
    transformer match {
      case x: PipelineModel => new LocalPipelineModel(x)

      // Classification models
      case x: DecisionTreeClassificationModel => new LocalDecisionTreeClassificationModel(x)
      case x: MultilayerPerceptronClassificationModel =>
        new LocalMultilayerPerceptronClassificationModel(x)
      case x: NaiveBayesModel                 => new LocalNaiveBayes(x)
      case x: RandomForestClassificationModel => new LocalRandomForestClassificationModel(x)
      case x: GBTClassificationModel          => new LocalGBTClassificationModel(x)
      // Clustering models
      case x: GaussianMixtureModel => new LocalGaussianMixtureModel(x)
      case x: KMeansModel          => new LocalKMeansModel(x)
      case x: SparkLocalLDAModel   => new LocalLDAModel(x)

      // Preprocessing
      case x: Binarizer            => new LocalBinarizer(x)
      case x: CountVectorizerModel => new LocalCountVectorizerModel(x)
      case x: DCT                  => new LocalDCT(x)
      case x: HashingTF            => new LocalHashingTF(x)
      case x: IndexToString        => new LocalIndexToString(x)
      case x: MaxAbsScalerModel    => new LocalMaxAbsScalerModel(x)
      case x: MinMaxScalerModel    => new LocalMinMaxScalerModel(x)
      case x: NGram                => new LocalNGram(x)
      case x: Normalizer           => new LocalNormalizer(x)
      case x: OneHotEncoder        => new LocalOneHotEncoder(x)
      case x: PCAModel             => new LocalPCAModel(x)
      case x: PolynomialExpansion  => new LocalPolynomialExpansion(x)
      case x: StandardScalerModel  => new LocalStandardScalerModel(x)
      case x: StopWordsRemover     => new LocalStopWordsRemover(x)
      case x: StringIndexerModel   => new LocalStringIndexerModel(x)
      case x: Tokenizer            => new LocalTokenizer(x)
      case x: VectorIndexerModel   => new LocalVectorIndexerModel(x)
      case x: IDFModel             => new LocalIDF(x)
      case x: ChiSqSelectorModel   => new LocalChiSqSelectorModel(x)
      case x: RegexTokenizer       => new LocalRegexTokenizer(x)
      case x: VectorAssembler      => new LocalVectorAssembler(x)

      // Regression
      case x: DecisionTreeRegressionModel => new LocalDecisionTreeRegressionModel(x)
      case x: LinearRegressionModel       => new LocalLinearRegressionModel(x)
      case x: RandomForestRegressionModel => new LocalRandomForestRegressionModel(x)
      case x: GBTRegressionModel          => new LocalGBTRegressor(x)

      case x => SpecificTransformerConversions.transformerToLocal(x)
    }
  }
}
