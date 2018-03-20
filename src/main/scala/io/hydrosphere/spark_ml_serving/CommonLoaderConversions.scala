package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.classification._
import io.hydrosphere.spark_ml_serving.clustering._
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.preprocessors._
import io.hydrosphere.spark_ml_serving.regression._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._

object CommonLoaderConversions extends DynamicLoaderConverter {
  implicit def sparkToLocal(m: Any): ModelLoader[_] = {
    m match {
      case _: PipelineModel.type => LocalPipelineModel

      case x: ModelLoader[_] => x

      // Classification models
      case _: DecisionTreeClassificationModel.type => LocalDecisionTreeClassificationModel
      case _: MultilayerPerceptronClassificationModel.type =>
        LocalMultilayerPerceptronClassificationModel
      case _: NaiveBayesModel.type                 => LocalNaiveBayes
      case _: RandomForestClassificationModel.type => LocalRandomForestClassificationModel
      case _: GBTClassificationModel.type          => LocalGBTClassificationModel
      // Clustering models
      case _: GaussianMixtureModel.type => LocalGaussianMixtureModel
      case _: KMeansModel.type          => LocalKMeansModel

      // Preprocessing
      case _: Binarizer.type            => LocalBinarizer
      case _: CountVectorizerModel.type => LocalCountVectorizerModel
      case _: DCT.type                  => LocalDCT
      case _: HashingTF.type            => LocalHashingTF
      case _: IndexToString.type        => LocalIndexToString
      case _: MaxAbsScalerModel.type    => LocalMaxAbsScalerModel
      case _: MinMaxScalerModel.type    => LocalMinMaxScalerModel
      case _: NGram.type                => LocalNGram
      case _: Normalizer.type           => LocalNormalizer
      case _: OneHotEncoder.type        => LocalOneHotEncoder
      case _: PCAModel.type             => LocalPCAModel
      case _: PolynomialExpansion.type  => LocalPolynomialExpansion
      case _: StandardScalerModel.type  => LocalStandardScalerModel
      case _: StopWordsRemover.type     => LocalStopWordsRemover
      case _: StringIndexerModel.type   => LocalStringIndexerModel
      case _: Tokenizer.type            => LocalTokenizer
      case _: VectorIndexerModel.type   => LocalVectorIndexerModel
      case _: IDFModel.type             => LocalIDF
      case _: ChiSqSelectorModel.type   => LocalChiSqSelectorModel
      case _: RegexTokenizer.type       => LocalRegexTokenizer
      case _: VectorAssembler.type      => LocalVectorAssembler

      // Regression
      case _: DecisionTreeRegressionModel.type => LocalDecisionTreeRegressionModel
      case _: LinearRegressionModel.type       => LocalLinearRegressionModel
      case _: RandomForestRegressionModel.type => LocalRandomForestRegressionModel
      case _: GBTRegressionModel.type          => LocalGBTRegressor

      case x => SpecificLoaderConversions.sparkToLocal(x)
    }
  }
}
