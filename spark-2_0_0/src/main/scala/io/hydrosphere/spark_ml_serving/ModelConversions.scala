package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.classification._
import io.hydrosphere.spark_ml_serving.regression._
import io.hydrosphere.spark_ml_serving.preprocessors._
import io.hydrosphere.spark_ml_serving.clustering._
import io.hydrosphere.spark_ml_serving.common.{LocalModel, LocalPipelineModel}
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._

trait ModelConversions {
  implicit def sparkToLocal[T <: Transformer](m: Any): LocalModel[T]
}

object ModelConversions extends ModelConversions {
  implicit def sparkToLocal[T <: Transformer](m: Any): LocalModel[T] = {
    m match {
      case _ : PipelineModel.type  => LocalPipelineModel

      case x: LocalModel[T] => x

      // Classification models
      case _: DecisionTreeClassificationModel.type  => LocalDecisionTreeClassificationModel
      case _: MultilayerPerceptronClassificationModel.type => LocalMultilayerPerceptronClassificationModel
      case _: NaiveBayes.type => LocalNaiveBayes
      case _: RandomForestClassificationModel.type => LocalRandomForestClassificationModel
      case _: LogisticRegressionModel.type => LocalLogisticRegressionModel

        // Clustering models
      case _: GaussianMixtureModel.type => LocalGaussianMixtureModel
      case _: KMeansModel.type  => LocalKMeansModel

        // Preprocessing
      case _: Binarizer.type => LocalBinarizer
      case _: CountVectorizerModel.type => LocalCountVectorizerModel
      case _: DCT.type => LocalDCT
      case _: HashingTF.type => LocalHashingTF
      case _: IndexToString.type => LocalIndexToString
      case _: MaxAbsScalerModel.type => LocalMaxAbsScalerModel
      case _: MinMaxScalerModel.type => LocalMinMaxScalerModel
      case _: NGram.type => LocalNGram
      case _: Normalizer.type => LocalNormalizer
      case _: OneHotEncoder.type => LocalOneHotEncoder
      case _: PCAModel.type => LocalPCAModel
      case _: PolynomialExpansion.type => LocalPolynomialExpansion
      case _: StandardScalerModel.type => LocalStandardScalerModel
      case _: StopWordsRemover.type  => LocalStopWordsRemover
      case _: StringIndexerModel.type => LocalStringIndexerModel
      case _: Tokenizer.type  => LocalTokenizer
      case _: VectorIndexerModel.type => LocalVectorIndexerModel
      case _: Word2VecModel.type => LocalWord2VecModel
      case _: IDFModel.type => LocalIDF
      case _: NaiveBayesModel.type => LocalNaiveBayes
      case _: ChiSqSelectorModel.type => LocalChiSqSelectorModel

        // Regression
      case _: DecisionTreeRegressionModel.type => LocalDecisionTreeRegressionModel
      case _: LinearRegressionModel.type => LocalLinearRegressionModel
      case _: RandomForestRegressionModel.type => LocalRandomForestRegressionModel
      case _: GBTRegressionModel.type => LocalGBTRegressor

      case _: KMeans.type => LocalKMeansModel
      case _ => throw new Exception(s"Unknown model: ${m.getClass}")
    }
  }
}
