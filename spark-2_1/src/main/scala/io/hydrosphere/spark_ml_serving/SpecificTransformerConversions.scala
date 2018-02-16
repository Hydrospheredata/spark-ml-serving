package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.classification._
import io.hydrosphere.spark_ml_serving.common.LocalTransformer
import io.hydrosphere.spark_ml_serving.preprocessors.LocalWord2VecModel
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.Word2VecModel

object SpecificTransformerConversions extends DynamicTransformerConverter {

  implicit def transformerToLocal(transformer: Transformer): LocalTransformer[_] = {
    transformer match {
      case x: LogisticRegressionModel => new LocalLogisticRegressionModel(x)
      case x: Word2VecModel           => new LocalWord2VecModel(x)

      case x => throw new Exception(s"Unknown model: ${x.getClass}")
    }
  }
}
