package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common.{LocalModel, LocalTransformer, Metadata}
import io.hydrosphere.spark_ml_serving.common.LocalPredictionModel
import org.apache.spark.ml.classification.GBTClassificationModel

class LocalGBTClassificationModel(override val sparkTransformer: GBTClassificationModel)
  extends LocalPredictionModel[GBTClassificationModel] { }

object LocalGBTClassificationModel extends LocalModel[GBTClassificationModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): GBTClassificationModel = {
    // TODO
    ???
  }

  override implicit def getTransformer(transformer: GBTClassificationModel): LocalTransformer[GBTClassificationModel] = {
    new LocalGBTClassificationModel(transformer)
  }
}
