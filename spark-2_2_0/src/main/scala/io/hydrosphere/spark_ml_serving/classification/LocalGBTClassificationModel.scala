package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.classification.GBTClassificationModel

class LocalGBTClassificationModel(override val sparkTransformer: GBTClassificationModel)
  extends LocalPredictionModel[GBTClassificationModel] { }

object LocalGBTClassificationModel extends LocalModel[GBTClassificationModel] {
  override def load(metadata: Metadata, data: LocalData): GBTClassificationModel = {
    // TODO
    ???
  }

  override implicit def getTransformer(transformer: GBTClassificationModel): LocalTransformer[GBTClassificationModel] = {
    new LocalGBTClassificationModel(transformer)
  }
}
