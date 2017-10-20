package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common.classification.LocalClassificationModel
import io.hydrosphere.spark_ml_serving.common.{LocalModel, LocalTransformer, Metadata}
import org.apache.spark.ml.classification.LinearSVCModel

class LocalLinearSVCModel(override val sparkTransformer: LinearSVCModel) extends LocalClassificationModel[LinearSVCModel]{

}

object LocalLinearSVCModel extends LocalModel[LinearSVCModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): LinearSVCModel = {
    // TODO
    ???
  }

  override implicit def getTransformer(transformer: LinearSVCModel): LocalTransformer[LinearSVCModel] = new LocalLinearSVCModel(transformer)
}