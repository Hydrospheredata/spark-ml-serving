package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.common.{LocalData, LocalModel, LocalTransformer, Metadata}
import org.apache.spark.ml.feature.ImputerModel

class LocalImputerModel(override val sparkTransformer: ImputerModel ) extends LocalTransformer[ImputerModel]{
  override def transform(localData: LocalData): LocalData = {
    // TODO
    ???
  }
}

object LocalImputerModel extends LocalModel[ImputerModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): ImputerModel = {
    // TODO
    ???
  }

  override implicit def getTransformer(transformer: ImputerModel): LocalTransformer[ImputerModel] = new LocalImputerModel(transformer)
}
