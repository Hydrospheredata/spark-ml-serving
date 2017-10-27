package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer

trait LocalModel[T <: Transformer] {
  def load(metadata: Metadata, data: LocalData): T
  implicit def getTransformer(transformer: T): LocalTransformer[T]
}
