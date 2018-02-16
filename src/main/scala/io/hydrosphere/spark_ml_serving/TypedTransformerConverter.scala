package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalTransformer
import org.apache.spark.ml.Transformer

trait TypedTransformerConverter[T <: Transformer] {
  implicit def toLocal(sparkTransformer: T): LocalTransformer[T]
}
