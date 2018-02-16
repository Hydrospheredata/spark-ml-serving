package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalTransformer
import org.apache.spark.ml.Transformer

trait DynamicTransformerConverter {
  implicit def transformerToLocal(m: Transformer): LocalTransformer[_]
}
