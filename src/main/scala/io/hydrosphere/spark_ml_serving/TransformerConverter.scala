package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalTransformable
import org.apache.spark.ml.Transformer

trait TransformerConverter {
  implicit def transformerToLocal(m: Transformer): LocalTransformable
}
