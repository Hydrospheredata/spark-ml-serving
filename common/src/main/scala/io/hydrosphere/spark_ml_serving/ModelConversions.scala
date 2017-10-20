package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalModel
import org.apache.spark.ml.Transformer

trait ModelConversions {
  implicit def sparkToLocal[T <: Transformer](m: Any): LocalModel[T]
}

object ModelConversions extends ModelConversions {
  override implicit def sparkToLocal[T <: Transformer](m: Any): LocalModel[T] = ???
}