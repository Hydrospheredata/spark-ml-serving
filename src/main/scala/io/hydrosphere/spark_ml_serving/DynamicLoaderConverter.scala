package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.ModelLoader

trait DynamicLoaderConverter {
  implicit def sparkToLocal(m: Any): ModelLoader[_]
}
