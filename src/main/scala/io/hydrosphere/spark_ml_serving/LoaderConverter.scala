package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalModelConverter

trait LoaderConverter {
  implicit def sparkToLocal(m: Any)(implicit conv: TransformerConverter): LocalModelConverter
}
