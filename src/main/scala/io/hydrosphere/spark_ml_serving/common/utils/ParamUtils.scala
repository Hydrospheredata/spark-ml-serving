package io.hydrosphere.spark_ml_serving.common.utils

import io.hydrosphere.spark_ml_serving.common.Metadata
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Param

object ParamUtils {
  def set[TransformerT <: Transformer, ParamT](transformer: TransformerT, param: Param[ParamT], metadata: Metadata): TransformerT = {
    transformer.set(param, extract(param, metadata))
  }

  def extract[T](param: Param[T], metadata: Metadata): T = {
    metadata.getAs[Any](param.name).getOrElse(throw new IllegalArgumentException(param.name)) match {
      case p: BigInt => p.intValue().asInstanceOf[T]
      case p => p.asInstanceOf[T]
    }
  }
}
