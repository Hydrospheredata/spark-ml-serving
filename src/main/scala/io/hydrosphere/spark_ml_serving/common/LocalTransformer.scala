package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common.utils.PrivateMethodExposer

trait LocalTransformer[SparkTransformer <: Transformer] {
  def sparkTransformer: SparkTransformer

  protected def invoke[Res](method: Symbol, vector: List[Double]): Res =
    PrivateMethodExposer(sparkTransformer)(method)(vector.toMlVector).asInstanceOf[Res]
  protected def invokeVec(method: Symbol, vector: List[Double]): List[Double] =
    invoke[Vector](method, vector).toList

  def transform(localData: LocalData): LocalData
}
