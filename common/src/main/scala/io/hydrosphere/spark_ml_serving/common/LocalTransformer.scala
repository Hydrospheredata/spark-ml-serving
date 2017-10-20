package io.hydrosphere.spark_ml_serving.common

import java.lang.reflect.Method

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.Transformer

import scala.reflect.ClassTag

/**
  * Created by bulat on 22.03.17.
  */
abstract class LocalTransformer[T <: Transformer](implicit ct: ClassTag[T]) {
  import io.hydrosphere.spark_ml_serving.common.DataUtils._

  protected val cls = ct.runtimeClass

  protected def invoke[Res](method: Symbol, vector: List[Double]): Res = PrivateMethodExposer(sparkTransformer)(method)(vector.toMlVector).asInstanceOf[Res]
  protected def invokeVec(method: Symbol, vector: List[Double]): List[Double] = invoke[Vector](method, vector).toList

  val sparkTransformer: T
  def transform(localData: LocalData): LocalData
}