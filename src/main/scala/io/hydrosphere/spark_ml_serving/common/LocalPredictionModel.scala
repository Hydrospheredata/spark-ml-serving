package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.ClassTag

abstract class LocalPredictionModel[T <: PredictionModel[Vector, T]](implicit ct: ClassTag[T]) extends LocalTransformer[T] {
  def predict(v: List[Double]): Double = invoke[Double]('predict, v)

  override def transform(localData: LocalData): LocalData = {
    val cls = ct.runtimeClass

    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val predictionCol = LocalDataColumn(
          sparkTransformer.getPredictionCol,
          column.data.map(_.asInstanceOf[List[Double]]).map(predict)
        )
        localData.withColumn(predictionCol)
      case None => localData
    }
  }
}
