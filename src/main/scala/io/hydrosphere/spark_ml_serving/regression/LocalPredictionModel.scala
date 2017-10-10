package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.DataUtils._
import io.hydrosphere.spark_ml_serving.{LocalData, LocalDataColumn, LocalTransformer}
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.ClassTag

abstract class LocalPredictionModel[T <: PredictionModel[Vector, T]](implicit ct: ClassTag[T]) extends LocalTransformer[T] {
  override def transform(localData: LocalData): LocalData = {
    val cls = ct.runtimeClass

    val predict = cls.getMethod("predict", classOf[Vector])
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val predictionCol = LocalDataColumn(
          sparkTransformer.getPredictionCol,
          column.data.mapToMlVectors.map { vector =>
            predict.invoke(sparkTransformer, vector).asInstanceOf[Double]
          }
        )
        localData.withColumn(predictionCol)
      case None => localData
    }
  }
}
