package io.hydrosphere.spark_ml_serving.common.classification

import scala.reflect.runtime.{universe => ru}
import io.hydrosphere.spark_ml_serving.common.{LocalData, LocalDataColumn}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.ClassTag

abstract class LocalProbabilisticClassificationModel[T <: ProbabilisticClassificationModel[Vector, T]](implicit ct: ClassTag[T]) extends LocalClassificationModel[T]{
  def raw2probabilityInPlace(vector: List[Double]): List[Double] = invokeVec('raw2probabilityInPlace, vector)
  def raw2probability(vector: List[Double]): List[Double] = raw2probabilityInPlace(vector)
  def raw2prediction(vector: List[Double]): Double = invoke[Double]('raw2prediction, vector)
  def probability2prediction(vector: List[Double]): Double = invoke[Double]('probability2prediction, vector)

  override def transform(localData: LocalData) = {
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        sparkTransformer.get(sparkTransformer.thresholds).foreach(t => require(t.length == sparkTransformer.numClasses))
        var result = localData

        val rawCol = sparkTransformer.get(sparkTransformer.rawPredictionCol).map { name =>
          val res = LocalDataColumn(
            name,
            column.data.map(_.asInstanceOf[List[Double]]).map(predictRaw)
          )
          result = result.withColumn(res)
          res
        }

        val probCol = sparkTransformer.get(sparkTransformer.probabilityCol).map { name =>
          val data = rawCol match {
            case Some(raw) => raw.data.map(_.asInstanceOf[List[Double]]).map(raw2probability)
            case None => column.data.map(_.asInstanceOf[List[Double]]).map(predictRaw)
          }
          val res = LocalDataColumn(name, data)
          result = result.withColumn(res)
          res
        }

        sparkTransformer.get(sparkTransformer.predictionCol).map { name =>
          val data = rawCol match {
            case Some(raw) => raw.data.map(raw2prediction)
            case None => probCol match {
              case Some(prob) => prob.data.map(probability2prediction)
              case None => column.data.map(_.asInstanceOf[List[Double]]).map(predict)
            }
          }
          val res = LocalDataColumn(name, data)
          result = result.withColumn(res)
          res
        }

        result
      case None => localData
    }
  }
}
