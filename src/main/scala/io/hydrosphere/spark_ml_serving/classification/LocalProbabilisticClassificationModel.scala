package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.DataUtils._
import io.hydrosphere.spark_ml_serving.{LocalData, LocalDataColumn, LocalTransformer}
import org.apache.spark.ml.classification.ProbabilisticClassificationModel
import org.apache.spark.ml.linalg.Vector

import scala.reflect.ClassTag

abstract class LocalProbabilisticClassificationModel[T <: ProbabilisticClassificationModel[Vector, T]](implicit ct: ClassTag[T]) extends LocalTransformer[T]{
  def transform(localData: LocalData) = {
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        sparkTransformer.get(sparkTransformer.thresholds).foreach(t => require(t.length == sparkTransformer.numClasses))

        var result = localData
        val cls = ct.runtimeClass

        val rawCol = sparkTransformer.get(sparkTransformer.rawPredictionCol).map { name =>
          val predictRaw = cls.getDeclaredMethod("predictRaw", classOf[Vector])
          val res = LocalDataColumn(
            name,
            column.data.mapToMlVectors.map {
              predictRaw.invoke(sparkTransformer, _).asInstanceOf[Vector].toList
            }
          )
          result = result.withColumn(res)
          res
        }


        val probCol = sparkTransformer.get(sparkTransformer.probabilityCol).map { name =>
          val data = rawCol match {
            case Some(raw) =>
              val raw2probabilityInPlace = cls.getDeclaredMethod("raw2probabilityInPlace", classOf[Vector])
              raw.data.mapToMlVectors.map { vector =>
                raw2probabilityInPlace.invoke(sparkTransformer, vector.copy).asInstanceOf[Vector].toList
              }
            case None =>
              val predictRaw = cls.getDeclaredMethod("predictRaw", classOf[Vector])
              column.data.mapToMlVectors.map {
                predictRaw.invoke(sparkTransformer, _).asInstanceOf[Vector].toList
              }
          }
          val res = LocalDataColumn(name, data)
          result = result.withColumn(res)
          res
        }

        val predCol = sparkTransformer.get(sparkTransformer.predictionCol).map { name =>
          val data = rawCol match {
            case Some(raw) =>
              val raw2predict = cls.getMethod("raw2prediction", classOf[Vector])
              raw.data.map { l =>
                raw2predict.invoke(sparkTransformer, l.toMlVector).asInstanceOf[Double]
              }
            case None => probCol match {
              case Some(prob) =>
                val prob2predict = cls.getMethod("probability2prediction", classOf[Vector])
                prob.data.map { l =>
                  prob2predict.invoke(sparkTransformer, l.toMlVector).asInstanceOf[Double]
                }
              case None =>
                val predict = cls.getMethod("predict", classOf[Vector])
                column.data.mapToMlVectors.map {
                  predict.invoke(sparkTransformer, _).asInstanceOf[Double]
                }
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
