package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common.classification.LocalClassificationModel
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.ml.linalg.Vector

class LocalLinearSVCModel(override val sparkTransformer: LinearSVCModel) extends LocalClassificationModel[LinearSVCModel]{
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        var result = localData

        sparkTransformer.get(sparkTransformer.rawPredictionCol).foreach{ name =>
          val res = LocalDataColumn(
            name,
            column.data.map(_.asInstanceOf[List[Double]]).map(predictRaw)
          )
          result = result.withColumn(res)
        }

        sparkTransformer.get(sparkTransformer.predictionCol).foreach{ name =>
          val res = LocalDataColumn(name, column.data.map(_.asInstanceOf[List[Double]]).map(predict))
          result = result.withColumn(res)
        }

        result
      case None => localData
    }
  }
}

object LocalLinearSVCModel extends LocalModel[LinearSVCModel] {
  override def load(metadata: Metadata, data: LocalData): LinearSVCModel = {
    val coefficients = DataUtils.constructVector(data.column("coefficients").get.data.head.asInstanceOf[Map[String, Any]])
    val cls = classOf[LinearSVCModel].getConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Double]
    )
    val inst = cls.newInstance(
      metadata.uid,
      coefficients,
      data.column("intercept").get.data.head.asInstanceOf[java.lang.Double]
    )
    inst
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setRawPredictionCol(metadata.paramMap("rawPredictionCol").asInstanceOf[String])

      .set(inst.labelCol, metadata.paramMap("labelCol").toString)
      .set(inst.aggregationDepth , metadata.paramMap("aggregationDepth").toString.toInt)
      .set(inst.fitIntercept, metadata.paramMap("fitIntercept").toString.toBoolean)
      .set(inst.maxIter, metadata.paramMap("maxIter").toString.toInt)
      .set(inst.regParam, metadata.paramMap("regParam").toString.toDouble)
      .set(inst.standardization, metadata.paramMap("standardization").toString.toBoolean)
      .set(inst.threshold, metadata.paramMap("threshold").toString.toDouble)
      .set(inst.tol, metadata.paramMap("tol").toString.toDouble)
  }

  override implicit def getTransformer(transformer: LinearSVCModel): LocalTransformer[LinearSVCModel] = new LocalLinearSVCModel(transformer)
}