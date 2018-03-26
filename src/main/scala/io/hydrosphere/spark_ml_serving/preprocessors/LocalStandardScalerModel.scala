package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.feature.{StandardScalerModel => OldStandardScalerModel}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}

class LocalStandardScalerModel(override val sparkTransformer: StandardScalerModel)
  extends LocalTransformer[StandardScalerModel] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val scaler = new OldStandardScalerModel(
          OldVectors.fromML(sparkTransformer.std),
          OldVectors.fromML(sparkTransformer.mean),
          sparkTransformer.getWithStd,
          sparkTransformer.getWithMean
        )

        val newData = column.data.mapToMlLibVectors.map(scaler.transform(_).toList)
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalStandardScalerModel
  extends SimpleModelLoader[StandardScalerModel]
  with TypedTransformerConverter[StandardScalerModel] {

  override def build(metadata: Metadata, data: LocalData): StandardScalerModel = {
    val constructor = classOf[StandardScalerModel].getDeclaredConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Vector]
    )
    constructor.setAccessible(true)

    val stdParams = data.column("std").get.data.head.asInstanceOf[Map[String, Any]]
    val std = DataUtils.constructVector(stdParams)

    val meanParams = data.column("mean").get.data.head.asInstanceOf[Map[String, Any]]
    val mean = DataUtils.constructVector(meanParams)
    constructor
      .newInstance(metadata.uid, std, mean)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
  }

  override implicit def toLocal(
    transformer: StandardScalerModel
  ) = new LocalStandardScalerModel(transformer)
}
