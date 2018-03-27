package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.feature.MaxAbsScalerModel
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}

class LocalMaxAbsScalerModel(override val sparkTransformer: MaxAbsScalerModel)
  extends LocalTransformer[MaxAbsScalerModel] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val maxAbsUnzero =
          Vectors.dense(sparkTransformer.maxAbs.toArray.map(x => if (x == 0) 1 else x))
        val newData = column.data.map(r => {
          val vec = r match {
            case d: Seq[Number @unchecked] if d.isInstanceOf[Seq[Number]] => d.map(_.doubleValue())
            case d =>
              throw new IllegalArgumentException(s"Unknown data type for LocalMaxAbsScaler: $d")
          }
          val brz = DataUtils.asBreeze(vec.toArray) / DataUtils.asBreeze(maxAbsUnzero.toArray)
          DataUtils.fromBreeze(brz).toList
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalMaxAbsScalerModel
  extends SimpleModelLoader[MaxAbsScalerModel]
  with TypedTransformerConverter[MaxAbsScalerModel] {
  override def build(metadata: Metadata, data: LocalData): MaxAbsScalerModel = {
    val maxAbsParams = data.column("maxAbs").get.data.head.asInstanceOf[Map[String, Any]]
    val maxAbs = DataUtils.constructVector(maxAbsParams)

    val constructor =
      classOf[MaxAbsScalerModel].getDeclaredConstructor(classOf[String], classOf[Vector])
    constructor.setAccessible(true)
    constructor
      .newInstance(metadata.uid, maxAbs)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
  }

  override implicit def toLocal(
    transformer: MaxAbsScalerModel
  ): LocalMaxAbsScalerModel = new LocalMaxAbsScalerModel(transformer)
}
