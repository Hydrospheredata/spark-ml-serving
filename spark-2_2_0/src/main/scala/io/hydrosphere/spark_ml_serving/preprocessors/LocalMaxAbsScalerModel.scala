package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.common.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.MaxAbsScalerModel
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}

class LocalMaxAbsScalerModel(override val sparkTransformer: MaxAbsScalerModel) extends LocalTransformer[MaxAbsScalerModel] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val maxAbsUnzero = Vectors.dense(sparkTransformer.maxAbs.toArray.map(x => if (x == 0) 1 else x))
        val newData = column.data.map(r => {
          val vec: List[Double] = r match {
            case d: List[Double] => d
            case d: List[Int] => d.map(_.toDouble)
            case d => throw new IllegalArgumentException(s"Unknown data type for LocalMaxAbsScaler: $d")
          }
          val brz = DataUtils.asBreeze(vec.toArray) / DataUtils.asBreeze(maxAbsUnzero.toArray)
          DataUtils.fromBreeze(brz).toList
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalMaxAbsScalerModel extends LocalModel[MaxAbsScalerModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): MaxAbsScalerModel = {
    val maxAbsList = data("maxAbs").
      asInstanceOf[Map[String, Any]].
      getOrElse("values", List()).
      asInstanceOf[List[Double]].toArray
    val maxAbs = new DenseVector(maxAbsList)

    val constructor = classOf[MaxAbsScalerModel].getDeclaredConstructor(classOf[String], classOf[Vector])
    constructor.setAccessible(true)
    constructor
      .newInstance(metadata.uid, maxAbs)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: MaxAbsScalerModel): LocalTransformer[MaxAbsScalerModel] = new LocalMaxAbsScalerModel(transformer)
}
