package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.feature.MinMaxScalerModel
import org.apache.spark.ml.linalg.{DenseVector, Vector}

class LocalMinMaxScalerModel(override val sparkTransformer: MinMaxScalerModel)
  extends LocalTransformer[MinMaxScalerModel] {
  override def transform(localData: LocalData): LocalData = {
    val originalRange =
      (DataUtils.asBreeze(sparkTransformer.originalMax.toArray) - DataUtils.asBreeze(
        sparkTransformer.originalMin.toArray
      )).toArray
    val minArray = sparkTransformer.originalMin.toArray
    val min      = sparkTransformer.getMin
    val max      = sparkTransformer.getMax

    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val newData = column.data.map(r => {
          val scale = max - min
          val vec = r match {
            case d: Seq[Number] if d.isInstanceOf[Seq[Number]] => d.map(_.doubleValue())
            case d =>
              throw new IllegalArgumentException(s"Unknown data type for LocalMinMaxScaler: $d")
          }
          val values = vec.toArray
          val size   = values.length
          var i      = 0
          while (i < size) {
            if (!values(i).isNaN) {
              val raw =
                if (originalRange(i) != 0) (values(i) - minArray(i)) / originalRange(i) else 0.5
              values.update(i, raw * scale + min)
            }
            i += 1
          }
          values.toList
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalMinMaxScalerModel
  extends SimpleModelLoader[MinMaxScalerModel]
  with TypedTransformerConverter[MinMaxScalerModel] {
  override def build(metadata: Metadata, data: LocalData): MinMaxScalerModel = {
    val originalMinList = data
      .column("originalMin")
      .get
      .data
      .head
      .asInstanceOf[Map[String, Any]]

    val originalMin = DataUtils.constructVector(originalMinList)

    val originalMaxList = data
      .column("originalMax")
      .get
      .data
      .head
      .asInstanceOf[Map[String, Any]]

    val originalMax = DataUtils.constructVector(originalMaxList)

    val constructor = classOf[MinMaxScalerModel].getDeclaredConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Vector]
    )
    constructor.setAccessible(true)
    constructor
      .newInstance(metadata.uid, originalMin, originalMax)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setMin(metadata.paramMap("min").toString.toDouble)
      .setMax(metadata.paramMap("max").toString.toDouble)
  }

  override implicit def toLocal(
    transformer: MinMaxScalerModel
  ) = new LocalMinMaxScalerModel(transformer)
}
