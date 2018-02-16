package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.OneHotEncoder

class LocalOneHotEncoder(override val sparkTransformer: OneHotEncoder)
  extends LocalTransformer[OneHotEncoder] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val col = column.data match {
          case d: List[Double] => d
          case d: List[Int]    => d.map(_.toDouble)
          case x               => throw new IllegalArgumentException(s"Incorrect index value: $x")
        }
        col.foreach(
          x =>
            assert(
              x >= 0.0 && x == x.toInt,
              s"Values from column ${sparkTransformer.getInputCol} must be indices, but got $x."
          )
        )

        val size = col.max.toInt
        val newData = col.map(r => {
          val res = Array.fill(size) { 0.0 }
          if (r < size) {
            res.update(r.toInt, 1.0)
          }
          res.toList
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalOneHotEncoder
  extends SimpleModelLoader[OneHotEncoder]
  with TypedTransformerConverter[OneHotEncoder] {

  override def build(metadata: Metadata, data: LocalData): OneHotEncoder = {
    var ohe = new OneHotEncoder(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])

    metadata.paramMap.get("dropLast").foreach { x =>
      ohe = ohe.setDropLast(x.asInstanceOf[Boolean])
    }
    ohe
  }

  override implicit def toLocal(transformer: OneHotEncoder) =
    new LocalOneHotEncoder(transformer)
}
