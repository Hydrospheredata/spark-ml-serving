package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.{Vector, Vectors}

class LocalPolynomialExpansion(override val sparkTransformer: PolynomialExpansion)
  extends LocalTransformer[PolynomialExpansion] {

  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val method = classOf[PolynomialExpansion].getMethod("createTransformFunc")
        val newData = column.data.map(r => {
          val row            = r.asInstanceOf[List[Any]].map(_.toString.toDouble).toArray
          val vector: Vector = Vectors.dense(row)
          method.invoke(sparkTransformer).asInstanceOf[Vector => Vector](vector).toList
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalPolynomialExpansion
  extends SimpleModelLoader[PolynomialExpansion]
  with TypedTransformerConverter[PolynomialExpansion] {

  override def build(metadata: Metadata, data: LocalData): PolynomialExpansion = {
    new PolynomialExpansion(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setDegree(metadata.paramMap("degree").asInstanceOf[Number].intValue())
  }

  override implicit def toLocal(
    transformer: PolynomialExpansion
  ) = new LocalPolynomialExpansion(transformer)
}
