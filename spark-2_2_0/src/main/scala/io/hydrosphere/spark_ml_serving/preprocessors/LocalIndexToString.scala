package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.SparkException
import org.apache.spark.ml.feature.IndexToString

class LocalIndexToString(override val sparkTransformer: IndexToString) extends LocalTransformer[IndexToString] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val labels = sparkTransformer.getLabels
        val indexer = (index: Double) => {
          val idx = index.toInt
          if (0 <= idx && idx < labels.length) {
            labels(idx)
          } else {
            throw new SparkException(s"Unseen index: $index ??")
          }
        }
        val newColumn = LocalDataColumn(sparkTransformer.getOutputCol, column.data map {
          case i: Int => indexer(i.toDouble)
          case d: Double => indexer(d)
          case d => throw new IllegalArgumentException(s"Unknown data to index: $d")
        })
        localData.withColumn(newColumn)
      case None => localData
    }
  }
}

object LocalIndexToString extends LocalModel[IndexToString] {
  override def load(metadata: Metadata, data: LocalData): IndexToString = {
    val ctor = classOf[IndexToString].getDeclaredConstructor(classOf[String])
    ctor.setAccessible(true)
    ctor
      .newInstance(metadata.uid)
      .setLabels(metadata.paramMap("labels").asInstanceOf[List[String]].to[Array])
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: IndexToString): LocalTransformer[IndexToString] = new LocalIndexToString(transformer)
}
