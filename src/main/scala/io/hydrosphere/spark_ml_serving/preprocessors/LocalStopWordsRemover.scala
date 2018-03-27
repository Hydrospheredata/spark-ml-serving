package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.StopWordsRemover

class LocalStopWordsRemover(override val sparkTransformer: StopWordsRemover)
  extends LocalTransformer[StopWordsRemover] {
  override def transform(localData: LocalData): LocalData = {
    val stopWordsSet   = sparkTransformer.getStopWords
    val toLower        = (s: String) => if (s != null) s.toLowerCase else s
    val lowerStopWords = stopWordsSet.map(toLower(_)).toSet

    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val newData = column.data.map(r => {
          if (sparkTransformer.getCaseSensitive) {
            r.asInstanceOf[Seq[String]].filter(s => !stopWordsSet.contains(s))
          } else {
            r.asInstanceOf[Seq[String]].filter(s => !lowerStopWords.contains(toLower(s)))
          }
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalStopWordsRemover
  extends SimpleModelLoader[StopWordsRemover]
  with TypedTransformerConverter[StopWordsRemover] {

  override def build(metadata: Metadata, data: LocalData): StopWordsRemover = {
    new StopWordsRemover(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setCaseSensitive(metadata.paramMap("caseSensitive").asInstanceOf[Boolean])
      .setStopWords(metadata.paramMap("stopWords").asInstanceOf[Seq[String]].toArray)
  }

  override implicit def toLocal(transformer: StopWordsRemover) =
    new LocalStopWordsRemover(transformer)
}
