package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.Tokenizer

class LocalTokenizer(override val sparkTransformer: Tokenizer) extends LocalTransformer[Tokenizer] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val method = classOf[Tokenizer].getMethod("createTransformFunc")
        val newData = column.data.map(_.asInstanceOf[String]).map(s => {
          method.invoke(sparkTransformer).asInstanceOf[String => Seq[String]](s)
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalTokenizer extends LocalModel[Tokenizer] {
  override def load(metadata: Metadata, data: LocalData): Tokenizer = {
    new Tokenizer(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: Tokenizer): LocalTransformer[Tokenizer] = new LocalTokenizer(transformer)
}
