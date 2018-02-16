package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.RegexTokenizer

class LocalRegexTokenizer(val sparkTransformer: RegexTokenizer)
  extends LocalTransformer[RegexTokenizer] {

  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(x) =>
        val newData = x.data.map { d =>
          val originStr = d.toString
          val re        = sparkTransformer.getPattern.r
          val str       = if (sparkTransformer.getToLowercase) originStr.toLowerCase() else originStr
          val tokens =
            if (sparkTransformer.getGaps) re.split(str).toSeq else re.findAllIn(str).toSeq
          val minLength = sparkTransformer.getMinTokenLength
          tokens.filter(_.length >= minLength).toList
        }
        localData.withColumn(
          LocalDataColumn(
            sparkTransformer.getOutputCol,
            newData
          )
        )
      case None => localData
    }
  }
}

object LocalRegexTokenizer
  extends SimpleModelLoader[RegexTokenizer]
  with TypedTransformerConverter[RegexTokenizer] {

  override def build(metadata: Metadata, data: LocalData): RegexTokenizer = {
    new RegexTokenizer(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setPattern(metadata.paramMap("pattern").asInstanceOf[String])
      .setGaps(metadata.paramMap("gaps").asInstanceOf[Boolean])
      .setMinTokenLength(metadata.paramMap("minTokenLength").asInstanceOf[Number].intValue())
      .setToLowercase(metadata.paramMap("toLowercase").asInstanceOf[Boolean])
  }

  override implicit def toLocal(transformer: RegexTokenizer) =
    new LocalRegexTokenizer(transformer)
}
