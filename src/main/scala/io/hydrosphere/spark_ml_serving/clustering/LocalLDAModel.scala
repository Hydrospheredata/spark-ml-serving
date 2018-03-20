package io.hydrosphere.spark_ml_serving.clustering

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.clustering.{LocalLDAModel => SparkLocalLDA}
import org.apache.spark.mllib.clustering.{LocalLDAModel => OldSparkLocalLDA}


class LocalLDAModel(override val sparkTransformer: SparkLocalLDA)
  extends LocalTransformer[SparkLocalLDA] {

  override def transform(localData: LocalData): LocalData = ???

}

object LocalLDAModel
  extends SimpleModelLoader[SparkLocalLDA]
  with TypedTransformerConverter[SparkLocalLDA] {

  override def build(metadata: Metadata, data: LocalData): SparkLocalLDA = {
    val topics = ???
    val gammaShape = ???
    val topicConcentration = ???
    val docConcentration = ???
    val vocabSize = ???
    val oldLDA = new OldSparkLocalLDA(topics, docConcentration, topicConcentration, gammaShape)

    val lda = new SparkLocalLDA(metadata.uid, vocabSize, oldLDA, null)

    // param optimizer
    // param keepLastCheckpoint
    // param seed
    // param subsamplingRate
    // param featuresCol
    // param learningDecay
    // param checkpointInterval
    // param learningOffset
    // param optimizeDocConcentration
    // param maxIter
    // param k

    // data topicConcentration
    // data gammaShape
    // data topicMatrix matrix
    // data vocabSize
    // data docConcentration vec
    ???
  }

  override implicit def toLocal(sparkTransformer: SparkLocalLDA): LocalTransformer[SparkLocalLDA] =
    new LocalLDAModel(sparkTransformer)

}
