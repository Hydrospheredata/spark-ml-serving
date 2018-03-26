package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.ChiSqSelectorModel
import org.apache.spark.mllib.feature.{ChiSqSelectorModel => OldChiSqSelectorModel}

class LocalChiSqSelectorModel(override val sparkTransformer: ChiSqSelectorModel)
  extends LocalTransformer[ChiSqSelectorModel] {
  lazy val parent: OldChiSqSelectorModel = {
    val field = sparkTransformer.getClass.getDeclaredField(
      "org$apache$spark$ml$feature$ChiSqSelectorModel$$chiSqSelector"
    )
    field.setAccessible(true)
    field.get(sparkTransformer).asInstanceOf[OldChiSqSelectorModel]
  }

  override def transform(localData: LocalData): LocalData = {
    import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._

    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val newColumn = LocalDataColumn(
          sparkTransformer.getOutputCol,
          column.data.mapToMlLibVectors.map { x =>
            val res = parent.transform(x).toList
            res
          }
        )
        localData.withColumn(newColumn)
      case None => localData
    }
  }
}

object LocalChiSqSelectorModel
  extends SimpleModelLoader[ChiSqSelectorModel]
  with TypedTransformerConverter[ChiSqSelectorModel] {

  override def build(metadata: Metadata, data: LocalData): ChiSqSelectorModel = {
    val parentConstructor =
      classOf[OldChiSqSelectorModel].getDeclaredConstructor(classOf[Array[Int]])
    parentConstructor.setAccessible(true)
    val selectedFeatures = data.column("selectedFeatures").get.data.head.asInstanceOf[Seq[Int]]
    val mlk              = parentConstructor.newInstance(selectedFeatures.toArray)

    val constructor = classOf[ChiSqSelectorModel]
      .getDeclaredConstructor(classOf[String], classOf[OldChiSqSelectorModel])
    constructor.setAccessible(true)
    var inst = constructor
      .newInstance(metadata.uid, mlk)
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])

    inst = inst.set(inst.labelCol, metadata.paramMap("labelCol").asInstanceOf[String])
    inst = inst.set(inst.numTopFeatures, metadata.paramMap("numTopFeatures").toString.toInt)
    inst
  }

  override implicit def toLocal(transformer: ChiSqSelectorModel) =
    new LocalChiSqSelectorModel(transformer)
}
