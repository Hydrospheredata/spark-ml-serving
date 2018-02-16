package io.hydrosphere.spark_ml_serving.clustering

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.mllib.clustering.{KMeansModel => OldKMeansModel}
import org.apache.spark.mllib.linalg.{Vector => MLlibVec}

import scala.reflect.runtime.universe

class LocalKMeansModel(override val sparkTransformer: KMeansModel)
  extends LocalTransformer[KMeansModel] {
  lazy val parent: OldKMeansModel = {
    val mirror     = universe.runtimeMirror(sparkTransformer.getClass.getClassLoader)
    val parentTerm = universe.typeOf[KMeansModel].decl(universe.TermName("parentModel")).asTerm
    mirror.reflect(sparkTransformer).reflectField(parentTerm).get.asInstanceOf[OldKMeansModel]
  }

  override def transform(localData: LocalData): LocalData = {
    import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._

    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val newColumn = LocalDataColumn(
          sparkTransformer.getPredictionCol,
          column.data.mapToMlLibVectors.map(x => parent.predict(x))
        )
        localData.withColumn(newColumn)
      case None => localData
    }
  }
}

object LocalKMeansModel
  extends SimpleModelLoader[KMeansModel]
  with TypedTransformerConverter[KMeansModel] {

  override def build(metadata: Metadata, data: LocalData): KMeansModel = {
    val mapRows = data.toMapList
    val centers = mapRows map { row =>
      val vec = DataUtils.constructVector(row("clusterCenter").asInstanceOf[Map[String, Any]])
      org.apache.spark.mllib.linalg.Vectors.fromML(vec)
    }
    val parentConstructor = classOf[OldKMeansModel].getDeclaredConstructor(classOf[Array[MLlibVec]])
    parentConstructor.setAccessible(true)
    val mlk = parentConstructor.newInstance(centers.toArray)

    val constructor =
      classOf[KMeansModel].getDeclaredConstructor(classOf[String], classOf[OldKMeansModel])
    constructor.setAccessible(true)
    var inst = constructor
      .newInstance(metadata.uid, mlk)
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])

    inst = inst.set(inst.k, metadata.paramMap("k").asInstanceOf[Number].intValue())
    inst = inst.set(inst.initMode, metadata.paramMap("initMode").asInstanceOf[String])
    inst = inst.set(inst.maxIter, metadata.paramMap("maxIter").asInstanceOf[Number].intValue())
    inst = inst.set(inst.initSteps, metadata.paramMap("initSteps").asInstanceOf[Number].intValue())
    inst = inst.set(inst.seed, metadata.paramMap("seed").toString.toLong)
    inst = inst.set(inst.tol, metadata.paramMap("tol").asInstanceOf[Double])
    inst
  }
  override implicit def toLocal(transformer: KMeansModel) =
    new LocalKMeansModel(transformer)
}
