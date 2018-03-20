package io.hydrosphere.spark_ml_serving.common

case class Metadata(
  `class`: String,
  timestamp: Long,
  sparkVersion: String,
  uid: String,
  paramMap: Map[String, Any],
  numFeatures: Option[Int] = None,
  numClasses: Option[Int]  = None,
  numTrees: Option[Int]    = None
) {
  def getAs[T](name: String): Option[T] = {
    paramMap.get(name).map(_.asInstanceOf[T])
  }

  def inputCol = getAs[String]("inputCol")

  def outputCol = getAs[String]("outputCol")
}

object Metadata {

  import org.json4s.DefaultFormats
  import org.json4s.jackson.JsonMethods._

  implicit val formats = DefaultFormats

  def fromJson(json: String): Metadata = {
    parse(json).extract[Metadata]
  }
}
