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
)

object Metadata {

  import org.json4s.DefaultFormats
  import org.json4s.jackson.JsonMethods._

  implicit val formats = DefaultFormats

  def fromJson(json: String): Metadata = {
    parse(json).extract[Metadata]
  }
}
