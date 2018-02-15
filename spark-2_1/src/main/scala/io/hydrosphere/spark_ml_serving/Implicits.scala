package io.hydrosphere.spark_ml_serving

object Implicits {
  implicit val loaders = LoaderConversions
  implicit val transformers = TransformerConversions
}
