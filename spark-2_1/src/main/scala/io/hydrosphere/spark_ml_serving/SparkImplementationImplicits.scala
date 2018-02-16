package io.hydrosphere.spark_ml_serving

trait SparkImplementationImplicits {
  implicit val loaders = SpecificLoaderConversions
  implicit val transformers = SpecificTransformerConversions
}

object SparkImplementationImplicits extends SparkImplementationImplicits