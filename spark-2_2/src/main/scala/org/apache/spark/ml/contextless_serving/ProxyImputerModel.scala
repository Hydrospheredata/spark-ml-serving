package org.apache.spark.ml.contextless_serving

import io.hydrosphere.spark_ml_serving.common.LocalData
import org.apache.spark.ml.feature.ImputerModel

/**
  * Proxy implementation of ImputerModel in order to hide DataFrame
  * @param imputerModel target imputer model
  * @param surrogate surrogate local df
  */
case class ProxyImputerModel(imputerModel: ImputerModel, surrogate: LocalData) extends ImputerModel(imputerModel.uid, null)
