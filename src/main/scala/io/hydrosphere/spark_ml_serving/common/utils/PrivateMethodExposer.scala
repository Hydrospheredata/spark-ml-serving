package io.hydrosphere.spark_ml_serving.common.utils

case class PrivateMethodExposer(x: AnyRef) {
  def apply(method: scala.Symbol): PrivateMethodCaller = new PrivateMethodCaller(x, method.name)
}
