package io.hydrosphere.spark_ml_serving.common.utils

import org.apache.spark.ml.Transformer

import scala.reflect.runtime.universe

class PumpedClass(classz: Class[_]) {
  def companion: Any = {
    val companionClassName = classz.getName + "$"
    val companionClass     = Class.forName(companionClassName)
    val moduleField        = companionClass.getField("MODULE$")
    moduleField.get(null)
  }
}

object PumpedClass {
  def companionFromClassName(className: String): Any = {
    val runtimeMirror = universe.runtimeMirror(this.getClass.getClassLoader)

    val module = runtimeMirror.staticModule(className + "$")
    val obj    = runtimeMirror.reflectModule(module)
    obj.instance
  }
}
