package io.hydrosphere.spark_ml_serving

object ScalaUtils {
  def companionOf(classz: Class[_]) : Any ={
    val companionClassName = classz.getName + "$"
    val companionClass = Class.forName(companionClassName)
    val moduleField = companionClass.getField("MODULE$")
    moduleField.get(null)
  }
}
