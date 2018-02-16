package io.hydrosphere.spark_ml_serving.common.utils

class PrivateMethodCaller(x: AnyRef, methodName: String) {
  def apply(_args: Any*): Any = {
    def _parents: Stream[Class[_]] = Stream(x.getClass) #::: _parents.map(_.getSuperclass)
    val args                       = _args.map(_.asInstanceOf[AnyRef])
    val parents                    = _parents.takeWhile(_ != null).toList
    val methods                    = parents.flatMap(_.getDeclaredMethods)
    val method = methods
      .find(_.getName == methodName)
      .getOrElse(throw new IllegalArgumentException("Method " + methodName + " not found"))
    method.setAccessible(true)
    method.invoke(x, args: _*)
  }
}
