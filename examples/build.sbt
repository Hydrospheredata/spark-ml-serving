scalaVersion := "2.11.8"

lazy val sparkVersion = "2.0.2"
lazy val localSparkVersion = sparkVersion.substring(0,sparkVersion.lastIndexOf(".")).replace('.', '_')

libraryDependencies ++= Seq(
  "io.hydrosphere" %% s"spark-ml-serving-$localSparkVersion" % "0.3.1",
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)
