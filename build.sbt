organization := "io.hydrosphere"
name := "spark-ml-serving"
scalaVersion := "2.11.8"
version := "0.1.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-mllib" % "2.0.0",

  "org.json4s" %% "json4s-native" % "3.2.10",

  "com.twitter" % "parquet-hadoop-bundle" % "1.6.0",
  "org.apache.parquet" % "parquet-common" % "1.7.0",
  "org.apache.parquet" % "parquet-column" % "1.7.0",
  "org.apache.parquet" % "parquet-hadoop" % "1.7.0",
  "org.apache.parquet" % "parquet-avro" % "1.7.0",

  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)

