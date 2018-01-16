lazy val sparkVersion: SettingKey[String] = settingKey[String]("Spark version")
lazy val versionRegex = "(\\d+)\\.(\\d+).*".r

lazy val commonSettings = Seq(
  organization := "io.hydrosphere",
  version := "0.2.1",
  scalaVersion := "2.11.8",
  sparkVersion := util.Properties.propOrElse("sparkVersion", "2.2.0")
)

lazy val commonDependencies = Seq(
  "org.json4s" %% "json4s-native" % "3.2.10",
  "org.scalactic" %% "scalactic" % "3.0.3" % "test",
  "org.scalatest" %% "scalatest" % "3.0.3" % "test"
)

lazy val common = project.in(file("common"))
  .settings(commonSettings)
  .settings(publishSettings: _*)
  .settings(
    name := "spark-ml-serving-common",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-mllib" % sparkVersion.value % "provided",

      "org.json4s" %% "json4s-native" % "3.2.10",

      "com.twitter" % "parquet-hadoop-bundle" % "1.6.0",
      "org.apache.parquet" % "parquet-common" % "1.7.0",
      "org.apache.parquet" % "parquet-column" % "1.7.0",
      "org.apache.parquet" % "parquet-hadoop" % "1.7.0",
      "org.apache.parquet" % "parquet-avro" % "1.7.0"
    )
  )


lazy val spark_20 = project.in(file("spark-2_0"))
  .settings(commonSettings)
  .dependsOn(common)
  .settings(publishSettings: _*)
  .settings(
    name := "spark-ml-serving-2_0",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-mllib" % "2.0.2" % "provided"
    ) ++ commonDependencies
  )

lazy val spark_21 = project.in(file("spark-2_1"))
  .settings(commonSettings)
  .dependsOn(common)
  .settings(publishSettings: _*)
  .settings(
    name := "spark-ml-serving-2_1",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-mllib" % "2.1.2" % "provided"
    ) ++ commonDependencies
  )

lazy val spark_22 = project.in(file("spark-2_2"))
  .settings(commonSettings)
  .dependsOn(common)
  .settings(publishSettings: _*)
  .settings(
    name := "spark-ml-serving-2_2",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided"
    ) ++ commonDependencies
  )

lazy val examples = project.in(file("examples"))
  .settings(commonSettings)
  .dependsOn(spark_22)
  .settings(
    name := "spark-ml-serving-examples"
  )

lazy val publishSettings = Seq(
    publishMavenStyle := true,
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots/")
      else
        Some("releases"  at nexus + "service/local/staging/deploy/maven2/")
    },
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false },

    pomExtra := <url>https://github.com/Hydrospheredata/spark-ml-serving</url>
      <licenses>
        <license>
          <name>Apache 2.0 License</name>
          <url>https://github.com/Hydrospheredata/spark-ml-serving/blob/master/LICENSE</url>
          <distribution>repo</distribution>
        </license>
      </licenses>
      <scm>
        <url>https://github.com/Hydrospheredata/mist.git</url>
        <connection>https://github.com/Hydrospheredata/mist.git</connection>
      </scm>
      <developers>
        <developer>
          <id>mkf-simpson</id>
          <name></name>
          <url>https://github.com/mkf-simpson</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>leonid133</id>
          <name>Leonid Blokhin</name>
          <url>https://github.com/leonid133</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>KineticCookie</id>
          <name>Bulat Lutfullin</name>
          <url>https://github.com/KineticCookie</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>dos65</id>
          <name>Vadim Chelyshov</name>
          <url>https://github.com/dos65</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
      </developers>
)
