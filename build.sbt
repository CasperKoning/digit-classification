name := "DigitClassification"

version := "1.0"

scalaVersion := "2.10.5"

val sparkCore =  "org.apache.spark" % "spark-core_2.10" % "1.4.0" % "provided"

val sparkMlLib = "org.apache.spark" % "spark-mllib_2.10" % "1.4.0"

libraryDependencies ++= Seq(
  sparkCore,
  sparkMlLib
)