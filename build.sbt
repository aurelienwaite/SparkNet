scalaVersion := "2.10.6"

scalacOptions ++= Seq("-feature")

// to regenerate the Java protobuf files, run:
// protoc -I=$SPARKNET_HOME/caffe/src/caffe/proto/ --java_out=$SPARKNET_HOME/src/main/scala/protobuf/' $SPARKNET_HOME/caffe/src/caffe/proto/caffe.proto

libraryDependencies += "com.google.protobuf" % "protobuf-java" % "2.5.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.0" % "provided"

libraryDependencies += "net.java.dev.jna" % "jna" % "4.2.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.0" % "test"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"

test in assembly := {}
