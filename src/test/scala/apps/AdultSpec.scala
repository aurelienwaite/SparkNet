import libs._

import org.scalatest._
import org.bytedeco.javacpp.caffe._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import java.nio.file.Paths

class AdultSpec extends FlatSpec {
  "LoadSqlDataset" should "be able to load the adult dataset" in {
    val conf = new SparkConf().setAppName("DataFrameLoaderTest").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sparkNetHome = sys.env("SPARKNET_HOME")

    val dataset = Paths.get(sparkNetHome, "data/adult.data").toString()
    val df = sqlContext.read.format("com.databricks.spark.csv").option("inferSchema", "true").load(dataset)
    val preprocessor = new DefaultPreprocessor(df.schema)

		// val model = sparkNetHome + "models/adult/adult.prototxt"
    val model = sparkNetHome + "models/adult/adult.prototxt"
    // val model = sparkNetHome + "/caffe/examples/cifar10/cifar10_full_train_test.prototxt"

		val netParam = new NetParameter();
		ReadProtoFromTextFileOrDie(model, netParam);
		val net = new JavaCPPCaffeNet(netParam, df.schema, preprocessor)

    val result = net.forward(df.take(64).iterator)
    for (i <- 0 to result.length - 1) {
      print(result(i).toString)
    }
    val rowRDD = sc.parallelize(result)
    val resultDF = sqlContext.createDataFrame(rowRDD, net.outputSchema())
    print("result", resultDF.take(1).deep.toString)
    print(resultDF.show())
	}
}
