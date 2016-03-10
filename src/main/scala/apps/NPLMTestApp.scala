package apps

import java.io.File

import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by rorywaite on 01/03/2016.
  */
object NPLMTestApp {

  def main(args: Array[String]) {

    import NPLMApp._

    val conf = new SparkConf()
      .setAppName("Test!")
      .setMaster("local[1]")
      .set("spark.driver.maxResultSize", "5G")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val data = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/models/lm.nplm-samples.s0.t5.v20000/train.parquet")
    val sampled = rowToArrays(data.sample(false, 0.1, 11l ).collect())
    for((f, l) <- sampled) {
      println(f.map(_.toInt).mkString(" ") +s" $l")
    }
    sys.exit()

    val devSet = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/models/lm.nplm-samples.s0.t5.v20000/dev.parquet")
    val asArrays = rowToArrays(devSet.collect())
    for((f, l) <- asArrays) {
      println(f.map(_.toInt).mkString(",") + s"\t$l" )
    }
    val solverBuilder = buildSolverProto(0, 0, asArrays.size) _
    initialiseCaffeLibrary("/misc/home/rwaite/mt-software/SparkNet", new File("/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/spark_net/nplm_prob.conf"), 4, solverBuilder)
    val testNet = workerStore.getNet("net")
    testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/good.caffemodel")
    computePerplexity(testNet, asArrays)
    val weights = testNet.getWeights()
    testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/spark_net/caffe_snapshot/sparknet_epoch_45")
    //testNet.setWeights(weights)
    computePerplexity(testNet, asArrays)
  }
}