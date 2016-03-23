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

    /*val data = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/models/lm.nplm-samples.s0.t5.v20000/train.parquet")
    val sampled = rowToArrays(data.sample(false, 0.1, 11l ).collect())
    for((f, l) <- sampled) {
      println(f.map(_.toInt).mkString(" ") +s" $l")
    }
    sys.exit()*/

    val devSet = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/models/lm.nplm-samples.s0.t5.v20000/dev.parquet")
    val minibatches = makeMinibatches(devSet.collect(),1).toSeq

    val trainBatches = makeMinibatches(devSet.collect(), 64).toSet
    val solverBuilder = buildSolverProto(0, 0, minibatches.size) _
    val testNet = initialiseCaffeLibrary("/misc/home/rwaite/mt-software/SparkNet", new File("/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/spark_net/nplm_prob.conf"), 4, solverBuilder)
    val randomWeights = testNet.getWeights()
    testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/good.caffemodel")
    println("weights loaded")
    computePerplexity(testNet, minibatches)
    val weights = testNet.getWeights()
    //testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/spark_net/caffe_snapshot/sparknet_epoch_45")
    testNet.setWeights(randomWeights)
    computePerplexity(testNet, minibatches)
    //testNet.setWeights(weights)
    //computePerplexity(testNet, minibatches)
    testNet.train(trainBatches.take(50).toSeq)
    val trained = testNet.getWeights()
    testNet.setWeights(trained)
    computePerplexity(testNet, minibatches)
  }
}
