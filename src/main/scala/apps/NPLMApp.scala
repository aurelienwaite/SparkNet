package apps

import java.io._
import java.net.URLDecoder

import caffe.Caffe._
import caffe.Caffe.SolverParameter.SolverMode.CPU

import com.sun.jna.Pointer
import libs._
import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg

import scala.annotation.tailrec


object NPLMApp {
  val trainBatchSize = 64
  val testBatchSize = 1
  val channels = 1
  val height = 1

  val workerStore = new WorkerStore()

  def buildSolverProto(snapShotPrefix: String)(netProto: NetParameter, lr: Float, partition: Int)  =
    SolverParameter.newBuilder
      .setNetParam(netProto)
      .setBaseLr(lr)
      .setLrPolicy("fixed")
      .setDisplay(1562)
      .setMaxIter(1562500)
      .setSnapshot(1000)
      .setSnapshotPrefix(s"${snapShotPrefix}_${partition}")
      .setSolverMode(CPU)
      .build()


  implicit val makeNPLMCallbacks = new MakeCallbacks[Array[Float]] {
    override def makeDataCallback(sizes : Sizes, minibatchSampler: MinibatchSampler[Array[Float], _ ],
                                  preprocessing: Option[(Array[Float], Array[Float]) => Unit] = None): CaffeLibrary.java_callback_t = {
      return new CaffeLibrary.java_callback_t() {
        import sizes._
        def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
          val currentBatch = minibatchSampler.nextImageMinibatch()
          assert(currentBatch.length == batchSize)
          val arrayShape = new Array[Int](numDims)
          for (i <- 0 to numDims - 1) {
            val dim = shape.getInt(i * intSize)
            arrayShape(i) = dim
          }
          val size = arrayShape.product
          for (j <- 0 to batchSize - 1) {
            val current = currentBatch(j)
            preprocessing.map {p =>
              val buffer = new Array[Float](size)
              p(current, buffer)
              data.write(j * size * dtypeSize, buffer, 0, size)
            } getOrElse {
              data.write(j * size * dtypeSize, current, 0, size)
            }

          }
        }
      }
    }

  }

  val nplmLevel = new Level(3500, "CaffeNPLM", 4){}
  val logger = Logger.getLogger(this.getClass)
  logger.setLevel(nplmLevel)
  def logNplm(toLog: String): Unit = logger.log(nplmLevel, toLog)

  case class Config(
                     numWorkers: Int = -1,
                     devSetFile: String = "",
                     trainSetFile: String = "",
                     netPrototext: File = new File("."),
                     syncInterval: Int = 1,
                     snapshotPrefix: String = "/tmp/",
                     sparkNetHomeOpt: Option[String] = CaffeNet.getSparkNetHome()
                   )

  def main(args: Array[String]) {
    val parser = new scopt.OptionParser[Config]("SparkNet NPLM") {
      head("SparkNet NPLM", "1.0")
      opt[Int]('w', "num_workers") required() valueName("number of workers") action { (x, c) =>
        c.copy(numWorkers = x)
      }
      opt[String]('t', "train") required() valueName ("training set of n-grams") action { (x, c) =>
        c.copy(trainSetFile = x)
      } text ("Training set is a required property")
      opt[String]('d', "dev") required() valueName ("development set of n-grams") action { (x, c) =>
        c.copy(devSetFile = x)
      } text ("Development set is a required property")
      opt[File]('n', "net_prototext") required() valueName ("Prototext for network") action { (x, c) =>
        c.copy(netPrototext = x)
      } text ("Net prototext is a required property")
      opt[String]('p', "snapshot_prefix") valueName ("Snapshot Prefix") action { (x, c) =>
        c.copy(snapshotPrefix = x)
      }
      opt[String]('s', "sparknet_home") valueName ("SparkNet Home") action { (x, c) =>
        c.copy(sparkNetHomeOpt = Option(x))
      }
      //opt[String]('o', "out") required() valueName ("output file") action { (x, c) =>
      //  c.copy(output = x)
      //} text ("Output file name is a required property")
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit(1))
    import cliConf._

    val conf = new SparkConf()
      .setAppName("CaffeNPLM")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val trainSet = sqlContext.read.parquet(trainSetFile)
    //trainSet.show()
    //trainSet.printSchema()

    val sparkNetHome = sparkNetHomeOpt getOrElse {
      val msg = "Cannot set SparkNet home"
      logger.log(Level.ERROR, msg)
      sys.error(msg)
    }
    logNplm(s"SparkNet home = $sparkNetHome")

    import org.apache.spark.sql.functions._
    val getSize = udf((features: linalg.Vector) => features.size)
    val ngramSizes = trainSet.select(getSize(trainSet("features"))).distinct().map(n => n.getAs[Int](0)).collect
    assert(ngramSizes.size == 1, sys.error("NGrams have different history lengths: " + ngramSizes.mkString(",")))
    val ngramSize= ngramSizes(0)
    logNplm(s"Using ngrams of order ${ngramSize + 1}")

    val coalesced = trainSet.repartition(numWorkers)

    logNplm("Creating minibatches")
    val minibatched = coalesced.mapPartitions{iter =>
      val stream = iter.toStream
      val toVecs = stream.map{ r  =>
        (r.getAs[linalg.Vector]("features"), r.getAs[Double]("label").toInt)
      }
      //Convert vectors to array representation, for fast array copy in the call backs
      val arrays = toVecs.map{ case (vec, label) =>
        (vec.toArray.map(_.toFloat), label)
      }
      val grouped = arrays.grouped(trainBatchSize)
      val batched = grouped.filter(_.size == trainBatchSize).map(_.toArray)
      batched.map(b => (b.map(_._1), b.map(_._2)))
    }.cache()

    val numTrainMinibatches = minibatched.count()
    logNplm(s"Number of minibatches = $numTrainMinibatches")

    val trainPartitionSizes = minibatched.mapPartitions(iter => Iterator(iter.size))
    val trainPartitionSizesString = trainPartitionSizes.collect().mkString(",")
    logNplm(s"Minibatches in partitions = $trainPartitionSizesString")



    val workers = sc.parallelize(0 until numWorkers, numWorkers)
    val solverBuilder = buildSolverProto(snapshotPrefix)_
    for(w <- workers) {
      logNplm(sys.props.get("jna.library.path").getOrElse("jna.library.path not set"))
      System.load(sparkNetHome + "/build/libccaffe.so")
      val caffeLib = CaffeLibrary.INSTANCE
      var netParameter = ProtoLoader.loadNetPrototxt(netPrototext.getAbsolutePath)
      netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, ngramSize)
      val solverParameter = solverBuilder(netParameter, 1.0f, w)
      val net = CaffeNet(caffeLib, solverParameter)
      workerStore.setNet("net", net)
    }


    @tailrec
    def iterate(netWeights : WeightCollection, i: Int, totalIterations: Int): WeightCollection= {
      logNplm("broadcasting weights")
      val broadcastWeights = sc.broadcast(netWeights)
      logNplm("setting weights on workers")
      workers.foreach(_ => workerStore.getNet("net").setWeights(broadcastWeights.value))

      /*if (i % 10 == 0) {
        log("testing, i")
        val testScores = testPartitionSizes.zipPartitions(testMinibatchRDD) (
          (lenIt, testMinibatchIt) => {
            assert(lenIt.hasNext && testMinibatchIt.hasNext)
            val len = lenIt.next
            assert(!lenIt.hasNext)
            val minibatchSampler = new MinibatchSampler(testMinibatchIt, len, len)
            workerStore.getNet("net").setTestData(minibatchSampler, len, None)
            Array(workerStore.getNet("net").test()).iterator // do testing
          }
        ).cache()
        val testScoresAggregate = testScores.reduce((a b) => (a, b).zipped.map(_ + _))
        val accuracies = testScoresAggregate.map(v => 100F * v / numTestMinibatches)
        log("%.2f".format(accuracies(0)) + "% accuracy", i)
      }*/

      logNplm("training")
      //

      trainPartitionSizes.zipPartitions(minibatched) (
        (lenIt, trainMinibatchIt) => {
          assert(lenIt.hasNext && trainMinibatchIt.hasNext)
          val len = lenIt.next
          assert(!lenIt.hasNext)
          val minibatchSampler = new MinibatchSampler[Array[Float], Int](trainMinibatchIt, len, syncInterval)
          workerStore.getNet("net").setTrainData(minibatchSampler, None)
          workerStore.getNet("net").train(syncInterval)
          Array(0).iterator
        }
      ).foreachPartition(_ => ())

      logNplm("collecting weights")
      val updatedWeights = workers.map(_ => { workerStore.getNet("net").getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      updatedWeights.scalarDivide(1F * numWorkers)
      if(i >= totalIterations)
        updatedWeights
      else
        iterate(updatedWeights, i+1, totalIterations)
    }

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.getNet("net").getWeights()).collect()(0)
    iterate(netWeights, 0, 100)
    logNplm("finished training")
  }
}
