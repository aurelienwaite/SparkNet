package apps

import java.io._


import com.google.protobuf.TextFormat

import libs._
import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg
import org.apache.spark.sql.Row
import org.bytedeco.javacpp.caffe._


import scala.annotation.tailrec
import scala.util.Random


object NPLMApp {
  val trainBatchSize = 64
  val testBatchSize = 1
  val channels = 1
  val height = 1
  // After model parameters are computed, we need to redestribute and add them together so as not
  // to overwhelm the driver. This number tells the driver how many submodels to use
  val coalescedModels = 10
  // compute perplexity after set number of intervals
  val perplexityInterval = 10


  def rowToArrays(in: Seq[Row]) = {
    val toVecs = in.map { r =>
      (r.getAs[linalg.Vector]("features"), r.getAs[Double]("label").toInt)
    }
    //Convert vectors to array representation, for fast array copy in the call backs
    toVecs.map { case (vec, label) =>
      (vec.toArray.map(_.toFloat), label)
    }
  }

  def minibatchToBlobs(in: Seq[(Array[Float], Int)], minibatchSize: Int) = {
    val grouped = in.grouped(minibatchSize)
    val batched = grouped.filter(_.size == minibatchSize).map(_.toArray)
    for (b <- batched ) yield {
      val data = b.map(_._1).flatten.toArray
      val label = b.map(_._2.toFloat).toArray
      Array(data, label)
    }
  }

  def buildSolverProto(epoch: Int, lr: Float, testIter: Int)(net: NetParameter) = {
    val solverParam = new SolverParameter()
    solverParam.set_allocated_net_param(net)
    solverParam.set_base_lr(lr)
    solverParam.set_lr_policy("fixed")
    solverParam.set_display(epoch/100)
    solverParam.set_max_iter(epoch)
    solverParam.set_snapshot(Int.MaxValue)
    solverParam.set_snapshot_prefix("/tmp/sparknet")
    solverParam.set_solver_mode(Caffe.CPU)
    solverParam.add_test_iter(testIter)
    solverParam.set_test_interval(Int.MaxValue)
    solverParam
  }

  /*implicit val makeNPLMCallbacks = new MakeCallbacks[Array[Float]] {
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

  }*/

  val nplmLevel = new Level(3500, "CaffeNPLM", 4){}
  val logger = Logger.getLogger(this.getClass)
  logger.setLevel(nplmLevel)
  def logNplm(toLog: String): Unit = logger.log(nplmLevel, toLog)

  def initialiseCaffeLibrary(
                            netPrototext: File,
                            solverBuilder: NetParameter => SolverParameter
                            ) = {
    //logNplm(sys.props.get("jna.library.path").getOrElse("jna.library.path not set"))
    //val caffeLib = CaffeLibrary.Instance.get(sparkNetHome + "/build/libccaffe.so")
    //val caffeLib = CaffeLibrary.Instance.get()
    //logNplm("Caffe library loaded")
    val netParameter = new NetParameter()
    ReadProtoFromTextFileOrDie(netPrototext.getAbsolutePath, netParameter)
    val solverParameter = solverBuilder(netParameter)
    logNplm("Built solver")
    //logNplm(TextFormat.printToString(solverParameter))
    logNplm("Caffe network loaded")
    solverParameter
  }


  def computePerplexity(testNet: CaffeNet, devSet: Iterator[Array[Array[Float]]]) = {
    logNplm("Computing perplexity")
    val lossesIter = for(minibatch <- devSet) yield {
      testNet.setMinibatch(minibatch)
      val forwardOut = testNet.forward(List("loss")).getOrElse("loss", sys.error("Unable to retrive loss from the network")).toFlat()
      assert(forwardOut.length == 0, sys.error(s"Loss dimension is ${forwardOut.length}"))
      forwardOut(0)
    }
    val losses = lossesIter.toSeq
    val newPerperplexity = losses.reduce(_ +_ ) / losses.size
    logNplm(s"Perplexity: $newPerperplexity")
    newPerperplexity
  }

  case class Config(
                     numWorkers: Int = -1,
                     devSetFile: String = "",
                     trainSetFile: String = "",
                     netPrototext: File = new File("."),
                     syncInterval: Int = 50,
                     snapshotPrefix: String = "/tmp/",
                     startLearningRate: Float = 1.0f,
                     numEpochs: Int = 50,
                     samplePercentage: Option[Double] = None
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
      opt[File]('n', "net_prototext") required() valueName ("prototext for network") action { (x, c) =>
        c.copy(netPrototext = x)
      } text ("Net prototext is a required property")
      opt[String]('p', "snapshot_prefix") valueName ("snapshot prefix") action { (x, c) =>
        c.copy(snapshotPrefix = x)
      }
      //opt[String]('s', "sparknet_home") valueName ("sparknet home") action { (x, c) =>
      //  c.copy(sparkNetHomeOpt = Option(x))
      //}
      opt[Int]('y', "sync_interval") valueName("sync interval tau") action { (x, c) =>
        c.copy(syncInterval = x)
      }
      opt[Double]('a', "sample_percentage") valueName("percentage subset of the training data") action { (x, c) =>
        c.copy(samplePercentage = Option(x))
      }
      opt[Double]('l', "learning_rate") valueName("initial learning rate") action { (x, c) =>
        c.copy(startLearningRate = x.toFloat)
      }
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit(1))
    import cliConf._

    val conf = new SparkConf()
      .setAppName("CaffeNPLM")
      .set("spark.driver.maxResultSize", "15G")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryoserializer.buffer.max", "512m")
      .registerKryoClasses(Array(classOf[WeightCollection]))
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    /*val sparkNetHome = sparkNetHomeOpt getOrElse {
      val msg = "Cannot set SparkNet home"
      logger.log(Level.ERROR, msg)
      sys.error(msg)
    }
    logNplm(s"SparkNet home = $sparkNetHome")*/

    import org.apache.spark.sql.functions._

    // Prepare dev set
    val devSet = sqlContext.read.parquet(devSetFile)
    val asArrays = rowToArrays(devSet.collect())
    val devSetMinibatches = minibatchToBlobs(asArrays, testBatchSize)
    //val devMinibatches = asArrays.grouped(1).map(_.toArray).map(b => (b.map(_._1), b.map(_._2))).toSeq
    logNplm(s"Dev set contains ${asArrays.length} records")

    // Prepare training set
    val loadedTrainSet = sqlContext.read.parquet(trainSetFile)
    val trainSet = samplePercentage.map(loadedTrainSet.sample(false, _, 11l )).getOrElse(loadedTrainSet)
    val getSize = udf((features: linalg.Vector) => features.size)
    val ngramSizes = trainSet.select(getSize(trainSet("features"))).distinct().map(n => n.getAs[Int](0)).collect
    assert(ngramSizes.size == 1, sys.error("NGrams have different history lengths: " + ngramSizes.mkString(",")))
    val ngramSize= ngramSizes(0)
    logNplm(s"Using ngrams of order ${ngramSize + 1}")
    val coalesced = trainSet.repartition(numWorkers)
    //coalesced.show()

    //Initialise the driver caffelib for testing perplexities
    val solverBuilder = buildSolverProto(0, 0, asArrays.size) _
    @transient lazy val solverParam = initialiseCaffeLibrary(netPrototext, solverBuilder)

    trait Updatable{
      def update()
    }
    @transient lazy val solver = new FloatSGDSolver(solverParam) with Updatable{
      override def update() = super.ApplyUpdate()
    }
    @transient lazy val net = CaffeNet(solverParam.net_param)

    @tailrec
    def iterateEpoch(epochWeights: WeightCollection, epochCounter: Int, learningRate: Float, perplexity: Double): WeightCollection = {
      /*
       Spark does not sort order of RDDs. To create an ordering, we use a salted hash function. The salt varies by epoch
       which results for a different order by epoch
       */
      //val salt = Random.nextInt()
      //logNplm(s"Sorting data with salt $salt")
      //val getSaltedHash = udf((features: linalg.Vector, label: Int) => Vector((Seq(salt, label) ++ features.toArray.map(_.toInt)) :_* ).hashCode())
      //val resorted = coalesced.orderBy(getSaltedHash(trainSet("features"), trainSet("label"))).repartition(numWorkers)
      //logNplm(s"Reordered data contains ${resorted.count()} rows")
      //resorted.show

      logNplm(f"Begin epoch with learning rate $learningRate%.4f and perplexity $perplexity%.4f")
      logNplm("Creating minibatches")
      val minibatched = coalesced.mapPartitions { iter =>
        val shuffled = Random.shuffle(iter.toVector)
        val arrays = rowToArrays(shuffled)
        minibatchToBlobs(arrays, trainBatchSize)
      }.cache()


      //val numTrainMinibatches = minibatched.count()
      //logNplm(s"Number of minibatches = $numTrainMinibatches")

      // Partition sizes also includes worker ID
      val trainPartitionSizes = minibatched.mapPartitions(i => Iterator(i.size)).cache()
      val trainPartitionSizesCollected = trainPartitionSizes.collect()
      for ((size, i) <- trainPartitionSizesCollected.zipWithIndex)
        logNplm( s"Partition $i has $size minibatches")
      val epoch = trainPartitionSizesCollected.min
      val solverBuilder = buildSolverProto(epoch, learningRate, asArrays.size) _
      val numIterations = epoch / syncInterval

      @tailrec
      def iterate(netWeights: WeightCollection, iterationCounter: Int): WeightCollection = {
        val i = numIterations - iterationCounter
        if(i % perplexityInterval == 0 ) {
          net.setWeights(netWeights)
          val perplexity = computePerplexity(net, devSetMinibatches)
          logNplm(s"perplexity for iteration $i: $perplexity")
        }
        logNplm("broadcasting weights")
        val broadcastWeights = sc.broadcast(netWeights)
        logNplm("training")
        val trained = minibatched.mapPartitions { iter =>
          logNplm("setting weights on worker")
          net.setWeights(broadcastWeights.value)
          iter.drop(syncInterval * i )
          logNplm("running minibatches")
          val elapsed = time {
            for (_ <- 0 until syncInterval) {
              val minibatch = iter.next()
              net.setMinibatch(minibatch)
              net.forwardBackward()
              solver.update()
            }
          }
          Iterator((net.getWeights(), elapsed))
        }
        logNplm("collecting weights")
        val (updatedWeights, elapsed) = trained.repartition(coalescedModels).reduce{
          case ((aWeights, aElapsed), (bWeights, bElapsed)) => (WeightCollection.add(aWeights, bWeights), Math.max(aElapsed, bElapsed))
        }
        logNplm(s"caffe library completed in $elapsed seconds")
        updatedWeights.scalarDivide(1F * numWorkers)
        if (iterationCounter == 1)
          updatedWeights
        else
          iterate(updatedWeights, iterationCounter - 1)
      }
      val optimisedWeights = iterate(epochWeights, numIterations)
      trainPartitionSizes.unpersist()
      minibatched.unpersist()
      net.setWeights(optimisedWeights)
      val snapshotPath = s"$snapshotPrefix/sparknet_epoch_${numEpochs - epochCounter}"
      logNplm(s"saving weights to $snapshotPath")
      net.saveWeightsToFile(snapshotPath)
      val newPerplexity = computePerplexity(net, devSetMinibatches)

      val (updatedLr, updatedWeights, updatedPerplexity) = if(newPerplexity > perplexity){
        logNplm("Halving learning rate")
        (learningRate/2f, optimisedWeights, newPerplexity)
      }
      else
        (learningRate, optimisedWeights, newPerplexity)
      if(epochCounter == 1)
        return updatedWeights
      else
        iterateEpoch(updatedWeights, epochCounter -1, updatedLr, updatedPerplexity)
    }
    val startWeights = net.getWeights()
    iterateEpoch(startWeights, numEpochs, startLearningRate, Double.PositiveInfinity)
    logNplm("finished training")

  }
}
