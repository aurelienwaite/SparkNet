package com.sdl.caffe

import java.io._
import java.net.URLDecoder

import scala.concurrent.duration._
import com.sun.jna.Pointer
import com.sun.jna.Memory

case class Minibatch(data: Array[Float], label: Array[Float]) extends Serializable

class CaffeNet(libLocation: String, solverParam: Array[Byte]) extends Serializable {

  private class LibHolder {
    CaffeLibrary.Instance.dispose()
    val caffeLib = CaffeLibrary.Instance.get(libLocation)
    val state = caffeLib.create_state()
    val ptr = new Memory(solverParam.length)
    ptr.write(0, solverParam, 0, solverParam.length)
    caffeLib.load_solver_from_protobuf(state, ptr, solverParam.length)
    val numLayers = caffeLib.num_layers(state)
    val layerNames = List.range(0, numLayers).map(i => caffeLib.layer_name(state, i))
    val layerNumBlobs = List.range(0, numLayers).map(i => caffeLib.num_layer_weights(state, i))
    val dtypeSize = caffeLib.get_dtype_size()
    val intSize = caffeLib.get_int_size()
  }

  @transient private lazy val lib = new LibHolder

  import lib._

  // Ensures that the net's parameters are serialised across the Spark cluster
  private var weightCache = caffeToWeights

  trait AssignableMinibatches {
    var minibatches: Option[Iterator[Array[Float]]] = None;
  }

  def makeCallback() = new CaffeLibrary.java_callback_t() with AssignableMinibatches {
    def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
      val m = minibatches.getOrElse(sys.error("Minibatch not set!"))
      assert(m.hasNext, "Run out of minibatches")
      val buffer = m.next()
      data.write(0, buffer, 0, buffer.size)
    }
  }

  // Callbacks need to be referenced somewhere, otherwise they will be garbage collected before caffe can call them
  @transient lazy val dataCallback = makeCallback()
  @transient lazy val labelCallback = makeCallback()

  private def syncCacheToCaffe() = {
    val byteIn = new DataInputStream(new ByteArrayInputStream(weightCache))
    val cachedLayerLength = byteIn.readInt
    assert(cachedLayerLength == numLayers, s"Number of layers do not match $cachedLayerLength vs $numLayers")
    for (i <- 0 until numLayers) {
      val layerName = byteIn.readUTF()
      assert(layerName == layerNames(i), s"Layer names do not match $layerName vs ${layerNames(i)}")
      val cachedBlobLength = byteIn.readInt()
      assert(cachedBlobLength == layerNumBlobs(i), s"Different number of blobs $cachedBlobLength vs ${layerNumBlobs(i)}")
      for (j <- 0 until layerNumBlobs(i)) {
        val cachedShapeLength = byteIn.readInt
        val cachedShape = (for(_ <- 0 until cachedShapeLength) yield byteIn.readInt).toArray
        val caffeBlob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(caffeBlob)
        assert(shape.deep == cachedShape.deep, s"$layerName blob $j shapes do not match ${shape.deep} vs ${cachedShape.deep}")
        val blob_pointer = caffeLib.get_data(caffeBlob)
        val size = shape.product
        var t = 0
        while (t < size) {
          blob_pointer.setFloat(dtypeSize * t, byteIn.readFloat())
          t += 1
        }
      }
    }
    byteIn.close()
  }

  private def setData(in: Seq[Minibatch]) = {
    dataCallback.minibatches = Option(in.map(_.data).toIterator)
    labelCallback.minibatches = Option(in.map(_.label).toIterator)
  }

  private def caffeToWeights: Array[Byte]= {
    val byteOut = new ByteArrayOutputStream()
    val out = new DataOutputStream(byteOut)
    out.writeInt(numLayers)
    for (i <- 0 until numLayers) {
      out.writeUTF(layerNames(i))
      out.writeInt(layerNumBlobs(i))
      for (j <- 0 until layerNumBlobs(i))  {
        val blob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        out.writeInt(shape.length)
        for(s<-shape) out.writeInt(s)
        val blob_pointer = caffeLib.get_data(blob)
        val size = shape.product
        var t = 0
        while (t < size) {
          out.writeFloat(blob_pointer.getFloat(dtypeSize * t))
          t += 1
        }
      }
    }
    out.close()
    byteOut.toByteArray
  }

  /**
    * Returns true if all minibatches trained, otherwise returns false if
    * times out
    *
    * @param in Minibatchers
    * @param maxTime The maximum duration to spend training
    * @return
    */
  private def timedTrain(in: Seq[Minibatch], maxTime: Duration): (Boolean, Duration) = {
    val start = System.currentTimeMillis()
    for(_ <- in) {
      val elapsed = Duration(System.currentTimeMillis() - start, MILLISECONDS)
      if(elapsed < maxTime) caffeLib.solver_step(state, 1) else return (false, elapsed)
    }
    (true, Duration(System.currentTimeMillis() - start, MILLISECONDS))
  }

  def train(in: Seq[Minibatch], maxTime: Option[Duration]) = {
    caffeLib.set_mode_cpu()
    syncCacheToCaffe()
    setData(in)
    caffeLib.set_train_data_callback(state, 0, dataCallback)
    caffeLib.set_train_data_callback(state, 1, labelCallback)
    val t = maxTime.getOrElse(Duration.Inf)
    val allTrained = timedTrain(in, t);
    weightCache = caffeToWeights
    allTrained
  }

  private def testInit(numMinibatches: Int): Int = {
    caffeLib.set_mode_cpu()
    caffeLib.solver_test(state, numMinibatches) // you must run this before running caffeLib.num_test_scores(state)
    val numTestScores = caffeLib.num_test_scores(state)
    numTestScores
  }

  def test(toTest: Seq[Minibatch]): IndexedSeq[Float] = {
    syncCacheToCaffe()
    // Testing over a large number of minibatches causes caffe to freeze. Instead, iterate over a subset. The number
    // is set by testGrouping, which is just a sensible default
    val testGrouping = 10
    val res = for(group <- toTest.grouped(testGrouping)) yield {
      setData(group)
      caffeLib.set_test_data_callback(state, 0, dataCallback)
      caffeLib.set_test_data_callback(state, 1, labelCallback)
      val numTestScores = testInit(group.size)
      for (i <- 0 to numTestScores - 1) yield {
        caffeLib.get_test_score(state, i) // for softmax layers, this returns the sum of the averaged minibatch loss
      }
    }
    res.reduce{(scoresA, scoresB) =>
      for ((a,b) <- scoresA zip scoresB) yield a + b
    }
  }


  def setWeights(w: Array[Byte]) = {
    weightCache = w
  }

  def getWeights(): Array[Byte] = {
    weightCache
  }

  def loadWeightsFromFile(filename: String) {
    caffeLib.load_weights_from_file(state, filename)
    weightCache = caffeToWeights
  }

  def saveWeightsToFile(filename: String) {
    syncCacheToCaffe()
    caffeLib.save_weights_to_file(state, filename)
  }

  private def getShape(blob: Pointer): Array[Int] = {
    val numAxes = caffeLib.get_num_axes(blob)
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = caffeLib.get_axis_shape(blob, k)
    }
    return shape
  }

}

object CaffeNet {

  def getSparkNetHome(): Option[String]= {
    val env = sys.env.get("SPARKNET_HOME")
    if (env.isDefined) return env
    val path = classOf[CaffeNet].getProtectionDomain().getCodeSource().getLocation().getPath()
    val decodedPath = new File(URLDecoder.decode(path, "UTF-8"))
    for{
      p1 <- Option(decodedPath.getParentFile)
      p2 <- Option(p1.getParentFile)
      p3 <- Option(p2.getParent)
    } yield p3
  }

}
