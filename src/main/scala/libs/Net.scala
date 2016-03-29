package libs

import java.io.{Externalizable, File, ObjectInput, ObjectOutput}
import java.net.URLDecoder

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
  private var weightCache = syncWeights

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

  private def setData(in: Seq[Minibatch]) = {
    assert(weightCache.size == numLayers)
    for (i <- 0 to numLayers - 1) {
      val layer = weightCache.getOrElse(layerNames(i), sys.error(s"Caffe net does not contain layer ${layerNames(i)}"))
      assert(layer.length == layerNumBlobs(i), s"Different number of blobs ${layer.length} vs ${layerNumBlobs(i)}")
      for (j <- 0 to layerNumBlobs(i) - 1) {
        val blob = layer(j)
        val caffeBlob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(caffeBlob)
        assert(shape.deep == blob.shape.deep) // check that weights are the correct shape
        val blob_pointer = caffeLib.get_data(caffeBlob)
        val size = shape.product
        var t = 0
        while (t < size) {
          blob_pointer.setFloat(dtypeSize * t, blob.data(t))
          t += 1
        }
      }
    }
    dataCallback.minibatches = Option(in.map(_.data).toIterator)
    labelCallback.minibatches = Option(in.map(_.label).toIterator)
  }

  private def syncWeights = {
    val w = for (i <- 0 to numLayers - 1) yield {
      layerNames(i) -> (for (j <- 0 to layerNumBlobs(i) - 1) yield {
        val blob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        val data = new Array[Float](shape.product)
        val blob_pointer = caffeLib.get_data(blob)
        val size = shape.product
        var t = 0
        while (t < size) {
          data(t) = blob_pointer.getFloat(dtypeSize * t)
          t += 1
        }
        Blob(shape, data)
      })
    }
    w.toMap
  }



  def train(in: Seq[Minibatch]) = {
    caffeLib.set_mode_cpu()
    setData(in)
    caffeLib.set_train_data_callback(state, 0, dataCallback)
    caffeLib.set_train_data_callback(state, 1, labelCallback)
    caffeLib.solver_step(state, in.size)
    weightCache = syncWeights
  }

  private def testInit(numMinibatches: Int): Int = {
    caffeLib.set_mode_cpu()
    caffeLib.solver_test(state, numMinibatches) // you must run this before running caffeLib.num_test_scores(state)
    val numTestScores = caffeLib.num_test_scores(state)
    numTestScores
  }

  def test(toTest: Seq[Minibatch]): IndexedSeq[Float] = {
    setData(toTest)
    caffeLib.set_test_data_callback(state, 0, dataCallback)
    caffeLib.set_test_data_callback(state, 1, labelCallback)
    val numTestScores = testInit(toTest.size)
    for (i <- 0 to numTestScores - 1) yield {
      caffeLib.get_test_score(state, i) // for accuracy layers, this returns the average accuracy over a minibatch
    }
  }


  /*def forward() = {
    caffeLib.set_mode_cpu()
    caffeLib.forward(state)
  }

  def backward() = {
    caffeLib.set_mode_cpu()
    caffeLib.backward(state)
  }*/

  def setWeights(w: Weights) = {
    weightCache = w
  }

  def getWeights(): Weights = {
    weightCache
  }

  def loadWeightsFromFile(filename: String) {
    caffeLib.load_weights_from_file(state, filename)
  }

  def saveWeightsToFile(filename: String) {
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
