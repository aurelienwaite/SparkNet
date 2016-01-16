package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

trait NetInterface {
  def forward(rowIt: Iterator[Row]): Array[Row]
  def forwardBackward(rowIt: Iterator[Row])
  def getWeights(): WeightCollection
  def setWeights(weights: WeightCollection)
}

class JavaCPPCaffeNet(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor) {
  private val inputSize = netParam.input_size
  private val batchSize = netParam.input_shape(0).dim(0).toInt
  private val transformations = new Array[Any => NDArray](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames = schema.map(entry => entry.name)
  private val caffeNet = new FloatNet(netParam)

  for (i <- 0 to inputSize - 1) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name)
    inputIndices(i) = columnNames.indexOf(name)
  }

  // Preallocate a buffer for data input into the net
  val inputs = new FloatBlobVector(inputSize)
  for (i <- 0 to inputSize - 1) {
    val dims = new Array[Int](netParam.input_shape(i).dim_size)
    for (j <- dims.indices) {
      dims(j) = netParam.input_shape(i).dim(j).toInt
    }
    inputs.put(i, new FloatBlob(dims))
  }

  def transformInto(iterator: Iterator[Row], data: FloatBlobVector): Unit = {
    var batchIndex = 0
    while (iterator.hasNext) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        val result = transformations(i)(row(inputIndices(i)))
        val flatArray = result.toFlat() // TODO: Make this efficient
        val blob = data.get(i)
        val buffer = blob.cpu_data()
        val offset = blob.offset(batchIndex)
        buffer.position(offset)
        buffer.put(flatArray: _*)
      }
      batchIndex += 1
      if (batchIndex == batchSize) {
        return
      }
    }
  }

  def forward(rowIt: Iterator[Row]): Array[Row] = {
    val result = new Array[Array[Float]](batchSize)
    transformInto(rowIt, inputs)
    val tops = caffeNet.Forward(inputs)
    for (i <- 0 to batchSize) {
      val top = tops.get(i).cpu_data()
      val array = Array[Float](top.position())
      top.get(result(i))
    }
    return result.map(row => Row(row))
  }

  def forwardBackward(rowIt: Iterator[Row]) = {
    transformInto(rowIt, inputs)
    caffeNet.ForwardBackward(inputs)
  }
}
