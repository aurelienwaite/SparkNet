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
  private val transformations = new Array[Any => NDArray](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames = schema.map(entry => entry.name)

  for (i <- 0 to inputSize) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name)
    inputIndices(i) = columnNames.indexOf(name)
  }

  def transformInto(iterator: Iterator[Row], data: FloatBlobVector) = {
    var batchIndex = 0
    while (iterator.hasNext) {
      val row = iterator.next
      for (i <- 0 to inputSize) {
        val result = transformations(i)(row(inputIndices(i)))
        val flatArray = result.toFlat() // TODO: Make this efficient
        val blob = data.get(i)
        val buffer = blob.cpu_data()
        val offset = blob.offset(batchIndex)
        buffer.position(offset)
        buffer.put(flatArray: _*)
      }
      batchIndex += 1
    }
  }
}
