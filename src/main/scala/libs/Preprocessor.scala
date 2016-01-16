package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import scala.collection.mutable.ArrayBuffer

trait Preprocessor {
  def convert(name: String): Any => NDArray
}

class DefaultPreprocessor(schema: StructType) {
  def convert(name: String): Any => NDArray = {
    schema(name).dataType match {
      case FloatType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Float]), Array[Int](1))
      }
      case DoubleType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Double].toFloat), Array[Int](1))
      }
      case IntegerType => (element: Any) => {
        NDArray(Array[Float](element.asInstanceOf[Int].toFloat), Array[Int](1))
      }
    }
  }
}
