/**
  * Created by rorywaite on 23/03/2016.
  */
package object libs {

  case class Blob(shape: Array[Int], data: Array[Float]) {
    override def toString() = {
      "Shape: " + shape.mkString(",") + "; Data: " + data.mkString(",")
    }
  }

  type Weights = Map[String, scala.collection.immutable.IndexedSeq[Blob]]
}
