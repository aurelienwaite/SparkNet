/**
  * Created by rorywaite on 23/03/2016.
  */
package object libs {

  case class Blob(shape: Array[Int], data: Array[Float])

  type Weights = Map[String, scala.collection.immutable.IndexedSeq[Blob]]
}
