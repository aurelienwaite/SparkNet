/**
  * Created by rorywaite on 16/03/2016.
  */
package object libs {

  def time(f: => Unit) = {
    val s = System.currentTimeMillis
    f
    (System.currentTimeMillis - s) / 1000
  }

}
