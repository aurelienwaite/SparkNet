package libs

/**
  * Created by rorywaite on 09/02/2016.
  */
object Perplexity {

  def compute(labels: Seq[Int], net: Net, features: Option[Seq[Array[Float]]] = None) = {
    var i=0
    val logLiklihood =
      (for (f<-features.toTraversable; (in, l) <- f zip labels) yield {
        val prob = net.test(l)
        i+=1
        Math.log(prob)
      }).reduce(_ + _)
    Math.exp(-1 * logLiklihood / labels.size)
  }

}
