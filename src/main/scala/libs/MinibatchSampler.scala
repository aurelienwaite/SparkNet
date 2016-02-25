package libs

class MinibatchSampler[DATA, LABEL](minibatchIt: Iterator[(Array[DATA], Array[LABEL])],
                                    totalNumBatches: Int, numSampledBatches: Int, startPointer: Option[Int] = None) {
  // The purpose of this method is to take minibatchIt, which is an iterator
  // over images and labels, and to turn it into two iterators, one over images
  // and one over labels. The iterator over images is used to create a callback
  // that Caffe uses to get the next minibatch of images. The iterator over
  // labels is used to create a callback that Caffe uses to get the next
  // minibatch of labels. We cannot use the same iterator for both purposes
  // because incrementing the iterator in one callback will increment it in the
  // other callback as well, since they are the same object.

  // totalNumBatches = minibatchIt.length (but we can't call minibatchIt.length because that would consume the entire iterator)
  // numSampledBatches is the number of minibatches that we subsample from minibatchIt

  var it = minibatchIt // we need to update the iterator by calling it.drop, and we need it to be a var to do this
  val r = scala.util.Random
  val startMapped = startPointer.map(_ * numSampledBatches)
  val startIdx = if (totalNumBatches == numSampledBatches)
    0
  else
    startMapped.getOrElse(r.nextInt(totalNumBatches - numSampledBatches + 1))
  val indices = Array.range(startIdx, startIdx + numSampledBatches)
  var indicesIndex = 0
  var currMinibatchPosition = -1

  var currMinibatchImages = None: Option[IndexedSeq[DATA]]
  var currMinibatchLabels = None: Option[IndexedSeq[LABEL]]

  private def nextMinibatch() = {
    it = it.drop(indices(indicesIndex) - currMinibatchPosition - 1)
    currMinibatchPosition = indices(indicesIndex)
    indicesIndex += 1
    assert(it.hasNext)
    val (images, labels) = it.next
    currMinibatchImages = Option(images)
    currMinibatchLabels = Option(labels)
  }

  def nextImageMinibatch(): IndexedSeq[DATA] = {
    if (currMinibatchImages.isEmpty) {
      nextMinibatch()
      return currMinibatchImages.get
    } else {
      val images = currMinibatchImages.get
      currMinibatchImages = None
      currMinibatchLabels = None
      return images
    }
  }

  def nextLabelMinibatch(): IndexedSeq[LABEL] = {
    if (currMinibatchLabels.isEmpty) {
      nextMinibatch()
      return currMinibatchLabels.get
    } else {
      val labels = currMinibatchLabels.get
      currMinibatchImages = None
      currMinibatchLabels = None
      return labels
    }
  }

}
