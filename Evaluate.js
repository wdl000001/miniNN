function ClassificationAccuracy(y_true, y_pred) {
  let k = y_true.shape[0];
  let right = 0;
  for (let i = 0; i < k; i++) {
    let p = y_pred.maxArgIx([i]);
    let t = y_true.maxArgIx([i]);
    if (p == t) {
      right++;
    }
  }
  return right / k;
}

module.exports = {
  ClassificationAccuracy
}
