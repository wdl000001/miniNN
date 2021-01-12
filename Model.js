function Model() {
  this.layers = [];
  this.output = null;
  this.loss = null;
  this.optimizer = null;
  this.learn_rate = 0.001;
  this.l2 = 0;
}


Model.prototype.init = function() {
  let { layers } = this;
  for (let i = 0; i < layers.length; i++) {
    let l = layers[i];
    if (l.isLayer && l.init != null) {
      l.init();
    }
  }
};

Model.prototype.addLayer = function(layer) {
  this.layers.push(layer);
};

Model.prototype.setLoss = function(loss) {
  this.loss = loss;
};

Model.prototype.setLearnRate = function(rate) {
  this.learn_rate = rate;
  let { layers } = this;
  for (let i = 0; i < layers.length; i++) {
    let l = layers[i];
    if (l.isLayer && l.learn_rate != null) {
      l.learn_rate = rate;
    }
  }
};

Model.prototype.setL2 = function(l2) {
  this.l2 = l2;
  let { layers } = this;
  for (let i = 0; i < layers.length; i++) {
    let l = layers[i];
    if (l.isLayer && l.l2 != null) {
      l.l2 = l2;
    }
  }
};

Model.prototype.front = function(X) {
  let { layers } = this;
  let zi = X;
  for (let i = 0; i < layers.length; i++) {
    let layer = layers[i];
    zi_1 = layer.func(zi);
    zi = zi_1;
  }
  return zi;
};

Model.prototype.train = function(arr) {
  let [X, Y] = arr;
  //前馈
  let { layers } = this;
  let zi = X;
  let numLayers = layers.length;
  for (let i = 0; i < numLayers; i++) {
    let layer = layers[i];
    zi = layer.func(zi);
  }
  let loss = this.loss.func(Y, zi);
  // console.log(Y.get([0]).toString())
  // console.log(zi.get([0]).toString())
  //反向
  //dloss/zi
  //loss层
  let gradient = this.loss.d_func(Y, zi); //梯度 [k, outdim]
  //layers 梯度回传
  for (let i = numLayers - 1; i >= 0; i--) {
    let layer = layers[i];
    gradient = layer.d_func(gradient);
  }
  return loss;
};

module.exports = Model;