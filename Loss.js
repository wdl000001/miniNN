const NArray = require('./NArray');

function BaseLoss() {
  this.func = null;
  this.d_func = null;
}

function MSE() {
  BaseLoss.call(this);
  //0.5*(y_true - y_pred)^2 / m
  this.func = function(y_true, y_pred) {
    let delta = y_true.sub(y_pred, null, true);
    delta.mapElements(x => x * x, delta);
    let arr = delta.array;
    let s = 0;
    for (let i = 0; i < arr.length; i++) {
      s += arr[i];
    }
    return s * 0.5 / y_pred.shape[0];
  };

  this.d_func = function(y_true, y_pred) {
    let delta = y_pred.sub(y_true, null, true);
    return delta;
  };
}

function CrossEntropy() {
  BaseLoss.call(this);
  // Σ -y_t*log(y_p)
  this.func = function(y_true, y_pred) {
    let log = y_pred.mapElements(x => Math.log(x + 1e-8));
    this.log = log;
    let mul = log.mul(y_true);
    let s = 0;
    let arr = mul.array;
    for (let i = 0; i < arr.length; i++) {
      s += arr[i];
    }
    return -s / y_true.shape[0];
  };
  this.d_func = function(y_true, y_pred) {
    //-yt/yp
    let out = y_true.scale(-1, null, true);
    out.div(y_pred, out);
    return out;
  };
}

module.exports = {
  BaseLoss,
  MSE,
  CrossEntropy,
};

if (require.main === module) {
  let mse = new MSE();
  let a = new NArray([8, 10]);
  for (let i = 0; i < 8; i++) {
    let ix = Math.floor(Math.random() * 10);
    a.set([i, ix], 1);
  }
  let b = new NArray([8, 10]);
  for (let i = 0; i < b.array.length; i++) {
    b.array[i] = Math.random();
  }
  console.log(a.toString());
  console.log(b.toString());
  console.log('----------------------');
  let loss = mse.func(a, b);
  console.log(loss);
  let delta = mse.d_func(a, b);
  console.log(delta.toString());
  //
  console.log('-----------');
  let cross = new CrossEntropy();

  loss = cross.func(a, b);
  console.log(loss);
  delta = cross.d_func(a, b);
  console.log(delta.toString());
}